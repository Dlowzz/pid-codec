"""Recalibrate shared main/side rate levels and preserve the training cursor.

Each level uses fixed per-channel scales and scalar rate multipliers. A shared
multiplier or separate main/side multipliers with a target bit split is supported.
Side features always come from the selected quantized main reconstruction;
shared-multiplier search recomputes them for every candidate. Only the two fixed
quantizer buffers change in the checkpoint.
This is an explicit training transition, not an exact resume of the old rates.
"""
import argparse
import gc
import hashlib
import json
import torch
from torch.utils.data import DataLoader, Subset

from .data import Images, fingerprint
from .models import load_models
from .quantization import MultiRateQuantizer, SymbolEntropy, estimate_bpp
from .smoke import same
from .train import (alias, atomic_save, load_config, resolved, setup,
                    training_identity, validate)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--output-config', required=True)
    parser.add_argument('--targets', type=float, nargs='+', default=[.006, .012, .025, .05, .1])
    parser.add_argument('--main-fraction', type=float,
                        help='Allocate this fraction of total estimated bpp to main symbols (e.g. 0.6666666667)')
    parser.add_argument('--images', type=int, default=32)
    parser.add_argument('--offset', type=int, default=64)
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()
    cfg = load_config(args.config)
    source = resolved(args.checkpoint)
    out = resolved(args.output_dir)
    config_path = resolved(args.output_config)
    if config_path.exists() or (out / 'calibration').exists():
        raise FileExistsError('Calibration outputs are immutable; use a new run directory')
    assert all(a < b for a, b in zip(args.targets, args.targets[1:]))
    assert args.targets and args.targets[0] > 0
    assert args.main_fraction is None or 0 < args.main_fraction < 1
    saved = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    data_hash = fingerprint(resolved(cfg['manifest_dir']))
    assert saved['stage'] == '2C' and not saved['debug'] and not saved['stage_complete']
    assert saved['identity'] == training_identity(cfg, '2C', data_hash)
    assert saved['data_fingerprint'] == data_hash
    assert cfg['stage_2c_joint_finetune'] and cfg['side_branch']['channels'] == 8
    assert saved['batch']['world'] == 1, 'This transition preserves the current single-GPU RNG state'
    calibration = out / 'calibration'
    calibration.mkdir(parents=True)
    # Pin a hard link so a later source-run retention policy cannot remove it.
    pinned = out / f"source_step{saved['stage_step']:06d}.pt"
    if pinned.exists():
        assert pinned.samefile(source)
    else:
        pinned.hardlink_to(source)
    source_hash = file_hash(pinned)
    setup(cfg['seed'])
    torch.set_grad_enabled(False)
    codec, _, decoder = load_models(cfg, '2C', checkpoint_state=saved)
    codec.eval().requires_grad_(False)
    decoder.eval().requires_grad_(False)
    dataset = Images(resolved(cfg['manifest_dir']), 'val', cfg['seed'])
    indices = list(range(args.offset, args.offset + args.images))
    assert indices[-1] < len(dataset) and args.offset >= cfg['validation_images']
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch,
                        num_workers=cfg['workers'], pin_memory=True)
    cache, initial_side = [], []
    midpoint = codec.quantizer.num_rates // 2
    for i, x in enumerate(loader):
        x = x.cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z = codec.encode(x)
            fx = decoder.full_vae(x.float())
            base = decoder(codec.quantizer(z, midpoint))
            fb = decoder.full_vae(base.float())
            side = decoder.side_branch.query(fx, fb)
        cache.append((z.cpu(), fx.cpu()))
        initial_side.append(side.float().cpu())
        print(json.dumps(dict(phase='collect', batch=i + 1, batches=len(loader))), flush=True)
    main_std = torch.cat([z.float() for z, _ in cache]).std((0, 2, 3), correction=0).clamp_min(1e-6)
    side_std = torch.cat(initial_side).std((0, 2, 3), correction=0).clamp_min(1e-6)
    channel_std = torch.cat([main_std, side_std])
    del initial_side, x, z, fx, base, fb, side
    observations = {}
    side_cache = {}

    def collect_side(main_multiplier):
        if main_multiplier in side_cache:
            return side_cache[main_multiplier]
        main_q = MultiRateQuantizer((main_std * main_multiplier)[None]).cuda()
        entropy = SymbolEntropy()
        chunks = []
        for z, fx in cache:
            z, fx = z.cuda(), fx.cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                zhat, symbols, _ = main_q(z, 0, return_symbols=True)
                base = decoder(zhat)
                fb = decoder.full_vae(base.float())
                side = decoder.side_branch.query(fx, fb)
            entropy.update(symbols)
            chunks.append(side.float().cpu())
        result = (torch.cat(chunks), entropy.result()['estimated_bpp'])
        side_cache[main_multiplier] = result
        return result

    def measure(multiplier):
        if multiplier in observations:
            return observations[multiplier]
        features, main = collect_side(multiplier)
        side = estimate_bpp(features.numpy(), (side_std * multiplier).numpy())['estimated_bpp']
        total = main + side
        result = dict(multiplier=multiplier, main_estimated_bpp=main,
                      side_estimated_bpp=side, total_estimated_bpp=total)
        observations[multiplier] = result
        print(json.dumps(dict(phase='candidate', **result)), flush=True)
        return result

    def fit_multiplier(features, std, target):
        low, high, best = -12., 12., None
        array = features.float().numpy()
        for _ in range(32):
            mid = (low + high) / 2
            multiplier = 2**mid
            bpp = estimate_bpp(array, (std * multiplier).numpy())['estimated_bpp']
            error = abs(bpp - target)
            if best is None or error < best[0]:
                best = (error, multiplier, bpp)
            if bpp > target:
                low = mid
            else:
                high = mid
        assert best[0] < max(.0001, target * .01), (target, best)
        return best[1:]

    reports = []
    for rate, target in enumerate(args.targets):
        if args.main_fraction is not None:
            target_main = target * args.main_fraction
            target_side = target - target_main
            main_multiplier, main_bpp = fit_multiplier(torch.cat([z for z, _ in cache]), main_std, target_main)
            features, measured_main = collect_side(main_multiplier)
            assert abs(main_bpp - measured_main) < .00005
            main_bpp = measured_main  # Report the actual CUDA quantizer symbols.
            side_multiplier, side_bpp = fit_multiplier(features, side_std, target_side)
            reports.append(dict(rate=rate, target_total_bpp=target, target_main_bpp=target_main,
                                target_side_bpp=target_side, main_multiplier=main_multiplier,
                                side_multiplier=side_multiplier, main_estimated_bpp=main_bpp,
                                side_estimated_bpp=side_bpp, total_estimated_bpp=main_bpp + side_bpp))
            print(json.dumps(dict(phase='selected', **reports[-1])), flush=True)
            continue
        low, high, best = -6., 6., None
        assert measure(2**low)['total_estimated_bpp'] > target
        assert measure(2**high)['total_estimated_bpp'] < target
        for _ in range(14):
            mid = (low + high) / 2
            row = measure(2**mid)
            error = abs(row['total_estimated_bpp'] - target)
            if best is None or error < best[0]:
                best = (error, row)
            if error < max(.0001, target * .005):
                break
            if row['total_estimated_bpp'] > target:
                low = mid
            else:
                high = mid
        assert best[0] < max(.0002, target * .015), (target, best)
        reports.append(dict(rate=rate, target_total_bpp=target, main_multiplier=best[1]['multiplier'],
                            side_multiplier=best[1]['multiplier'], **best[1]))
        print(json.dumps(dict(phase='selected', **reports[-1])), flush=True)
    main_multipliers = torch.tensor([r['main_multiplier'] for r in reports])
    side_multipliers = torch.tensor([r['side_multiplier'] for r in reports])
    deltas = torch.cat([main_multipliers[:, None] * main_std[None],
                        side_multipliers[:, None] * side_std[None]], dim=1)
    assert (torch.diff(deltas, dim=0) < 0).all()
    common = dict(source_checkpoint=str(pinned), source_sha256=source_hash,
                  source_stage_step=saved['stage_step'], source_identity=saved['identity'],
                  data_fingerprint=data_hash, calibration_split='val', indices=indices,
                  channel_std=channel_std.tolist(), main_multipliers=main_multipliers.tolist(),
                  side_multipliers=side_multipliers.tolist(), target_main_fraction=args.main_fraction,
                  reports=reports, fixed=True, shape=[24, 16, 16],
                  formula='delta[r,c] = stream_multiplier[r] * fixed_channel_std[c]; one shared rate ID',
                  side_scale_reference_main_rate=midpoint,
                  protocol='Joint per-channel zero-order empirical entropy; main + side symbols only, no entropy coder, hyperlatent or headers. Not actual bitstream bpp.')
    write_json(calibration / 'joint.json', dict(common, deltas=deltas.tolist(),
                                              candidates=list(observations.values())))
    for name, bounds in [('main', slice(0, 16)), ('side', slice(16, 24))]:
        write_json(calibration / f'{name}.json', dict(common, deltas=deltas[:, bounds].tolist(),
                                                   channel_indices=list(range(24))[bounds]))
    cfg = dict(cfg, output_dir=str(out),
               quantization_anchors=str(calibration / 'main.json'),
               side_quantization_anchors=str(calibration / 'side.json'),
               rate_targets=[r.get('target_main_bpp', r['main_estimated_bpp']) for r in reports],
               side_rate_targets=[r.get('target_side_bpp', r['side_estimated_bpp']) for r in reports],
               total_rate_targets=args.targets, joint_rate_calibration=str(calibration / 'joint.json'))
    encoder_state = dict(saved['encoder'], **{'quantizer.deltas': deltas[:, :16].contiguous()})
    side_state = dict(saved['side_branch'], **{'quantizer.deltas': deltas[:, 16:].contiguous()})
    codec.quantizer = MultiRateQuantizer(deltas[:, :16]).cuda()
    decoder.side_branch.quantizer = MultiRateQuantizer(deltas[:, 16:], channels=8).cuda()
    seed = {k: saved[k] for k in ['format', 'stage', 'stage_step', 'global_optimizer_step',
            'sample_cursor', 'data_fingerprint', 'identity', 'debug', 'pid_lora',
            'pid_lora_config', 'pid_base_weights', 'side_config', 'side_parameter_counts']}
    seed.update(encoder=encoder_state, side_branch=side_state, config=dict(cfg),
                checkpoint_kind='weights_only_quantization_bootstrap', quantization_transition=common)
    bootstrap = out / 'initial_requantized.pt'
    atomic_save(bootstrap, seed)
    cfg.update(initial_checkpoint=str(bootstrap), initial_checkpoint_sha256=file_hash(bootstrap))
    config_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(config_path, cfg)
    del cache, observations, side_cache, seed
    gc.collect()
    torch.cuda.empty_cache()
    stage_out = out / 'stage_2C'
    stage_out.mkdir()
    print('Validating all recalibrated levels on the separate fixed validation subset', flush=True)
    result = validate(codec, decoder, cfg, '2C', saved['stage_step'], stage_out)
    resumed = dict(saved, encoder=encoder_state, side_branch=side_state, config=cfg,
                   identity=training_identity(cfg, '2C', data_hash), best=result['selection_loss'],
                   transition=dict(kind='joint_quantization_recalibration', source_checkpoint=str(pinned),
                                   source_sha256=source_hash, source_identity=saved['identity'],
                                   previous=saved.get('transition'), calibration=str(calibration / 'joint.json'),
                                   optimizer_preserved=True, schedule_preserved=True, rng_and_cursor_preserved=True))
    path = stage_out / f"step_{saved['stage_step']:06d}.pt"
    atomic_save(path, resumed)
    alias(path, stage_out / 'latest.pt')
    alias(path, stage_out / 'best.pt')
    restored = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    preserved = ['optimizer', 'scheduler', 'rng_by_rank', 'sample_cursor', 'batch',
                 'global_optimizer_step', 'stage_step', 'pid_lora', 'discriminator',
                 'discriminator_optimizer', 'discriminator_scheduler']
    for key in preserved:
        same(saved[key], restored[key], key)
    for key in ['encoder', 'side_branch']:
        same({k: v for k, v in saved[key].items() if k != 'quantizer.deltas'},
             {k: v for k, v in restored[key].items() if k != 'quantizer.deltas'}, key)
    write_json(out / 'transition_verified.json', dict(passed=True, source_checkpoint=str(pinned),
               source_sha256=source_hash, checkpoint=str(path), source_step=saved['stage_step'],
               preserved_fields=preserved, all_trainable_weights_unchanged=True,
               changed_model_buffers=['encoder.quantizer.deltas', 'side_branch.quantizer.deltas'],
               validation= str(stage_out / f"validation_{saved['stage_step']:06d}.json")))
    print(json.dumps(dict(complete=True, config=str(config_path), checkpoint=str(path), rates=reports)), flush=True)


if __name__ == '__main__':
    main()
