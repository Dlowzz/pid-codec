"""Calibrate fixed six-rate side steps on the current query features."""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import Images, fingerprint
from .models import load_models
from .quantization import SymbolEntropy, estimate_bpp
from .train import load_config, resolved, setup


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/pid_codec/stage2c_joint.json')
    args = parser.parse_args()
    cfg = load_config(args.config)
    dest = resolved(cfg['side_quantization_anchors'])
    if dest.exists():
        raise FileExistsError('Side anchors are immutable; use a new path for recalibration')
    setup(cfg['seed'])
    torch.set_grad_enabled(False)
    source_cfg = dict(cfg)
    source_cfg.pop('side_quantization_anchors')
    codec, _, decoder = load_models(source_cfg, '2C')
    codec.eval().requires_grad_(False)
    decoder.eval().requires_grad_(False)
    data_hash = fingerprint(resolved(cfg['manifest_dir']))
    assert codec.initialization['data_fingerprint'] == data_hash
    rates = codec.quantizer.num_rates
    channels = cfg['side_branch']['channels']
    assert rates == len(cfg['side_rate_targets']) == 6

    def collect(split, indices):
        data = Subset(Images(resolved(cfg['manifest_dir']), split, cfg['seed']), indices)
        loader = DataLoader(data, batch_size=cfg['side_calibration_batch'],
                            num_workers=cfg['workers'], pin_memory=True)
        chunks = [[] for _ in range(rates)]
        entropy = [SymbolEntropy() for _ in range(rates)]
        for index, images in enumerate(loader):
            x = images.cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                # Encode the image once; the side distribution depends on main rate.
                z = codec.encode(x)
                fx = decoder.full_vae(x)
                for rate in range(rates):
                    zhat, symbols, _ = codec.quantizer(z, rate, return_symbols=True)
                    base = decoder(zhat)
                    fb = decoder.full_vae(base.float())
                    side = decoder.side_branch.query(fx, fb)
                    chunks[rate].append(side.float().cpu().numpy())
                    entropy[rate].update(symbols)
            print(json.dumps(dict(split=split, batch=index + 1, batches=len(loader))), flush=True)
        return [np.concatenate(parts) for parts in chunks], [stat.result() for stat in entropy]

    count = len(Images(resolved(cfg['manifest_dir']), 'train', cfg['seed']))
    train_ids = torch.randperm(count, generator=torch.Generator().manual_seed(cfg['seed']))[:cfg['side_calibration_train_images']].tolist()
    offset = cfg['calibration_val_offset']
    val_ids = list(range(offset, offset + cfg['side_calibration_val_images']))
    train, _ = collect('train', train_ids)
    std = np.concatenate(train).transpose(1, 0, 2, 3).reshape(channels, -1).std(1).clip(1e-6)
    val, main_rates = collect('val', val_ids)
    deltas = []
    reports = []
    for rate, target in enumerate(cfg['side_rate_targets']):
        low, high, best = -12., 12., None
        for _ in range(32):
            mid = (low + high) / 2
            delta = std * 2**mid
            stats = estimate_bpp(val[rate], delta)
            error = abs(stats['estimated_bpp'] - target)
            if best is None or error < best[0]:
                best = (error, delta, stats, 2**mid)
            if stats['estimated_bpp'] > target:
                low = mid
            else:
                high = mid
        assert best[0] < .0002, (rate, target, best[2])
        deltas.append(best[1].tolist())
        row = dict(rate=rate, target_side_bpp=target, multiplier=best[3],
                   main_estimated_bpp=main_rates[rate]['estimated_bpp'],
                   side_estimated_bpp=best[2]['estimated_bpp'],
                   total_estimated_bpp=main_rates[rate]['estimated_bpp'] + best[2]['estimated_bpp'],
                   side_entropy=best[2])
        reports.append(row)
        print(json.dumps(row), flush=True)
    assert (np.diff(np.asarray(deltas), axis=0) < 0).all(), 'Rate steps must decrease from coarse to fine'
    result = dict(deltas=deltas, channel_std=std.tolist(), reports=reports,
                  train_indices=train_ids, val_indices=val_ids,
                  source_checkpoint=cfg['initial_checkpoint'], source_sha256=cfg['initial_checkpoint_sha256'],
                  source_stage_step=codec.initialization['source_stage_step'],
                  source_identity=codec.initialization['source_identity'], data_fingerprint=data_hash,
                  fixed=True, channels=channels, rate_pairing='same rate ID for main and side',
                  protocol='Per-channel hard round + STE. Estimated main + side entropy only; no coder/header cost.')
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2) + '\n')
    print('Saved', dest, flush=True)


if __name__ == '__main__':
    main()
