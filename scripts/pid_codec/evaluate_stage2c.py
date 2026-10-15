"""Kodak evaluation of the existing complete Stage 2C reconstruction path."""
import argparse
import csv
import gc
import hashlib
import html
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim

from .models import JointCodec, ROOT, load_models
from .quantization import SymbolEntropy
from .train import make_perceptual, setup


NAMES = [f'kodim{i:02d}.png' for i in range(1, 25)]
EXAMPLES = {'kodim01.png', 'kodim05.png', 'kodim13.png', 'kodim23.png'}
BRANCHES = ('base', 'conditioned', 'final')
METRICS = ('psnr', 'ssim', 'ms_ssim', 'lpips_vgg', 'dists', 'mse')


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rgb8(tensor):
    return ((tensor.float().clamp(-1, 1) + 1) * 127.5).round().byte()


def picture(tensor):
    return Image.fromarray(tensor[0].permute(1, 2, 0).cpu().numpy())


def paired(original, reconstructed, metric):
    x, y = original.float() / 255, reconstructed.float() / 255
    mse = F.mse_loss(y, x).item()
    result = dict(
        psnr=-10 * math.log10(max(mse, 1e-12)), mse=mse,
        ssim=ssim(x, y, data_range=1., win_size=11, win_sigma=1.5).item(),
        ms_ssim=ms_ssim(x, y, data_range=1.).item(),
        lpips_vgg=metric(y * 2 - 1, x * 2 - 1).mean().item(),
        dists=metric.dists_loss(y * 2 - 1, x * 2 - 1).mean().item(),
    )
    if not all(math.isfinite(v) for v in result.values()):
        raise FloatingPointError(result)
    return result


@torch.no_grad()
def reconstruct(args):
    if (args.output / 'metrics.json').exists():
        raise FileExistsError('Completed reconstruction exists; use --phase distribution or report')
    paths = sorted(args.input.glob('*.png'))
    if [p.name for p in paths] != NAMES:
        raise ValueError('Expected exactly Kodak kodim01.png through kodim24.png')
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    assert saved['stage'] == '2C' and not saved['debug']
    cfg = saved['config']
    # Inference checks model/base-asset compatibility below, not optimizer identity.
    assert saved['pid_base_weights'] == json.loads((ROOT / 'configs/pid_weights.json').read_text())
    setup(cfg['seed'])
    # setup enables gradients for training; this evaluator remains inference-only.
    torch.set_grad_enabled(False)
    codec, _, decoder = load_models(cfg, '2C', checkpoint_state=saved)
    joint = JointCodec(codec, decoder).eval().requires_grad_(False)
    metric = make_perceptual(cfg, '2C')
    count_rates = codec.quantizer.num_rates
    assert count_rates == len(cfg['rate_targets'])
    if decoder.side_branch.quantizer is not None:
        assert decoder.side_branch.quantizer.num_rates == count_rates
    side_quantized = decoder.side_branch.quantizer is not None
    report = dict(
        checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
        stage='2C', step=saved['stage_step'], training_identity=saved['identity'],
        decoder_seed=cfg['decoder_seed'], pid_steps=4, input_size=[512, 512], output_size=[512, 512],
        main_latent_shape=[16, 16, 16], side_latent_shape=[8, 16, 16], count=24,
        input_directory=str(args.input), rate_targets=cfg['rate_targets'], side_quantized=side_quantized,
        side_rate_accounted=side_quantized, total_bpp=None, joint_finetune=decoder.joint_finetune,
        rate_protocol=('Main + side per-channel empirical entropy pooled over all 24 images; excludes entropy coding and headers. No actual bitstream bpp.' if side_quantized else
                       'Main per-channel zero-order empirical entropy pooled over all 24 images; excludes continuous side latent, entropy coding and headers.'),
        metric_protocol='Saved/rounded uint8 RGB, per-image arithmetic mean. PSNR/MSE on [0,1]; SSIM 11x11 Gaussian sigma=1.5; LPIPS VGG v0.1; DISTS [0,1]. Full enhanced Stage 2C path, batch=1, BF16, no_grad.',
        preprocessing='Existing 512x512 Kodak center crops, without resizing; not full 768x512 Kodak evaluation.',
        package_versions={name: importlib.metadata.version(name) for name in ('torch', 'lpips', 'pytorch-msssim', 'DISTS-pytorch')},
        inputs=[], per_image=[], rates=[],
    )
    del saved
    (args.output / 'originals').mkdir(exist_ok=True)
    (args.output / 'components').mkdir(exist_ok=True)
    (args.output / 'comparisons').mkdir(exist_ok=True)
    for rate in range(count_rates):
        (args.output / f'rate{rate}/final').mkdir(parents=True, exist_ok=True)
    entropy = [SymbolEntropy() for _ in range(count_rates)]
    side_entropy = [SymbolEntropy(channels=cfg['side_branch']['channels']) for _ in range(count_rates)]
    torch.cuda.reset_peak_memory_stats()
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            image = image.convert('RGB')
            if image.size != (512, 512):
                raise ValueError(f'Input must already be a 512 crop: {path}')
            array = np.asarray(image).copy()
        original = torch.from_numpy(array).permute(2, 0, 1)[None].cuda()
        x = original.float() / 127.5 - 1
        destination = args.output / 'originals' / path.name
        if not destination.exists():
            try:
                os.link(path, destination)
            except OSError:
                shutil.copy2(path, destination)
        report['inputs'].append(dict(image=path.name, sha256=sha256(path)))
        with torch.autocast('cuda', dtype=torch.bfloat16):encoded=codec.encode(x)
        target_features=decoder.full_vae(x.float()).detach()
        for rate in range(count_rates):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                details = joint(x, rate, return_details=True, encoded=encoded, target_features=target_features)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            entropy[rate].update(details['symbols'])
            per_image_entropy = SymbolEntropy()
            per_image_entropy.update(details['symbols'])
            row = dict(image=path.name, rate=rate, reconstruction_seconds=elapsed,
                       main_estimated_bpp_per_image=per_image_entropy.result()['estimated_bpp'],
                       residual_rms=details['residual'].float().square().mean().sqrt().item())
            if side_quantized:
                side_entropy[rate].update(details['side_symbols'])
                single = SymbolEntropy(channels=cfg['side_branch']['channels'])
                single.update(details['side_symbols'])
                row['side_estimated_bpp_per_image'] = single.result()['estimated_bpp']
                row['total_estimated_bpp_per_image'] = row['main_estimated_bpp_per_image'] + row['side_estimated_bpp_per_image']
            for branch, key in [('base', 'base'), ('conditioned', 'conditioned'), ('final', 'prediction')]:
                pred = details[key]
                if not torch.isfinite(pred).all():
                    raise FloatingPointError(f'{path.name}, rate {rate}, {branch}')
                pixels = rgb8(pred)
                row[branch] = paired(original, pixels, metric)
                if branch == 'final':
                    picture(pixels).save(args.output / f'rate{rate}/final' / path.name)
                elif path.name in EXAMPLES:
                    picture(pixels).save(args.output / 'components' / f'{path.stem}_rate{rate}_{branch}.png')
            if path.name in EXAMPLES:
                picture(rgb8(details['residual'] * 4)).save(args.output / 'components' / f'{path.stem}_rate{rate}_residual_x4.png')
            report['per_image'].append(row)
            del details
        write_json(args.output / 'metrics.partial.json', report)
        print(f'Reconstructed {index + 1}/24: {path.name}', flush=True)
    for rate in range(count_rates):
        rows = [r for r in report['per_image'] if r['rate'] == rate]
        stats = entropy[rate].result()
        result = dict(rate=rate, target_main_bpp=cfg['rate_targets'][rate],
                      main_estimated_bpp=stats.pop('estimated_bpp'), main_entropy=stats,
                      reconstruction_seconds=np.mean([r['reconstruction_seconds'] for r in rows]).item())
        if side_quantized:
            side_stats = side_entropy[rate].result()
            side_bpp = side_stats.pop('estimated_bpp')
            result.update(target_side_bpp=cfg['side_rate_targets'][rate], side_estimated_bpp=side_bpp,
                          side_entropy=side_stats, total_estimated_bpp=result['main_estimated_bpp'] + side_bpp)
        for branch in BRANCHES:
            result[branch] = {key: np.mean([r[branch][key] for r in rows]).item() for key in METRICS}
        result['final_vs_base'] = {key: result['final'][key] - result['base'][key] for key in METRICS}
        report['rates'].append(result)
    report['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
    write_json(args.output / 'metrics.json', report)
    print('Reconstruction and paired metrics complete', flush=True)


def distribution(args):
    import torch_fidelity
    report = json.loads((args.output / 'metrics.json').read_text())
    weights = Path(os.environ.get('TORCH_HOME', ROOT / 'checkpoints/.torch-cache')) / 'hub/checkpoints/weights-inception-2015-12-05-6726825d.pth'
    if not weights.is_file():
        raise FileNotFoundError(f'Existing Inception weights required: {weights}')
    torch.set_num_threads(4)
    for row in report['rates']:
        rate = row['rate']
        if 'distribution' in row:
            continue
        print(f'FID/KID rate {rate}', flush=True)
        with torch.no_grad():
            result = torch_fidelity.calculate_metrics(
                input1=str(args.output / 'originals'), input2=str(args.output / f'rate{rate}/final'),
                cuda=args.distribution_device == 'cuda', batch_size=4, isc=False, fid=True, kid=True,
                kid_subset_size=20, kid_subsets=100, rng_seed=42,
                feature_extractor_weights_path=str(weights), cache=False, verbose=False,
            )
        assert all(math.isfinite(v) for v in result.values())
        row['distribution'] = result
        write_json(args.output / 'metrics.json', report)
    report['distribution_protocol'] = dict(
        library='torch-fidelity', version=importlib.metadata.version('torch-fidelity'),
        device=args.distribution_device,
        feature_extractor='inception-v3-compat', features=2048, scope='whole 512 image; standard internal Inception resize',
        kid_subset_size=20, kid_subsets=100, seed=42, kid_units='raw, no multiplier; std across subsets, not a confidence interval',
        limitation='Only 24 paired images. FID has substantial small-sample bias and KID is noisy; diagnostic results, not a reliable distribution-quality ranking.',
    )
    write_json(args.output / 'metrics.json', report)


def make_report(args):
    report = json.loads((args.output / 'metrics.json').read_text())
    quantized = report.get('side_quantized', False)
    plotted_rate = 'total_estimated_bpp' if quantized else 'main_estimated_bpp'
    rate_label = 'Main + side est.' if quantized else 'Main est.'
    rows = report['per_image']
    lookup = {(r['image'], r['rate']): r for r in rows}
    for filename, items, rate_key in [('per_image.csv', rows, 'main_estimated_bpp_per_image'), ('rates.csv', report['rates'], 'main_estimated_bpp')]:
        flattened = []
        for row in items:
            for branch in BRANCHES:
                rate_values = {key:value for key,value in row.items() if 'estimated_bpp' in key}
                flattened.append(dict(image=row.get('image', 'MEAN24'), rate=row['rate'], branch=branch,
                                      **rate_values, **row[branch]))
        with (args.output / filename).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=flattened[0].keys())
            writer.writeheader(); writer.writerows(flattened)
    final_rows = [dict(rate=row['rate'], **{k:v for k,v in row.items() if 'estimated_bpp' in k},
                       **row['final'], **row.get('distribution', {})) for row in report['rates']]
    with (args.output / 'final_metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in final_rows for key in row)))
        writer.writeheader(); writer.writerows(final_rows)
    page = ['<!doctype html><meta charset="utf-8"><title>Stage 2C Kodak24</title>',
            '<style>body{font:16px sans-serif;margin:24px}.row{display:flex;gap:8px;overflow:auto}figure{margin:0}figcaption{font-size:13px;white-space:pre-line}img{max-width:none}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ccc}</style>',
            f'<h1>Stage 2C step {report["step"]}: Kodak24 / 512 center crops</h1>',
            f'<p>Complete side-conditioned path. {html.escape(report["rate_protocol"])} Metrics use saved uint8 RGB. Original Kodak images are center-cropped, not resized.</p>',
            '<p><a href="metrics.json">Complete metrics / protocol</a> · <a href="final_metrics.csv">Final metrics CSV (all metrics)</a> · <a href="rates.csv">Rate summary CSV</a> · <a href="per_image.csv">Per-image CSV</a> · <a href="curves.png">Metric curves</a></p>',
            f'<table><tr><th>Rate</th><th>{rate_label} bpp</th><th>PSNR</th><th>SSIM</th><th>LPIPS-VGG</th><th>DISTS</th><th>FID (24)</th><th>KID raw (24)</th></tr>']
    for rate in report['rates']:
        m = rate['final']; d = rate.get('distribution', {})
        page.append(f'<tr><td>{rate["rate"]}</td><td>{rate[plotted_rate]:.5f}</td><td>{m["psnr"]:.3f}</td><td>{m["ssim"]:.5f}</td><td>{m["lpips_vgg"]:.5f}</td><td>{m["dists"]:.5f}</td><td>{d.get("frechet_inception_distance", float("nan")):.3f}</td><td>{d.get("kernel_inception_distance_mean", float("nan")):.6f}</td></tr>')
    page.append('</table><p>24-image FID/KID are small-sample diagnostics. KID can be negative.</p>')
    for name in NAMES:
        page.append(f'<h2>{html.escape(name)}</h2><div class="row"><figure><figcaption>Original</figcaption><img src="originals/{name}" width="512" loading="lazy"></figure>')
        for rate in report['rates']:
            r = rate['rate']; m = lookup[name, r]['final']
            title = f'Rate {r} | {rate_label} {rate[plotted_rate]:.5f} bpp\nPSNR {m["psnr"]:.2f} | LPIPS {m["lpips_vgg"]:.4f}'
            page.append(f'<figure><figcaption>{html.escape(title)}</figcaption><a href="rate{r}/final/{name}"><img src="rate{r}/final/{name}" width="512" loading="lazy"></a></figure>')
        page.append('</div>')
        if name in EXAMPLES:
            stem = Path(name).stem
            count_rates = len(report['rates'])
            strip = Image.new('RGB', (512 * (count_rates + 1), 568), 'white'); draw = ImageDraw.Draw(strip)
            sources = [('Original', args.output / 'originals' / name)]
            for rate in report['rates']:
                r = rate['rate']; m = lookup[name, r]['final']
                sources.append((f'R{r} | {rate_label} {rate[plotted_rate]:.5f} bpp\nPSNR {m["psnr"]:.2f} | LPIPS {m["lpips_vgg"]:.4f}', args.output / f'rate{r}/final' / name))
            for j, (label, path) in enumerate(sources):
                with Image.open(path) as image:
                    strip.paste(image, (j * 512, 56))
                draw.multiline_text((j * 512 + 6, 7), label, fill='black')
            strip.save(args.output / 'comparisons' / f'{stem}_rates.png')
            canvas = Image.new('RGB', (512 * 5, 554 * count_rates), 'white'); draw = ImageDraw.Draw(canvas)
            for row, rate in enumerate(report['rates']):
                r = rate['rate']
                base_label = 'Base: current encoder + LoRA' if report.get('joint_finetune') else 'Base: frozen Stage 2B'
                sources = [('Original', args.output / 'originals' / name), (base_label, args.output / 'components' / f'{stem}_rate{r}_base.png'), ('AdaLN conditioned', args.output / 'components' / f'{stem}_rate{r}_conditioned.png'), ('Signed RGB residual x4 (gray=0)', args.output / 'components' / f'{stem}_rate{r}_residual_x4.png'), ('Final', args.output / f'rate{r}/final' / name)]
                for j, (label, path) in enumerate(sources):
                    with Image.open(path) as image:
                        canvas.paste(image, (j * 512, row * 554 + 42))
                    draw.text((j * 512 + 5, row * 554 + 8), f'Rate {r}: {label}', fill='black')
            canvas.save(args.output / 'comparisons' / f'{stem}_components.jpg', quality=92)
            page.append(f'<p><a href="comparisons/{stem}_rates.png">Full-resolution {count_rates}-rate strip</a> · <a href="comparisons/{stem}_components.jpg">Base / AdaLN / residual / final</a></p>')
    (args.output / 'gallery.html').write_text('\n'.join(page))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    bpps = [r[plotted_rate] for r in report['rates']]
    for ax, key, label in zip(axes.flat, ['psnr', 'ssim', 'lpips_vgg', 'dists'], ['PSNR (dB)', 'SSIM', 'LPIPS-VGG', 'DISTS']):
        for branch in BRANCHES:
            branch_bpps = [r['main_estimated_bpp'] for r in report['rates']] if branch=='base' else bpps
            ax.plot(branch_bpps, [r[branch][key] for r in report['rates']], marker='o', label=branch)
        ax.set_xlabel('Estimated bpp (base: main only; enhanced: main + side)' if quantized else 'Main-stream estimated bpp (side rate uncounted)'); ax.set_ylabel(label)
        ax.grid(alpha=.3); ax.legend()
    fig.suptitle(f'Stage 2C step {report["step"]}: Kodak24 / 512 center crops')
    fig.tight_layout(); fig.savefig(args.output / 'curves.png', dpi=150); plt.close(fig)
    print('Report:', args.output / 'gallery.html', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('all', 'reconstruct', 'distribution', 'report'), default='all')
    parser.add_argument('--distribution-device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    for key in ('checkpoint', 'input', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    if not args.output.is_relative_to(ROOT):
        parser.error('Output must stay under pid-codec')
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('TORCH_HOME', str(ROOT / 'checkpoints/.torch-cache'))
    if args.phase in ('all', 'reconstruct'):
        reconstruct(args)
        gc.collect(); torch.cuda.empty_cache()
    if args.phase in ('all', 'distribution'):
        distribution(args)
    make_report(args)


if __name__ == '__main__':
    main()
