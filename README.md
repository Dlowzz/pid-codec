# PiD codec encoder — Stages 1, 2A and 2B

Source-only implementation of a learned compact image encoder conditioned on NVIDIA PiD. This repository contains no weights, image data, dataset manifests, environments, generated results, or code from other compression experiments.

## Architecture and objectives

```
RGB 512×512 → pretrained DC-AE-f32c32-mix-1.0 encoder
            → [B,32,16,16] → trainable 1×1 Conv(32,16)
            → [B,16,16,16] → frozen official PiD 4-step decoder
            → reconstructed RGB 512×512
```

The reference condition uses PiD's bicubic antialiased 512→128 downsampling and its frozen Qwen VAE, including the VAE's native latent normalization. `degrade_sigma=0`.

- Stage A: `MSE(z_codec, z_reference)`. AdamW, base LR `1e-4`, default 30,000 optimizer steps.
- Stage B: `L1(x_hat,x) + 0.5*LPIPS_VGG(x_hat,x) + lambda*MSE(z_codec,z_reference)`. AdamW, base LR `5e-5`, 50,000 steps. Alignment weight decreases linearly from `0.5` to `0.05`.
- Both stages: 500-step warmup, cosine decay, BF16, gradient clipping at 1.0. Image losses operate on `[-1,1]` tensors.
- Only the encoder and projection are trainable. Stage B retains input gradients through all four frozen PiD steps and frozen LPIPS. Noise uses fixed seed 42. PyTorch deterministic algorithms are enabled; exact replay assumes unchanged hardware/software and batch geometry.
- Stage 1 has no quantization; Stage 2A adds fixed multi-rate scalar quantization. Neither stage includes an entropy model, hyperprior, ROI stream, GAN, or EMA. This is not yet a complete bitstream codec.

## Upstream PiD and external prerequisites

Run `bash scripts/setup_pid.sh` to fetch [official NVIDIA PiD](https://github.com/nv-tlabs/PiD) at commit `2c8814c2b91cc41a2be7809962c891e0d0ccff5f` and apply `patches/pid-fixed-step-noise.patch`. The patch only adds optional externally supplied per-step noise to the official sampler; ordinary inference is unchanged. Upstream PiD retains its own license and copyright notices.

Activate your own Python environment before running the commands below. The tested stack used Python 3.12, PyTorch 2.8/CUDA 12.8, Diffusers 0.37.1, Transformers 4.57.3 and LPIPS 0.1.4, plus official PiD dependencies. No environment is shipped or automatically installed. Use the upstream instructions for dependencies and access-controlled models, including Gemma text-encoder access where required.

Required external weights:

- DC-AE: pinned repository, revision and SHA-256 in `configs/dcae_weights.json`. `bash scripts/pid_codec/run.sh download` downloads and verifies it under ignored `checkpoints/pid_codec/dcae/`.
- PiD Qwen four-step student and Qwen 2D VAE: filenames, revision and SHA-256 in `configs/pid_weights.json`. Obtain them from the official `nvidia/PiD` model repository and place the listed `checkpoints/...` files relative to `third_party/PiD/`.
- PiD's text-encoder weights and LPIPS VGG weights are also external dependencies. Authenticate locally as needed; never put access tokens in tracked files.

`configs/pid_inference.json` contains only decoder-loading settings. No inference-result directory is required.

## Data and launch

Supply local text lists containing one image path per line. Relative image paths are resolved relative to each list. An optional existing SQLite index can be provided with `--sqlite-index`; its schema is `images(id, root, path, source)`. No recursive dataset scan runs at training startup.

```bash
bash scripts/pid_codec/run.sh data --image-list /path/to/local-images.txt --image-list /path/to/openimages.txt
bash scripts/pid_codec/run.sh download
bash scripts/pid_codec/run.sh checks
bash scripts/pid_codec/run.sh profile-a
bash scripts/pid_codec/run.sh profile-b
bash scripts/pid_codec/run.sh smoke
bash scripts/pid_codec/run.sh train-a
```

The cached index and train/validation IDs live under ignored `data/pid_codec/`. The lowest 2,000 seeded path hashes form the fixed validation set; remaining images train. Input lists must contain more than 2,000 distinct paths. Exact path deduplication is performed; content deduplication is not asserted. Existing caches are reused; build a separate cache deliberately when changing datasets.

Training resizes the short edge to 512 while preserving aspect ratio, then randomly crops 512×512 and optionally flips horizontally. Validation uses a center crop. Crop/flip randomness follows the committed sample cursor, so worker prefetch does not change resumed samples.

Profiling measures real optimizer updates with gradient accumulation and compares DataLoader workers. It detects GPUs instead of hardcoding batch sizes. The allocator is capped at 94% to leave room for CUDA libraries. Reference results on one A800 80 GB were A: microbatch 25 × accumulation 3; B: 13 × 2, both with four workers. Re-profile your machine; these numbers are not presets.

## Checkpoints, validation and stage transition

```bash
bash scripts/pid_codec/run.sh resume-a
bash scripts/pid_codec/run.sh train-b
bash scripts/pid_codec/run.sh resume-b
```

By default B requires a completed production A checkpoint. To deliberately transition early, use an explicit checkpoint and flag:

```bash
bash scripts/pid_codec/run.sh train-b --allow-early-transition --init-from runs/pid_codec/stage1/stage_A/step_008000.pt
```

Early transition does not relabel A as having completed 30k steps. B creates a fresh optimizer/scheduler and records checkpoint provenance and actual lineage step count. Debug checkpoints are rejected for production initialization.

Every 1,000 steps, save atomic periodic checkpoints and hard-linked `latest.pt`/`best.pt`, with encoder/projection, optimizer, scheduler, stage/step, sample cursor, per-rank RNG, batch geometry and code/data identity. BF16 scaler and EMA are explicitly absent. SIGTERM/SIGINT save after a complete optimizer update. Resume rejects changed training identity or batch geometry. Portability edits in this export change code identity: checkpoints from the original server checkout are not automatically interchangeable with this export's resume guard.

Validation uses 32 fixed images every 1,000 steps, saves four original/reference-PiD/codec-PiD comparisons, and reports float-image PSNR, LPIPS and latent MSE. The full 2,000-image holdout is evaluated at normal stage completion. Best A uses latent MSE; best B uses `L1+0.5*LPIPS`, independent of alignment-weight decay.

## Validation record and limitations

The original training implementation was exercised with official pretrained weights: correct latent shapes, parity with official four-step inference, nonzero reconstruction gradients into the encoder/projection while PiD remains frozen, and bit-exact encoder/optimizer/scheduler comparison for four continuous updates versus two updates plus resume in both stages. Its Stage A was deliberately stopped at 8,430 updates and Stage B initialized from the best latent-MSE checkpoint at 8,000; these run artifacts are not distributed here.

The Stage 1 source export uses portable paths and external environment activation. Later-stage configs preserve the local experiment paths and source-checkpoint hashes; configure these for your installation before profiling. Stage 1 results above describe the original checkout; Stage 2A uses this checkout and stores its own GPU checks under ignored run directories. Run `checks`, GPU profiling and `smoke` on your installation before long training. The released PiD decoder was designed for larger resolutions; a working 512×512 forward path does not guarantee artifact-free reconstruction. Training quality should be assessed using held-out reconstructions, not latent MSE alone.


## Stage 2A: multi-rate quantization-aware finetuning

`configs/pid_codec/stage2a.json` extends the same loader, differentiable PiD sampler,
optimizer loop, profiling, atomic checkpoints and resume guards. Set
`initial_checkpoint` to a production Stage B best checkpoint. This starts a new
optimizer/scheduler; it does not resume Stage B's optimizer. The source stage,
actual step and original global step are recorded. Read-only model/data paths
may be symlinks; run outputs must remain inside this checkout.

After the projection, each image randomly chooses one of six fixed `[16]`
per-channel steps. The FP32 quantizer uses `y=z/delta`,
`q=y+(round(y)-y).detach()`, `z_hat=q*delta`. PiD receives `z_hat`.
Anchors are immutable buffers saved with the encoder; they cannot collapse to
zero. PiD, Qwen reference VAE and LPIPS remain frozen, with reconstruction
backpropagation through PiD and the STE to the encoder/projection.

Calibration measures channel standard deviation on 512 seeded training samples,
then scales it independently for each rate using 128 validation images (indices
32–159, separate from the 32 images used for best-checkpoint selection). Targets
are 0.02/0.03/0.04/0.06/0.08/0.10 estimated bpp. We pool integer-symbol counts
per channel, compute zero-order entropy, and divide estimated bits by the number
of original 512×512 pixels. This excludes table/header/coder overhead and is not
a real bitstream rate. Fixed deltas do not guarantee constant entropy as the
encoder distribution changes; validation always reports newly measured bpp.

The loss is `L1 + 0.5*LPIPS + L_frequency + lambda_align*MSE(z_hat,z_reference)`.
Alignment decreases linearly 0.4→0.15 over 30,000 optimizer updates. Frequency
loss follows [DeCo's official implementation](https://github.com/Zehong-Ma/DeCo/blob/0792af05c9d8dce6c61e5636d136488f264065c7/src/diffusion/flow_matching/training_repa_DeCo.py):
RGB→YCbCr, orthonormal nonoverlapping 8×8 block DCT, JPEG tables at quality 85,
`inv_gamma`, gamma 1, per-channel mean-normalized weights. Unlike DeCo's velocity
objective, ours compares reconstructed and original RGB images on `[-1,1]`.
Frequency computation is FP32 even in BF16 training. Raw and weighted losses,
rate fractions, gradient norm, LR, memory and throughput are logged separately.

AdamW uses base LR 1e-4, 500-step warmup, cosine decay to 10% of base LR,
weight decay 0.01 and gradient clipping 1.0. Profiling performs real accumulated
optimizer updates, starting its search near Stage B's last feasible batch;
it still tests the upper bound and selects the measured maximum. Accumulation
is `ceil(64/(microbatch*GPU_count))`.

Activate the existing PiD environment before these commands:

```bash
bash scripts/pid_codec/run.sh calibrate-2a
# Supply a local copy of the pinned official DeCo file for numerical parity:
bash scripts/pid_codec/run.sh checks-2a basic --official /path/to/training_repa_DeCo.py
bash scripts/pid_codec/run.sh checks-2a gradient
bash scripts/pid_codec/run.sh profile-2a
bash scripts/pid_codec/run.sh smoke-2a
bash scripts/pid_codec/run.sh train-2a
# After interruption:
bash scripts/pid_codec/run.sh resume-2a
```

Do not rerun calibration after training starts: anchors and their provenance
are part of training identity. The real gradient check deliberately excludes
alignment so it cannot mask a broken PiD gradient path. It also measures
frequency versus reconstruction gradient norms at the lowest/highest rates.
The smoke test compares four uninterrupted updates against two plus exact
resume, including encoder, fixed anchors, AdamW and scheduler state.

Every 1,000 steps, evaluate all six rates on the same 32 fixed validation images.
Best selection averages `L1+0.5*LPIPS` equally across rates. Each rate reports
estimated bpp, entropy, PSNR, LPIPS, L1, frequency loss and post-quantization
alignment MSE. Four image grids show original/reference plus all six rates;
individual 512×512 reconstructions are also saved. Keep the last three visual
sets and periodic checkpoints. Full 2,000-image all-rate evaluation runs at
normal stage completion. Runtime artifacts live under
`runs/pid_codec/stage2A/`; none are included in Git.

### Wider-rate continuation

The wider six-level policy uses estimated bpp targets
`0.006 / 0.010 / 0.020 / 0.040 / 0.070 / 0.100` in
`configs/pid_codec/stage2a_wide.json`. Anchors can now be recalibrated against an
actual Stage 2A encoder, retaining the original Stage B bootstrap provenance:

```bash
bash scripts/pid_codec/run.sh calibrate-2a \
  --config configs/pid_codec/stage2a_wide.json \
  --checkpoint /path/to/current-stage2a.pt
```

Use a fresh anchor output path: calibration refuses to overwrite existing
anchors. Changing the config alone does not change the quantizer buffers inside
an existing checkpoint. The server continuation explicitly migrates those
buffers and the code/config identity, verifies preserved encoder/AdamW/scheduler
state, and records provenance under `runs/pid_codec/stage2A/wide_rates/`.
The new run uses `stage_2A_wide`, keeping old rate-policy metrics separate.
Best selection restarts for the new set of rates. Steps, data cursor, RNG and
learning-rate progress continue. Fixed anchors cannot collapse, but actual
entropy can drift with the evolving encoder and image content; all-rate
validation reports measured estimates rather than only target labels.

## Stage 2B: joint PiD LoRA and encoder finetuning

Stage 2B initializes the encoder, 32→16 projection, and fixed six-rate scalar
quantizer from the Stage 2A wide-rate best checkpoint (step 3000). It loads the
original official Qwen PiD 4-step checkpoint **before** injecting PEFT adapters,
using the official image-stream targets (`qkv_x`, `proj_x`, `mlp_x.w1/w2/w3`,
`fc1`, `fc2`), rank=alpha=32, dropout=0. The original PiD parameters, Qwen
reference VAE, text encoder, LPIPS, and DISTS stay frozen. Encoder/projection
parameters and LoRA matrices are FP32 with BF16 autocast. The joint module is
used for DDP so adapters receive synchronized gradients too.

The network retains its pretrained velocity parameterization. The official
sampler converts it to x0 with `x0 = x_t - t * v` at each of the four steps.
Loss supervises the **final clamped reconstruction**, with gradients through
all four steps and hard-round STE:

```
MSE(x_hat, x) + LPIPS_VGG(x_hat, x) + JPEG_DCT(x_hat, x)
+ 0.2 * DISTS(x_hat, x) + lambda_align * MSE(z_hat, z_reference)
```

MSE, LPIPS and JPEG use `[-1,1]`; the DISTS wrapper converts to `[0,1]` and
explicitly enables input gradients (the standalone upstream implementation
otherwise defaults to `no_grad`). All losses use FP32 mean reductions. JPEG
keeps DeCo quality=85, inv_gamma, gamma=1. Alignment starts at approximately
0.375, inherited from Stage 2A step 3000, and decreases linearly to 0.15.
PiD LoRA LR is 1e-4; encoder/projection LR is 5e-5. Defaults are AdamW,
weight_decay=0.01, 500-step warmup, cosine decay to 10%, clip=1, 30k steps,
no EMA. There is no GAN, DMD, REPA, entropy model, or trainable quantizer.

The configured targets remain 0.006/0.01/0.02/0.04/0.07/0.1 bpp, but fixed
anchors do not enforce exact rates. Validation reports pooled per-channel
integer-symbol empirical entropy, **not actual bitstream bpp**. The source
checkpoint measured roughly 0.0090/0.0139/0.0228/0.0432/0.0732/0.1031 bpp.

Run from this repository after activating the existing PiD environment:

```bash
bash scripts/pid_codec/run.sh checks-2b
bash scripts/pid_codec/run.sh profile-2b
bash scripts/pid_codec/run.sh smoke-2b
bash scripts/pid_codec/run.sh train-2b
# Subsequent restart:
bash scripts/pid_codec/run.sh resume-2b
```

Additional dependencies are pinned in `configs/requirements-stage2b.txt`.
The existing DISTS/VGG assets are reused; no broad PyIQA dependency upgrade is
required. The 2B config pins its source checkpoint and learned DISTS weights.
Profiling measures the complete joint optimizer update, including DISTS,
then selects a stable microbatch and accumulation for effective batch ≈64.

Outputs go to `runs/pid_codec/stage2B/stage_2B` on the data volume. Initial
validation caches immutable Stage 2A source reconstructions. Later grids
show original RGB, the original PiD reference baseline (LoRA disabled), all
six current rates, and the six source reconstructions. Every 1000 steps,
32 fixed validation images produce PSNR, LPIPS, DISTS, MSE, L1, JPEG loss,
latent MSE and empirical bpp. Full validation runs at completion. Best is
selected by mean-over-rates `MSE + LPIPS + JPEG + 0.2*DISTS`, excluding the
changing alignment term.

Checkpoint files contain encoder/projection/quantizer state, LoRA matrices
and target metadata, both optimizer groups, scheduler, sample cursor, RNG,
source/base-weight identity, stage and step. Frozen PiD weights are referenced
rather than duplicated. Checkpoints are written every 1000 steps, at the end,
and on a handled interruption, not every step. Latest/best are hard links;
the last three periodic checkpoints are retained. Debug checks use separate
outputs and never initialize the formal run from diagnostic updates.

Before the first formal Stage 2B launch, `python -m scripts.pid_codec.overfit`
checks four fixed **training** images at the highest rate with the production
loss, anchors, decoder noise and peak learning rates. The diagnostic uses
microbatch 4 and 20 warmup steps, and stops after at least 100 steps once mean
PSNR improves ≥3 dB, LPIPS decreases ≥20%, reconstruction objective decreases
≥30%, and every image improves ≥0.5 dB. The initial budget is 500 steps;
`--resume --max-steps 1500` extends the diagnostic from its saved optimizer
state and cosine horizon, with identical images, loss, peak LRs and pass
criteria. Its weights are
never used to initialize production. Results and before/after grids are in
`runs/pid_codec/stage2B/overfit_4`.

After launch, `python -m scripts.pid_codec.monitor --run
runs/pid_codec/stage2B/stage_2B --seconds 1200` checks actual wall-clock
training progress, finite losses/gradients, loss-weight accounting, optimizer
LR groups, data cursor continuity, throughput, GPU memory and process health.
The default tmux session name is `pid-codec-stage2b`.


### Training curves

After activating the PiD environment, run
`python -m scripts.pid_codec.plot_metrics` to refresh CPU-only plots from the
existing Stage 2B logs. This produces raw/smoothed losses, gradient norms before
clipping, clipping frequency, weighted loss shares, LR, memory, throughput,
and all-rate validation PSNR/LPIPS/DISTS/bpp curves under
`runs/pid_codec/stage2B/curves/`. Validation curves show only measured checkpoints;
loss shares are not gradient shares. The plots and source logs remain local.
