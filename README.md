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

## Stage 2C: query-based side latent for pixel reconstruction

`configs/pid_codec/stage2c.json` starts a separate 5,000-update run from the
local Stage 2B best checkpoint. Encoder, projection, main quantizer, original
PiD and the existing PiD LoRA are frozen; the LoRA remains enabled. Only the
new query extractor, spatial AdaLN adapter and RGB residual decoder train.
The source checkpoint is pinned under the new run directory and verified by
SHA256. Previous stage artifacts are not changed.

For each image/rate, the main encoder runs once. Its quantized latent produces
`x_base` with the existing four-step PiD under `no_grad`. The same frozen Qwen
VAE encodes **full 512x512** `x` and `x_base` into normalized `[B,16,64,64]`
features, both detached. This path does not use the downsampling reference API.
A Linear projects concatenated `Fx, Fbase, Fx-Fbase` (48 channels) to 128.
Two cross-attention + FFN layers use 256 learned spatial queries and fixed 2D
sin/cos positions. Each query attends to the corresponding 4x4 VAE feature
window, preserving local correspondence. A Linear produces `[B,8,16,16]`.
The side latent is continuous in this experiment, without quantization/coding.

A learned 2x upsampling CNN maps the side latent to PiD's 32x32 patch grid.
Independent zero-initialized Linear heads add six spatial shift/scale/gate
increments to the **image** AdaLN output of the latter half of `patch_blocks`.
Text/time modulation is retained. The adapter is active in **all four**
sampling steps. Runtime hooks live in `scripts/pid_codec/side.py`; no PiD
network forward is copied or edited. The side scope is entered inside each
checkpointed network function, so backward recomputation sees the same side
condition. Existing LoRA and activation checkpointing remain compatible.
The pinned upstream source files are included in training identity checks.

A second upsampling CNN maps the same side latent through five 2x stages to a
signed RGB residual, with a zero-initialized output head. The final output is
`clamp(x_cond + residual, -1, 1)`. Velocity prediction is unchanged. Both zero
heads initially reproduce the frozen base in inference; upstream query gradients become
nonzero after the heads receive their first update.

The final-image objective is:

```
1.25*MSE + 0.80*LPIPS_VGG + 1.0*JPEG_frequency + 0.15*DISTS
+ 0.10*MSE(QwenVAE_full(x_hat), stopgrad(QwenVAE_full(x)))
```

Image MSE/JPEG/LPIPS use `[-1,1]`; DISTS uses `[0,1]`. Full VAE consistency
calls the official underlying encoder and normalization without its `no_grad`
wrapper, with VAE activation checkpointing. The original compact-latent
alignment is logged with weight **zero**. Raw/weighted losses, side/residual
RMS, and query/AdaLN/RGB gradient norms before clipping are recorded.

All new optimizer groups use AdamW LR 1e-4, weight decay .01, BF16, 500-step
warmup, cosine decay to 10%, clip=1. Profiling reruns the complete two-PiD
forward/loss/backward update, selecting microbatch and accumulation for an
effective batch near 64. The same immutable data split and stateless loader
resume are used; the new stage begins at the source checkpoint's data cursor.

```bash
bash scripts/pid_codec/run.sh checks-2c
bash scripts/pid_codec/run.sh profile-2c
bash scripts/pid_codec/run.sh smoke-2c
bash scripts/pid_codec/run.sh train-2c
# After a handled interruption:
bash scripts/pid_codec/run.sh resume-2c
```

The checks verify full-resolution VAE normalization/input gradients, frozen
parameters, both side gradient paths, and four-step adapter coverage. They
compare checkpointed and non-checkpointed forwards in the same grad mode.
BF16 grad-enabled and `no_grad` kernels can differ slightly; this separate
comparison is recorded in `checks/parity.json`. The resume smoke test requires
bit-exact weights, optimizer/RNG state, validation metrics and all 12 final
PNGs (two images across six rates) against uninterrupted training.

Checkpoints under `runs/pid_codec/stage2C/stage_2C` include the new side modules,
frozen codec/LoRA state, optimizer, scheduler, RNG, data cursor and source
identity. Latest/best/periodic policies reuse the existing loop. Every 1,000
updates all six main rates are validated with the **complete enhanced path**.
Each rate reports base/AdaLN/final PSNR, LPIPS, DISTS, MSE, JPEG and VAE MSE.
Grids show original, base, AdaLN reconstruction, signed RGB residual (4x gain,
zero shown as gray), and final output. Best selection uses the mean final
five-term objective across rates. `main_estimated_bpp` includes only main
integer symbols: it excludes the continuous side latent and must never be
reported as total codec bitrate. This stage is not a complete bitstream codec.

### Stage 2C joint finetuning

`configs/pid_codec/stage2c_joint.json` continues from the saved Stage 2C
side-only checkpoint in a separate `runs/pid_codec/stage2C_joint` directory.
All encoder, projection, existing PiD LoRA and side weights are loaded exactly.
The optimizer and 500-step warmup/cosine schedule start fresh for 5,000 updates;
the data cursor and global optimizer-step provenance continue from the source.
Source SHA256, the joint-mode flag and each group's LR are checked on resume.

Trainable groups are the full DC-AE encoder/projection (LR `5e-5`), the existing
PiD LoRA (LR `1e-4`), and Query/AdaLN/RGB side modules (each LR `8e-5`). Original
PiD weights, Qwen VAE, perceptual networks and quantization anchors stay frozen.
The encoder/STE forward now retains gradients into the second four-step PiD
pass. The first base reconstruction and the full-resolution VAE features used
by the query remain stop-gradient. Losses, fixed decoder noise, six-rate
validation, checkpoint frequency and target effective batch of 64 are reused.
The displayed base reconstruction now uses the current encoder and LoRA.

For this joint run, `side_branch.adaln_scope=all` extends modulation to all
14 patch blocks at every one of the four sampling steps. Source heads 0–6
are mapped to blocks 7–13; added heads for blocks 0–6 start at zero. Query,
upsampler, RGB decoder and trained late heads are preserved exactly. The
initialization check compares reconstructions against the source's last-half
adapter with identical quantization and requires exact agreement. Coverage of
network depth is independent of sampling time. The original side-only config
defaults to `last_half` and keeps its architecture.

The shared `[B,8,16,16]` side latent now uses the same per-channel hard-round
STE quantizer as the main latent, with six fixed `[6,8]` step anchors. Main
and side use the same sampled rate ID, and **both** AdaLN and RGB consume the
quantized side latent. Calibration collects current query outputs conditioned
on each main rate from 64 training and 64 separate validation images. Training
channel standard deviations set relative channel steps; validation entropy
calibrates one scale per rate. Side targets are `.003/.005/.01/.02/.035/.05`
bpp, matching main target entropy per symbol with half as many side symbols.
Existing main anchors are retained, so these are additional rate costs.
Steps stay fixed without a rate loss; actual estimated rates can drift during
training. Validation and Kodak reports separate main, side and summed empirical
entropy. The sum is an estimate, not an encoded bitstream size.

```bash
bash scripts/pid_codec/run.sh calibrate-side  # Once, before the joint run
bash scripts/pid_codec/run.sh checks-2c-joint
bash scripts/pid_codec/run.sh profile-2c-joint
bash scripts/pid_codec/run.sh smoke-2c-joint
bash scripts/pid_codec/run.sh train-2c-joint
# After a handled interruption:
bash scripts/pid_codec/run.sh resume-2c-joint
```

The same checks additionally require nonzero encoder and LoRA gradients and
actual updates in every optimizer group and gradients in every AdaLN head.
Main bpp remains an empirical estimate
reported separately from quantized-side and total estimated bpp. Checks include
side hard-round values, STE gradients, fixed steps, and exact resume of side
quantizer buffers and rate metrics.


### GAN fitting run (uniform OpenImages + benchmarks)

`configs/pid_codec/stage2c_gan_fit.json` continues the quantized joint Stage 2C
model from its saved step 277 in a separate `runs/pid_codec/stage2D_gan_fit`
directory. The existing LoRA, DC-AE, spatial Query/AdaLN and RGB residual
branches remain jointly trainable. PiD base weights, Qwen VAE and perceptual
networks remain frozen.

The generator objective is `1.25*MSE + 0.80*LPIPS + 0.15*DISTS +
0.10*VAE_latent_MSE + 0.01*GAN`. JPEG frequency computation is disabled.
`GAN = -mean(D(x_hat))`; D uses the standard 70x70 PatchGAN and hinge loss.
D has its own AdamW optimizer (lr 5e-5, betas 0.5/0.9). Generator learning
rates remain LoRA 1e-4, side modules 8e-5 and DC-AE/projection 5e-5.
The source comparisons and distinctions between GAN objectives are recorded
in `configs/pid_codec/gan_references.json`. No additional pretrained backbone,
entropy model or diffusion distillation objective is introduced.

The local `data/pid_codec_openimages_fit_uniform` manifest contains 706,405
OpenImages training images plus Kodak24, Tecnick100 (classic 1200x1200 RGB)
and CLIC2020 Professional Validation41. Every image occurs exactly once in
the training index: ordinary shuffled sampling, no benchmark oversampling.
OpenImages retains its original 1,593 held-out validation images. All LSDIR
and MLIC rows are excluded. `prepare_fit_data.py` constructs this index once
from the existing SQLite manifest, without rescanning OpenImages.

The `fit` split is a diagnostic view of benchmark images that are deliberately
included in training. Its scores are **not independent benchmark results**.
Periodic validation keeps OpenImages and fitting reports separate and saves
all six rates. Checkpoint selection uses the OpenImages reconstruction
objective, excluding the non-stationary GAN score. Both main and side bpp
remain empirical entropy estimates, not actual entropy-coded bitstreams.

```bash
source /root/data/Compression/envs/pid/activate.sh
python -m scripts.pid_codec.checks_gan
bash scripts/pid_codec/run.sh profile-gan
bash scripts/pid_codec/run.sh smoke-gan
bash scripts/pid_codec/run.sh train-gan
# After an interruption:
bash scripts/pid_codec/run.sh resume-gan
```

The same training loop saves discriminator weights, optimizer and scheduler
along with the codec, per-rank RNG and committed sample cursor. The smoke test
compares continuous training with interrupted/resumed training, including D
state and reconstructed images. A fresh run refuses to overwrite an existing
run directory. Complete checkpoints can be restored without their original
bootstrap checkpoint; inference checks architecture and base weights instead
of requiring an unchanged optimizer-training identity. Validation caches the
rate-independent encoder/target-VAE features and reuses perceptual models.
A progress file distinguishes validation progress from stalled training.

### Five shared main/side quantization levels

`calibrate_joint.py` recalibrates the **sum** of main and side empirical entropy
from a complete, gracefully stopped joint checkpoint. Targets default to
`0.006, 0.012, 0.025, 0.05, 0.10` bpp. With `--main-fraction 0.6666666666666666`,
the 16 main and 8 side channels receive a 2:1 target bit split. Both streams use
the same rate ID but separate multipliers:
`delta_stream[r,c] = multiplier_stream[r] * channel_std[c]`. Per-channel scales
are fixed from calibration. Omitting `--main-fraction` instead uses a single
shared multiplier and lets the measured bit split emerge from the distributions.
Concatenating the two integer tensors produces `[B,24,16,16]` symbols suitable
for a later joint entropy model. No hyperprior is added here.

The side latent depends on the quantized main reconstruction, so calibration
recomputes that reconstruction and the query features for the selected main
steps (every candidate in shared-multiplier mode). It uses a separate validation subset, then runs the existing full
reconstruction validation on the original fixed images. Rate targets are
calibration points, not hard constraints: fixed steps without a rate loss can
still drift as the encoder/query distributions change. Estimates exclude
entropy coder overhead and any future hyperlatent stream.

```bash
python -m scripts.pid_codec.calibrate_joint \
  --config configs/pid_codec/stage2c_gan_fit.json \
  --checkpoint runs/pid_codec/stage2D_gan_fit/stage_2C/latest.pt \
  --output-dir runs/pid_codec/stage2D_gan_joint5_balanced \
  --output-config configs/pid_codec/stage2c_gan_joint5_balanced.json \
  --main-fraction 0.6666666666666666
python -m scripts.pid_codec.checks_gan --config configs/pid_codec/stage2c_gan_joint5_balanced.json
python -m scripts.pid_codec.profile --stage 2C --config configs/pid_codec/stage2c_gan_joint5_balanced.json
python -m scripts.pid_codec.train --stage 2C \
  --config configs/pid_codec/stage2c_gan_joint5_balanced.json \
  --profile runs/pid_codec/stage2D_gan_joint5_balanced/profiles/2C/selected.json --resume auto
```

The transition changes only the two quantizer buffers. All trainable weights,
generator/discriminator optimizers and schedulers, RNG, step counters and the
data cursor are verified unchanged after serialization. The old run is kept;
the new run starts with an evaluated `latest`/`best` checkpoint and must use
`--resume auto`. The smaller `initial_requantized.pt` is a weights-only bootstrap
for profiling, not an optimizer resume checkpoint. Training still samples one
uniform random rate **per image**, using that same rate for both streams;
validation evaluates every configured rate for every image.
