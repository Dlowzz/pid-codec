# PiD codec encoder — Stage 1

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
- There is no quantization, entropy model, hyperprior, ROI stream, GAN, or EMA. This is not yet a complete bitstream codec.

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

This source export removes server-specific paths and environment activation. Its portability checks do not constitute a new full GPU training run. Run `checks`, GPU profiling and `smoke` on your installation before long training. The released PiD decoder was designed for larger resolutions; a working 512×512 forward path does not guarantee artifact-free reconstruction. Training quality should be assessed using held-out reconstructions, not latent MSE alone.
