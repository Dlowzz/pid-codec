#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/third_party/PiD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_CACHE="$ROOT/checkpoints/hf-cache"
export TOKENIZERS_PARALLELISM=false
mkdir -p logs
export TORCH_HOME="$ROOT/checkpoints/.torch-cache"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
action="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
launch_training() {
  local stage="$1" config="$2" profile="$3"
  shift 3
  local gpu_count
  gpu_count="$(python -c 'import torch; print(torch.cuda.device_count())')"
  local extra=()
  if [[ "$action" == resume-* ]]; then extra=(--resume auto); fi
  python -m torch.distributed.run --standalone --nproc_per_node="$gpu_count" --module scripts.pid_codec.train \
    --stage "$stage" --config "$config" --profile "$profile" "${extra[@]}" "$@"
}
case "$action" in
  data) python -m scripts.pid_codec.prepare data "$@" ;;
  download) HF_HUB_DISABLE_XET=1 python -m scripts.pid_codec.prepare download ;;
  checks) python -m scripts.pid_codec.checks basic
          python -m scripts.pid_codec.checks encoder-gradient ;;
  smoke) python -u -m scripts.pid_codec.smoke ;;
  calibrate-2a) python -u -m scripts.pid_codec.calibrate "$@" ;;
  checks-2a) python -u -m scripts.pid_codec.checks_stage2 "$@" ;;
  profile-2a) python -u -m scripts.pid_codec.profile --stage 2A --config configs/pid_codec/stage2a.json "$@" ;;
  smoke-2a) python -u -m scripts.pid_codec.smoke_stage2 "$@" ;;
  train-2a|resume-2a) launch_training 2A configs/pid_codec/stage2a.json runs/pid_codec/stage2A/profiles/2A/selected.json "$@" ;;
  checks-2b) python -u -m scripts.pid_codec.checks_stage2b "$@" ;;
  profile-2b) python -u -m scripts.pid_codec.profile --stage 2B --config configs/pid_codec/stage2b.json "$@" ;;
  smoke-2b) python -u -m scripts.pid_codec.smoke_stage2 --stage 2B "$@" ;;
  train-2b|resume-2b) launch_training 2B configs/pid_codec/stage2b.json runs/pid_codec/stage2B/profiles/2B/selected.json "$@" ;;
  checks-2c) python -u -m scripts.pid_codec.checks_stage2c "$@" ;;
  profile-2c) python -u -m scripts.pid_codec.profile --stage 2C --config configs/pid_codec/stage2c.json "$@" ;;
  smoke-2c) python -u -m scripts.pid_codec.smoke_stage2 --stage 2C "$@" ;;
  checks-2c-joint) python -u -m scripts.pid_codec.checks_stage2c --config configs/pid_codec/stage2c_joint.json "$@" ;;
  calibrate-side) python -u -m scripts.pid_codec.calibrate_side "$@" ;;
  profile-2c-joint) python -u -m scripts.pid_codec.profile --stage 2C --config configs/pid_codec/stage2c_joint.json "$@" ;;
  smoke-2c-joint) python -u -m scripts.pid_codec.smoke_stage2 --stage 2C --config configs/pid_codec/stage2c_joint.json "$@" ;;
  train-2c-joint|resume-2c-joint) launch_training 2C configs/pid_codec/stage2c_joint.json runs/pid_codec/stage2C_joint/profiles/2C/selected.json "$@" ;;
  train-2c|resume-2c) launch_training 2C configs/pid_codec/stage2c.json runs/pid_codec/stage2C/profiles/2C/selected.json "$@" ;;
  profile-a) python -u -m scripts.pid_codec.profile --stage A "$@" ;;
  profile-b) python -u -m scripts.pid_codec.profile --stage B "$@" ;;
  train-a|resume-a|train-b|resume-b)
    stage=A
    if [[ "$action" == *-b ]]; then stage=B; fi
    extra=()
    if [[ "$action" == train-b ]]; then extra=(--init-from runs/pid_codec/stage1/stage_A/latest.pt); fi
    launch_training "$stage" configs/pid_codec/stage1.json "runs/pid_codec/stage1/profiles/$stage/selected.json" "${extra[@]}" "$@"
    ;;
  profile-gan) python -u -m scripts.pid_codec.profile --stage 2C --config configs/pid_codec/stage2c_gan_fit.json "$@" ;;
  smoke-gan) python -u -m scripts.pid_codec.smoke_stage2 --stage 2C --config configs/pid_codec/stage2c_gan_fit.json "$@" ;;
  train-gan|resume-gan) launch_training 2C configs/pid_codec/stage2c_gan_fit.json runs/pid_codec/stage2D_gan_fit/profiles/2C/selected.json "$@" ;;
  *) echo 'Usage: run.sh {data|download|checks|smoke|profile-a|profile-b|train-a|resume-a|train-b|resume-b|calibrate-2a|checks-2a|profile-2a|smoke-2a|train-2a|resume-2a|checks-2b|profile-2b|smoke-2b|train-2b|resume-2b|checks-2c|profile-2c|smoke-2c|train-2c|resume-2c|calibrate-side|checks-2c-joint|profile-2c-joint|smoke-2c-joint|train-2c-joint|resume-2c-joint|profile-gan|smoke-gan|train-gan|resume-gan} [arguments]' ;;
esac
