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
  train-2a|resume-2a)
    GPUs="$(python -c 'import torch; print(torch.cuda.device_count())')"
    extra=()
    if [[ "$action" == resume-* ]]; then extra=(--resume auto); fi
    python -m torch.distributed.run --standalone --nproc_per_node="$GPUs" --module scripts.pid_codec.train \
      --stage 2A --config configs/pid_codec/stage2a.json \
      --profile runs/pid_codec/stage2A/profiles/2A/selected.json "${extra[@]}" "$@"
    ;;
  checks-2b) python -u -m scripts.pid_codec.checks_stage2b "$@" ;;
  profile-2b) python -u -m scripts.pid_codec.profile --stage 2B --config configs/pid_codec/stage2b.json "$@" ;;
  smoke-2b) python -u -m scripts.pid_codec.smoke_stage2 --stage 2B "$@" ;;
  train-2b|resume-2b)
    GPUs="$(python -c 'import torch; print(torch.cuda.device_count())')"
    extra=()
    if [[ "$action" == resume-* ]]; then extra=(--resume auto); fi
    python -m torch.distributed.run --standalone --nproc_per_node="$GPUs" --module scripts.pid_codec.train \
      --stage 2B --config configs/pid_codec/stage2b.json \
      --profile runs/pid_codec/stage2B/profiles/2B/selected.json "${extra[@]}" "$@"
    ;;
  profile-a) python -u -m scripts.pid_codec.profile --stage A "$@" ;;
  profile-b) python -u -m scripts.pid_codec.profile --stage B "$@" ;;
  train-a|resume-a|train-b|resume-b)
    GPUs="$(python -c 'import torch; print(torch.cuda.device_count())')"
    stage=A; profile=runs/pid_codec/stage1/profiles/A/selected.json
    if [[ "$action" == *-b ]]; then stage=B; profile=runs/pid_codec/stage1/profiles/B/selected.json; fi
    extra=()
    if [[ "$action" == resume-* ]]; then extra=(--resume auto); fi
    if [[ "$action" == train-b ]]; then extra=(--init-from runs/pid_codec/stage1/stage_A/latest.pt); fi
    python -m torch.distributed.run --standalone --nproc_per_node="$GPUs" --module scripts.pid_codec.train \
      --stage "$stage" --profile "$profile" "${extra[@]}" "$@"
    ;;
  *) echo 'Usage: run.sh {data|download|checks|smoke|profile-a|profile-b|train-a|resume-a|train-b|resume-b|calibrate-2a|checks-2a|profile-2a|smoke-2a|train-2a|resume-2a|checks-2b|profile-2b|smoke-2b|train-2b|resume-2b} [arguments]' ;;
esac
