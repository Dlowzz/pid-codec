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
  *) echo 'Usage: run.sh {data|download|checks|smoke|profile-a|profile-b|train-a|resume-a|train-b|resume-b} [arguments]' ;;
esac
