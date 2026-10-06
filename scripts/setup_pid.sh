#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REV=2c8814c2b91cc41a2be7809962c891e0d0ccff5f
DEST="$ROOT/third_party/PiD"
if [[ ! -d "$DEST/.git" ]]; then
  git clone https://github.com/nv-tlabs/PiD.git "$DEST"
  git -C "$DEST" checkout --detach "$REV"
fi
[[ "$(git -C "$DEST" rev-parse HEAD)" == "$REV" ]] || { echo 'PiD revision mismatch; inspect your checkout.' >&2; exit 1; }
PATCH="$ROOT/patches/pid-fixed-step-noise.patch"
if git -C "$DEST" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo 'PiD patch already applied.'
else
  git -C "$DEST" apply --check "$PATCH"
  git -C "$DEST" apply "$PATCH"
fi
