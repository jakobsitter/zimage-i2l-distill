#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

exec python -m zimage_distill.teacher_pairs \
  --repo-root "$repo_root" \
  --output-dir "$repo_root/data/teacher_pairs" \
  "$@"
