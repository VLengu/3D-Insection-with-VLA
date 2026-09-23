#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_root}"
: "${LOCKED_EVAL_UNLOCK:?LOCKED_EVAL_UNLOCK must point to the signed audit record}"
python -m stage2_gpu.runtime evaluate --config-dir stage2_gpu/configs --unlock "${LOCKED_EVAL_UNLOCK}"
