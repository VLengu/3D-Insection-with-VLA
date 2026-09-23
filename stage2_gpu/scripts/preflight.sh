#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_root}"
python -m stage2_gpu.preflight --mode gpu --output stage2_gpu/runtime/preflight.json
