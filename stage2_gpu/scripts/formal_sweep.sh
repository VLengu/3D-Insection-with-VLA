#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_root}"
python -m stage2_gpu.runtime train --stage formal --config-dir stage2_gpu/configs --seeds 42 43 44
