#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
venv_path="${STAGE2_GPU_VENV:-${project_root}/.venv_gpu}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "install_linux.sh must run on Linux" >&2
  exit 2
fi

"${python_bin}" -m venv "${venv_path}"
"${venv_path}/bin/python" -m pip install --upgrade pip
"${venv_path}/bin/python" -m pip install -r "${project_root}/stage2_gpu/requirements-linux.lock.txt"
"${venv_path}/bin/python" -m stage2_gpu.preflight --mode gpu
