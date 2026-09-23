"""Gate M5 static checks on Mac and runtime checks on the Linux/GPU host."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
from pathlib import Path
import subprocess
import sys

from stage2_gpu import PACKAGE_VERSION
from stage2_gpu.contracts import assert_four_way_equivalence, load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "stage2_gpu" / "configs"


def static_checks() -> dict[str, object]:
    paths = sorted(CONFIG_DIR.glob("*.json"))
    configs = [load_config(path) for path in paths]
    assert_four_way_equivalence(configs)
    scripts = sorted((ROOT / "stage2_gpu" / "scripts").glob("*.sh"))
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)
    required = [
        ROOT / "stage2_gpu" / "Dockerfile",
        ROOT / "stage2_gpu" / "requirements-linux.lock.txt",
        ROOT / "stage2_gpu" / "manifests" / "upstream.json",
        ROOT / "stage2_gpu" / "HANDOFF_CHECKLIST_ZH.md",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing handoff files: {missing}")
    return {
        "package_version": PACKAGE_VERSION,
        "configs": [config.condition for config in configs],
        "config_sha256": {config.condition: config.canonical_sha256() for config in configs},
        "shell_scripts": [script.name for script in scripts],
        "status": "pass",
    }


def gpu_checks() -> dict[str, object]:
    if platform.system() != "Linux":
        raise RuntimeError("GPU runtime preflight must run on Linux")
    torch = importlib.import_module("torch")
    lerobot = importlib.import_module("lerobot")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return {
        "status": "pass",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "lerobot": getattr(lerobot, "__version__", "unknown"),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("static", "gpu"), default="static")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = static_checks() if args.mode == "static" else {**static_checks(), "gpu": gpu_checks()}
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
