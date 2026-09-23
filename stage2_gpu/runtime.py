"""Guarded GPU entry point.

M5-Mac freezes orchestration and safety checks. Training remains fail-closed until
the real SmolVLA adapter passes G0 on the target Linux/CUDA machine.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stage2_gpu.contracts import assert_four_way_equivalence, load_config
from stage2_gpu.test_lock import require_split_access


def configs_from(directory: Path):
    configs = [load_config(path, resolve_environment=True) for path in sorted(directory.glob("*.json"))]
    assert_four_way_equivalence(configs)
    return configs


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--stage", choices=("smoke", "overfit", "pilot", "formal"), required=True)
    train.add_argument("--config-dir", type=Path, required=True)
    train.add_argument("--seeds", nargs="*", type=int, default=[42])
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config-dir", type=Path, required=True)
    evaluate.add_argument("--unlock", type=Path, required=True)
    args = parser.parse_args()

    configs = configs_from(args.config_dir)
    if args.command == "train":
        for split in ("train", "val"):
            require_split_access(split, phase=args.stage)
        raise RuntimeError(
            "M5-Mac orchestration passed, but execution is intentionally locked until "
            "the pinned SmolVLA adapter completes G0 forward/backward on Linux/CUDA."
        )
    for split in configs[0].raw["dataset"]["locked_splits"]:
        require_split_access(split, phase="locked_evaluate", unlock=args.unlock)
    raise RuntimeError("locked evaluation runner is enabled only after G0-G4 completion")


if __name__ == "__main__":
    main()
