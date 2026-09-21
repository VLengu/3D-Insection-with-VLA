"""Summarize a fixed-seed timing-ablation sweep without extra dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def format_stat(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{mean(values):.3f} ± {stdev(values):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("checkpoints/scaled"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--modes", nargs="+", default=["base", "early", "middle", "late", "all"])
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["val_success", "ood_success", "occluded_ood_success", "occluded_ood_final_distance"],
    )
    args = parser.parse_args()
    metrics = args.metrics
    print("| Mode | " + " | ".join(metrics) + " |")
    print("|---|" + "---|" * len(metrics))
    for mode in args.modes:
        records = []
        for seed in args.seeds:
            path = args.root / f"{mode}_seed{seed}" / f"history_{mode}_seed{seed}.json"
            if not path.exists():
                raise FileNotFoundError(path)
            records.append(json.loads(path.read_text())[-1])
        fields = [format_stat([record[name] for record in records]) for name in metrics]
        print(f"| {mode} | " + " | ".join(fields) + " |")


if __name__ == "__main__":
    main()
