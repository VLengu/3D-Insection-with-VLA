from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwm3d.height_world import write_height_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paired hidden-height datasets.")
    parser.add_argument("--output", type=Path, default=Path("data_height"))
    parser.add_argument("--train-pairs", type=int, default=6_000)
    parser.add_argument("--eval-pairs", type=int, default=750)
    args = parser.parse_args()
    specs = (
        ("train", args.train_pairs, 101),
        ("val", args.eval_pairs, 102),
        ("height_ood", args.eval_pairs, 103),
        ("layout_ood", args.eval_pairs, 104),
        ("camera_ood", args.eval_pairs, 105),
        ("paired_test", args.eval_pairs, 106),
    )
    for split, pairs, seed in specs:
        path = args.output / f"{split}.npz"
        write_height_dataset(path, pairs, split, seed)
        print(f"wrote {path}: {pairs} pairs / {pairs * 2} samples")


if __name__ == "__main__":
    main()
