from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwm3d.data import write_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate offline push-to-goal datasets.")
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--train-size", type=int, default=12_000)
    parser.add_argument("--val-size", type=int, default=1_500)
    parser.add_argument("--ood-size", type=int, default=1_500)
    args = parser.parse_args()
    for split, count, seed in (
        ("train", args.train_size, 7),
        ("val", args.val_size, 8),
        ("ood", args.ood_size, 9),
        ("ood_occluded", args.ood_size, 10),
        ("ood_partial", args.ood_size, 11),
    ):
        path = args.output / f"{split}.npz"
        write_dataset(path, count=count, split=split, seed=seed)
        print(f"wrote {path} ({count} samples)")


if __name__ == "__main__":
    main()
