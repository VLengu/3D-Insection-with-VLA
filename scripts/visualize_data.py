from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a contact sheet for a generated dataset split.")
    parser.add_argument("--split", choices=("train", "val", "ood", "ood_occluded", "ood_partial"), default="train")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_preview.png"))
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()
    archive = np.load(args.data / f"{args.split}.npz")
    images = archive["images"][: args.count].transpose(0, 2, 3, 1)
    columns = 4
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * 64, rows * 82), "white")
    draw = ImageDraw.Draw(canvas)
    for index, array in enumerate(images):
        x, y = (index % columns) * 64, (index // columns) * 82
        canvas.paste(Image.fromarray(array), (x, y))
        draw.text((x + 2, y + 65), f"#{index}", fill="black")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
