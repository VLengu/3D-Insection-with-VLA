"""PIL-only visual check of paired RGB and different 3D expert paths."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _xy(point: np.ndarray, size: int) -> tuple[int, int]:
    return int(point[0] * (size - 1)), int((1.0 - point[1]) * (size - 1))


def _xz(point: np.ndarray, size: int) -> tuple[int, int]:
    return int(point[0] * (size - 1)), int((1.0 - point[2] / 0.5) * (size - 1))


def trajectory_panel(archive: np.lib.npyio.NpzFile, index: int, size: int = 240) -> Image.Image:
    panel = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(panel)
    trajectory = archive["trajectories"][index]
    center = archive["obstacle_centers"][index]
    half = archive["obstacle_halves"][index]
    height = float(archive["obstacle_heights"][index])
    mode = "OVER" if int(archive["path_types"][index]) == 0 else "AROUND"
    half_size = size // 2
    top_points = [_xy(point[:2], half_size) for point in trajectory]
    draw.rectangle((0, 0, size - 1, half_size - 1), outline=(180, 180, 180))
    obstacle_xy_0 = _xy(center - half, half_size)
    obstacle_xy_1 = _xy(center + half, half_size)
    draw.rectangle((obstacle_xy_0[0], obstacle_xy_1[1], obstacle_xy_1[0], obstacle_xy_0[1]), fill=(185, 185, 190))
    draw.line(top_points, fill=(25, 135, 75), width=3)
    xz_points = [tuple(sum(values) for values in zip(_xz(point, half_size), (0, half_size))) for point in trajectory]
    obstacle_left = int((center[0] - half[0]) * (half_size - 1))
    obstacle_right = int((center[0] + half[0]) * (half_size - 1))
    obstacle_top = half_size + int((1.0 - height / 0.5) * (half_size - 1))
    draw.rectangle((obstacle_left, obstacle_top, obstacle_right, size - 1), fill=(185, 185, 190))
    draw.line(xz_points, fill=(210, 95, 35), width=3)
    draw.text((half_size + 6, 8), f"{mode}  h={height:.2f}", fill=(20, 20, 20))
    draw.text((half_size + 6, half_size + 8), "XY / XZ", fill=(20, 20, 20))
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data_height/paired_test.npz"))
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("outputs/height_world_preview.png"))
    args = parser.parse_args()
    archive = np.load(args.data)
    pair_count = min(args.pairs, len(archive["images"]) // 2)
    cell = 240
    canvas = Image.new("RGB", (cell * 3, cell * pair_count), (245, 245, 245))
    for row in range(pair_count):
        low_index, high_index = 2 * row, 2 * row + 1
        rgb = Image.fromarray(archive["images"][low_index].transpose(1, 2, 0)).resize((cell, cell), Image.Resampling.NEAREST)
        canvas.paste(rgb, (0, row * cell))
        canvas.paste(trajectory_panel(archive, low_index, cell), (cell, row * cell))
        canvas.paste(trajectory_panel(archive, high_index, cell), (cell * 2, row * cell))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
