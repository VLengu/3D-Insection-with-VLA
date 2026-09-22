"""Reload a local LeRobotDataset and compare every frame with MuJoCo source data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from stage2.data.lerobot_export import (
    EPISODE_SIDECAR,
    REPO_ID,
    iter_scenes,
    lerobot_features,
)
from stage2.envs.dual_clearance.expert import rollout_expert
from stage2.envs.dual_clearance.mujoco_env import DualClearanceMujocoEnv


def _decoded_uint8(tensor: Any) -> np.ndarray:
    return (
        (tensor.detach().cpu().numpy() * 255.0)
        .round()
        .clip(0, 255)
        .astype(np.uint8)
        .transpose(1, 2, 0)
    )


def _tree_manifest(root: Path) -> tuple[list[str], int]:
    lines: list[str] = []
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "manifest.sha256":
            continue
        payload = path.read_bytes()
        total_bytes += len(payload)
        lines.append(f"{hashlib.sha256(payload).hexdigest()}  {path.relative_to(root)}")
    return lines, total_bytes


def validate_lerobot_dataset(
    config: dict[str, Any], root: Path, evidence_dir: Path
) -> dict[str, Any]:
    sidecar_rows = [
        json.loads(line)
        for line in (root / EPISODE_SIDECAR).read_text().splitlines()
        if line.strip()
    ]
    dataset = LeRobotDataset(REPO_ID, root=root)
    expected_features = lerobot_features(tuple(config["image_shape"]))
    errors: list[str] = []
    max_errors = {"state": 0.0, "oracle_2d": 0.0, "oracle_3d": 0.0, "action": 0.0}
    checked_frames = 0
    paired_initial_images: dict[str, list[np.ndarray]] = {}
    split_episode_counts: dict[str, int] = {}

    if len(sidecar_rows) != dataset.num_episodes:
        errors.append(
            f"sidecar episodes={len(sidecar_rows)} dataset episodes={dataset.num_episodes}"
        )
    if set(expected_features) - set(dataset.features):
        errors.append("required model features are missing")
    if any(key.startswith("evaluation.") for key in dataset.features):
        errors.append("evaluation metadata leaked into frame features")

    scenes = list(iter_scenes(config))
    if len(scenes) != len(sidecar_rows):
        errors.append(f"generated scenes={len(scenes)} sidecar rows={len(sidecar_rows)}")

    for generated, metadata in zip(scenes, sidecar_rows, strict=True):
        episode_index, split, pair_index, scene, task = generated
        if metadata["episode_index"] != episode_index:
            errors.append(f"episode ordering mismatch at {episode_index}")
        if metadata["pair_id"] != scene.pair_id or metadata["pair_member"] != scene.member:
            errors.append(f"pair metadata mismatch at episode {episode_index}")
        split_episode_counts[split] = split_episode_counts.get(split, 0) + 1

        env = DualClearanceMujocoEnv(image_size=int(config["image_shape"][1]))
        env.reset(scene)
        actions = np.asarray(rollout_expert(env).actions, dtype=np.float32)
        env.reset(scene)
        if len(actions) != metadata["length"]:
            errors.append(f"episode {episode_index}: action length mismatch")

        first_decoded: np.ndarray | None = None
        for frame_index, action in enumerate(actions):
            absolute_index = int(metadata["dataset_from_index"]) + frame_index
            row = dataset[absolute_index]
            observation = env.observation()
            source_image = env.render()
            decoded = _decoded_uint8(row["observation.images.image"])
            if first_decoded is None:
                first_decoded = decoded.copy()

            if not np.array_equal(decoded, source_image):
                errors.append(f"episode {episode_index} frame {frame_index}: image mismatch")
            comparisons = {
                "state": (row["observation.state"], observation["observation_state"]),
                "oracle_2d": (row["observation.oracle_2d"], observation["oracle_2d"]),
                "oracle_3d": (row["observation.oracle_3d"], observation["oracle_3d"]),
                "action": (row["action"], action),
            }
            for name, (loaded, source) in comparisons.items():
                loaded_array = loaded.detach().cpu().numpy()
                source_array = np.asarray(source, dtype=np.float32)
                error = float(np.max(np.abs(loaded_array - source_array)))
                max_errors[name] = max(max_errors[name], error)
                if error != 0.0:
                    errors.append(
                        f"episode {episode_index} frame {frame_index}: {name} error={error}"
                    )
            expected_timestamp = np.float32(frame_index / int(config["fps"]))
            if abs(float(row["timestamp"]) - float(expected_timestamp)) > 1e-6:
                errors.append(f"episode {episode_index} frame {frame_index}: timestamp mismatch")
            if int(row["episode_index"]) != episode_index:
                errors.append(f"episode {episode_index} frame {frame_index}: episode index mismatch")
            if int(row["frame_index"]) != frame_index:
                errors.append(f"episode {episode_index} frame {frame_index}: frame index mismatch")
            if int(row["index"]) != absolute_index:
                errors.append(f"episode {episode_index} frame {frame_index}: global index mismatch")
            if row["task"] != task:
                errors.append(f"episode {episode_index} frame {frame_index}: task mismatch")
            env.step(action)
            checked_frames += 1

        if not env.success():
            errors.append(f"episode {episode_index}: replay did not finish successfully")
        env.close()
        if first_decoded is not None:
            paired_initial_images.setdefault(scene.pair_id, []).append(first_decoded)

    unequal_pairs = [
        pair_id
        for pair_id, images in paired_initial_images.items()
        if len(images) != 2 or not np.array_equal(images[0], images[1])
    ]
    errors.extend(f"{pair_id}: decoded initial pair RGB differs" for pair_id in unequal_pairs)
    if checked_frames != len(dataset):
        errors.append(f"checked frames={checked_frames} dataset frames={len(dataset)}")

    manifest_lines, total_bytes = _tree_manifest(root)
    manifest_payload = "\n".join(manifest_lines) + "\n"
    (root / "manifest.sha256").write_text(manifest_payload)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "format": "LeRobotDataset-v3",
        "lerobot_repo_id": REPO_ID,
        "use_videos": False,
        "episodes": dataset.num_episodes,
        "pairs": len(paired_initial_images),
        "frames": len(dataset),
        "checked_frames": checked_frames,
        "split_episode_counts": split_episode_counts,
        "paired_initial_rgb_identical": len(paired_initial_images) - len(unequal_pairs),
        "max_roundtrip_absolute_error": max_errors,
        "dataset_files": len(manifest_lines),
        "dataset_bytes": total_bytes,
        "dataset_tree_sha256": hashlib.sha256(manifest_payload.encode()).hexdigest(),
        "errors": errors[:100],
        "status": "PASS" if not errors else "FAIL",
    }
    (evidence_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    selected_paths = [
        root / "meta/info.json",
        root / "meta/tasks.parquet",
        root / "meta/episodes/chunk-000/file-000.parquet",
        root / EPISODE_SIDECAR,
        root / "manifest.sha256",
    ]
    selected_hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in selected_paths
        if path.is_file()
    }
    (evidence_dir / "selected_hashes.json").write_text(
        json.dumps(selected_hashes, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    summary = validate_lerobot_dataset(
        json.loads(args.config.read_text()), args.dataset, args.evidence
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
