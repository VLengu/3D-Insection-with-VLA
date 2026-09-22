"""Export deterministic MuJoCo expert rollouts as LeRobotDataset v3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from stage2.data.generate import split_seed
from stage2.data.schema import (
    ACTION_NAMES,
    ORACLE_2D_NAMES,
    ORACLE_3D_NAMES,
    STATE_NAMES,
    TASK_VARIANTS,
)
from stage2.envs.dual_clearance.expert import rollout_expert
from stage2.envs.dual_clearance.geometry import SceneGeometry, generate_pair
from stage2.envs.dual_clearance.mujoco_env import DualClearanceMujocoEnv


REPO_ID = "vlengu/dual-clearance-tiny-local"
GENERATOR_VERSION = "dual-clearance-lerobot-m3-v1"
EPISODE_SIDECAR = "meta/dual_clearance_episodes.jsonl"
EXPORT_CONFIG = "meta/dual_clearance_export.json"


def lerobot_features(image_shape: tuple[int, int, int] = (3, 256, 256)) -> dict[str, dict[str, Any]]:
    return {
        "observation.images.image": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["channels", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": list(STATE_NAMES),
        },
        "observation.oracle_2d": {
            "dtype": "float32",
            "shape": (len(ORACLE_2D_NAMES),),
            "names": list(ORACLE_2D_NAMES),
        },
        "observation.oracle_3d": {
            "dtype": "float32",
            "shape": (len(ORACLE_3D_NAMES),),
            "names": list(ORACLE_3D_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES),),
            "names": list(ACTION_NAMES),
        },
    }


def iter_scenes(config: dict[str, Any]) -> Iterator[tuple[int, str, int, SceneGeometry, str]]:
    episode_index = 0
    master_seed = int(config["master_seed"])
    for split, count_value in config["splits"].items():
        base_seed = split_seed(master_seed, split)
        for pair_index in range(int(count_value)):
            pair = generate_pair(pair_index, split, base_seed + pair_index)
            task = TASK_VARIANTS[pair_index % len(TASK_VARIANTS)]
            for scene in pair.members:
                yield episode_index, split, pair_index, scene, task
                episode_index += 1


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def export_lerobot_dataset(config: dict[str, Any], root: Path) -> dict[str, Any]:
    if root.exists():
        raise FileExistsError(
            f"LeRobot requires a new dataset root, but this path already exists: {root}. "
            "Choose a new path or remove the generated directory explicitly."
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    image_shape = tuple(int(value) for value in config["image_shape"])
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=int(config["fps"]),
        features=lerobot_features(image_shape),
        root=root,
        robot_type="franka_panda_mujoco",
        use_videos=False,
        image_writer_threads=4,
        image_writer_processes=0,
    )
    episode_rows: list[dict[str, Any]] = []
    total_frames = 0

    for episode_index, split, pair_index, scene, task in iter_scenes(config):
        env = DualClearanceMujocoEnv(image_size=image_shape[1])
        env.reset(scene)
        expert = rollout_expert(env)
        if not expert.success:
            raise RuntimeError(f"Expert failed before export: {scene.pair_id}/{scene.member}")
        actions = np.asarray(expert.actions, dtype=np.float32)

        env.reset(scene)
        initial_rgb_sha256: str | None = None
        for action in actions:
            observation = env.observation()
            image = env.render()
            if initial_rgb_sha256 is None:
                initial_rgb_sha256 = _sha256_array(image)
            dataset.add_frame(
                {
                    "observation.images.image": image,
                    "observation.state": np.asarray(
                        observation["observation_state"], dtype=np.float32
                    ),
                    "observation.oracle_2d": np.asarray(
                        observation["oracle_2d"], dtype=np.float32
                    ),
                    "observation.oracle_3d": np.asarray(
                        observation["oracle_3d"], dtype=np.float32
                    ),
                    "action": action,
                    "task": task,
                }
            )
            env.step(action)
        final_info = env.info()
        env.close()
        if not final_info["success"]:
            raise RuntimeError(f"Export replay failed: {scene.pair_id}/{scene.member}: {final_info}")

        dataset.save_episode()
        episode_rows.append(
            {
                "episode_index": episode_index,
                "dataset_from_index": total_frames,
                "dataset_to_index": total_frames + len(actions),
                "length": len(actions),
                "split": split,
                "local_pair_index": pair_index,
                "pair_id": scene.pair_id,
                "pair_member": scene.member,
                "route_label": scene.route_label.value,
                "seed": scene.seed,
                "task": task,
                "initial_rgb_sha256": initial_rgb_sha256,
                "actions_sha256": _sha256_array(actions),
                "generator_version": GENERATOR_VERSION,
            }
        )
        total_frames += len(actions)

    dataset.finalize()
    sidecar_path = root / EPISODE_SIDECAR
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in episode_rows)
    )
    export_metadata = {
        "generator_version": GENERATOR_VERSION,
        "repo_id": REPO_ID,
        "format": "LeRobotDataset-v3",
        "use_videos": False,
        "master_seed": int(config["master_seed"]),
        "fps": int(config["fps"]),
        "episodes": len(episode_rows),
        "pairs": len(episode_rows) // 2,
        "frames": total_frames,
        "image_shape_chw": list(image_shape),
        "evaluation_metadata_location": EPISODE_SIDECAR,
        "evaluation_metadata_is_model_input": False,
    }
    (root / EXPORT_CONFIG).write_text(
        json.dumps(export_metadata, indent=2, sort_keys=True) + "\n"
    )
    return export_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = export_lerobot_dataset(json.loads(args.config.read_text()), args.output)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
