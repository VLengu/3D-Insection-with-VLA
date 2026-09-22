"""Action-chunk view over the local LeRobotDataset with pair-aware sampling."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import Dataset, Sampler

from stage2.data.lerobot_export import EPISODE_SIDECAR, REPO_ID


@dataclass(frozen=True)
class ChunkReference:
    sample_index: int
    episode_index: int
    absolute_index: int
    frame_index: int
    episode_length: int
    split: str
    pair_id: str
    pair_member: str


def build_action_chunk(
    actions: Sequence[torch.Tensor], start: int, chunk_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not actions:
        raise ValueError("actions cannot be empty")
    if start < 0 or start >= len(actions):
        raise IndexError("chunk start is outside the episode")
    prototype = actions[0]
    chunk = prototype.new_zeros((chunk_size, *prototype.shape))
    mask = torch.zeros(chunk_size, dtype=torch.bool, device=prototype.device)
    available = min(chunk_size, len(actions) - start)
    chunk[:available] = torch.stack(tuple(actions[start : start + available]))
    mask[:available] = True
    return chunk, mask


class LeRobotActionChunkDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        root: Path,
        *,
        split: str,
        action_chunk: int = 8,
        frame_stride: int = 1,
    ) -> None:
        if action_chunk <= 0 or frame_stride <= 0:
            raise ValueError("action_chunk and frame_stride must be positive")
        self.root = Path(root)
        self.action_chunk = action_chunk
        self.dataset = LeRobotDataset(REPO_ID, root=self.root)
        sidecar = [
            json.loads(line)
            for line in (self.root / EPISODE_SIDECAR).read_text().splitlines()
            if line.strip()
        ]
        self.references: list[ChunkReference] = []
        for episode in sidecar:
            if episode["split"] != split:
                continue
            for frame_index in range(0, int(episode["length"]), frame_stride):
                self.references.append(
                    ChunkReference(
                        sample_index=len(self.references),
                        episode_index=int(episode["episode_index"]),
                        absolute_index=int(episode["dataset_from_index"]) + frame_index,
                        frame_index=frame_index,
                        episode_length=int(episode["length"]),
                        split=split,
                        pair_id=episode["pair_id"],
                        pair_member=episode["pair_member"],
                    )
                )
        if not self.references:
            raise ValueError(f"No episodes found for split={split!r}")

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, index: int) -> dict[str, Any]:
        reference = self.references[index]
        row = self.dataset[reference.absolute_index]
        available = min(
            self.action_chunk, reference.episode_length - reference.frame_index
        )
        actions = [
            self.dataset[reference.absolute_index + offset]["action"]
            for offset in range(available)
        ]
        action_chunk, action_mask = build_action_chunk(actions, 0, self.action_chunk)
        return {
            "image": row["observation.images.image"],
            "state": row["observation.state"],
            "oracle_2d": row["observation.oracle_2d"],
            "oracle_3d": row["observation.oracle_3d"],
            "task": row["task"],
            "action_chunk": action_chunk,
            "action_mask": action_mask,
            "sample_index": reference.sample_index,
            "episode_index": reference.episode_index,
            "frame_index": reference.frame_index,
            "split": reference.split,
            "pair_id": reference.pair_id,
            "pair_member": reference.pair_member,
        }

    def paired_subset_indices(
        self, *, max_pairs: int, frames_per_member: int
    ) -> list[int]:
        if max_pairs <= 0 or frames_per_member <= 0:
            raise ValueError("max_pairs and frames_per_member must be positive")
        grouped: dict[str, dict[str, list[ChunkReference]]] = {}
        for reference in self.references:
            grouped.setdefault(reference.pair_id, {}).setdefault(
                reference.pair_member, []
            ).append(reference)
        selected: list[int] = []
        for pair_id in sorted(grouped)[:max_pairs]:
            members = grouped[pair_id]
            if set(members) != {"A", "B"}:
                raise ValueError(f"Pair {pair_id} does not contain A and B")
            for offset in range(frames_per_member):
                for member in ("A", "B"):
                    references = members[member]
                    # Interior, evenly spaced frames avoid making a small
                    # overfit probe mostly terminal padding.
                    position = round(
                        (offset + 1) * (len(references) - 1) / (frames_per_member + 1)
                    )
                    selected.append(references[position].sample_index)
        return selected


class PairedBatchSampler(Sampler[list[int]]):
    """Yield deterministic batches while keeping adjacent A/B references together."""

    def __init__(self, indices: Sequence[int], batch_size: int) -> None:
        if batch_size <= 0 or batch_size % 2:
            raise ValueError("batch_size must be a positive even number")
        if len(indices) % 2:
            raise ValueError("paired indices must contain an even number of samples")
        self.indices = tuple(int(index) for index in indices)
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[list[int]]:
        for start in range(0, len(self.indices), self.batch_size):
            yield list(self.indices[start : start + self.batch_size])

    def __len__(self) -> int:
        return (len(self.indices) + self.batch_size - 1) // self.batch_size
