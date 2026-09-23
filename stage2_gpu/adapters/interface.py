"""Frozen tensor contract for the later, pinned SmolVLA runtime adapter.

The full adapter is deliberately validated on Linux/GPU because its two hidden widths
are created only after the pinned checkpoint is loaded. This module prevents dataset and
training code from depending on LeRobot internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from stage2.models.structured_projector import StructuredBatch


InjectionSite = Literal["vl", "action"]
Modality = Literal["2d", "3d"]


@dataclass(frozen=True)
class SmolVLABatch:
    policy_batch: dict[str, torch.Tensor]
    structured: StructuredBatch
    injection_site: InjectionSite
    modality: Modality


def validate_adapter_batch(batch: SmolVLABatch) -> None:
    if batch.injection_site not in {"vl", "action"}:
        raise ValueError("invalid injection site")
    if batch.modality != batch.structured.modality:
        raise ValueError("declared modality differs from structured batch")
    if batch.structured.values.shape[0] == 0:
        raise ValueError("empty structured batch")
    forbidden = {"route_label", "pair_id", "pair_member", "success"}
    leaked = forbidden & set(batch.policy_batch)
    if leaked:
        raise ValueError(f"evaluation labels leaked into policy batch: {sorted(leaked)}")
