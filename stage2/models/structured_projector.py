"""Typed, equal-shape 2D/3D structured tokenization and projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from stage2.data.schema import ORACLE_2D_NAMES, ORACLE_3D_NAMES


PROJECTOR_VERSION = "typed-structured-projector-v1"
TOKEN_NAMES = ("object", "goal", "central_obstacle", "side_gate", "robot", "global")
SLOT_NAMES = (
    "position_x",
    "position_y",
    "position_z",
    "extent_x",
    "extent_y",
    "extent_z",
    "scalar_0",
    "scalar_1",
)
TOKEN_COUNT = len(TOKEN_NAMES)
RAW_WIDTH = len(SLOT_NAMES)

# Fixed physical scales, chosen from the declared workspace rather than test data.
# Rows follow TOKEN_NAMES and columns follow SLOT_NAMES.
NORMALIZATION_SCALES = (
    (1.0, 1.0, 0.5, 0.1, 0.1, 0.1, 1.0, 1.0),  # object
    (1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0),  # goal
    (1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0),  # central obstacle
    (1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 1.0),  # side gate; scalar_0=clearance
    (1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.1),  # robot; scalars=max_z,safety
    (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, torch.pi, 1.0),  # global; scalar_0=yaw
)


@dataclass(frozen=True)
class StructuredBatch:
    values: torch.Tensor
    feature_mask: torch.Tensor
    type_ids: torch.Tensor
    token_attention_mask: torch.Tensor
    modality: Literal["2d", "3d"]

    def to(self, device: torch.device | str) -> "StructuredBatch":
        return StructuredBatch(
            values=self.values.to(device),
            feature_mask=self.feature_mask.to(device),
            type_ids=self.type_ids.to(device),
            token_attention_mask=self.token_attention_mask.to(device),
            modality=self.modality,
        )


def _empty_batch(oracle: torch.Tensor, modality: Literal["2d", "3d"]) -> StructuredBatch:
    if oracle.ndim != 2:
        raise ValueError(f"{modality} oracle must have shape [batch, features]")
    if not oracle.is_floating_point():
        raise TypeError(f"{modality} oracle must use a floating dtype")
    batch_size = oracle.shape[0]
    values = oracle.new_zeros((batch_size, TOKEN_COUNT, RAW_WIDTH))
    mask = torch.zeros_like(values, dtype=torch.bool)
    type_ids = torch.arange(TOKEN_COUNT, device=oracle.device).expand(batch_size, -1)
    attention = torch.ones((batch_size, TOKEN_COUNT), dtype=torch.bool, device=oracle.device)
    return StructuredBatch(values, mask, type_ids, attention, modality)


def tokenize_oracle_2d(oracle: torch.Tensor) -> StructuredBatch:
    """Map the exact 15-field 2D schema into the canonical typed slots."""

    if oracle.shape[-1:] != (len(ORACLE_2D_NAMES),):
        raise ValueError(f"2d oracle width must be {len(ORACLE_2D_NAMES)}")
    batch = _empty_batch(oracle, "2d")
    values = batch.values
    mask = batch.feature_mask

    values[:, 0, 0:2] = oracle[:, 0:2]  # object xy
    mask[:, 0, 0:2] = True
    values[:, 1, 0:2] = oracle[:, 2:4]  # goal xy
    mask[:, 1, 0:2] = True
    values[:, 2, 0:2] = oracle[:, 4:6]  # central xy
    values[:, 2, 3:5] = oracle[:, 6:8]  # central half xy
    mask[:, 2, 0:2] = True
    mask[:, 2, 3:5] = True
    values[:, 3, 0:2] = oracle[:, 8:10]  # side gate xy
    values[:, 3, 3:5] = oracle[:, 10:12]  # side gate half xy
    mask[:, 3, 0:2] = True
    mask[:, 3, 3:5] = True
    values[:, 4, 0:2] = oracle[:, 12:14]  # robot base xy
    mask[:, 4, 0:2] = True

    valid = oracle[:, 14] > 0.5
    attention = valid[:, None].expand(-1, TOKEN_COUNT).clone()
    return StructuredBatch(values, mask, batch.type_ids, attention, "2d")


def tokenize_oracle_3d(oracle: torch.Tensor) -> StructuredBatch:
    """Map the exact 27-field 3D schema into the same canonical typed slots."""

    if oracle.shape[-1:] != (len(ORACLE_3D_NAMES),):
        raise ValueError(f"3d oracle width must be {len(ORACLE_3D_NAMES)}")
    batch = _empty_batch(oracle, "3d")
    values = batch.values
    mask = batch.feature_mask

    values[:, 0, 0:6] = oracle[:, 0:6]  # object xyz + half xyz
    mask[:, 0, 0:6] = True
    values[:, 1, 0:3] = oracle[:, 6:9]  # goal xyz
    mask[:, 1, 0:3] = True
    values[:, 2, 0:6] = oracle[:, 9:15]  # central xyz + half xyz
    mask[:, 2, 0:6] = True
    values[:, 3, 0:2] = oracle[:, 15:17]  # side gate xy
    values[:, 3, 3:5] = oracle[:, 17:19]  # side gate half xy
    values[:, 3, 6] = oracle[:, 19]  # clearance z
    mask[:, 3, 0:2] = True
    mask[:, 3, 3:5] = True
    mask[:, 3, 6] = True
    values[:, 4, 0:3] = oracle[:, 20:23]  # robot base xyz
    values[:, 4, 6:8] = oracle[:, 23:25]  # max carry z, safety margin
    mask[:, 4, 0:3] = True
    mask[:, 4, 6:8] = True
    values[:, 5, 6] = oracle[:, 25]  # camera yaw
    mask[:, 5, 6] = True

    valid = oracle[:, 26] > 0.5
    attention = valid[:, None].expand(-1, TOKEN_COUNT).clone()
    return StructuredBatch(values, mask, batch.type_ids, attention, "3d")


class StructuredProjector(nn.Module):
    """One shared architecture for both modalities, with no answer fields."""

    def __init__(self, hidden_dim: int = 128, mlp_ratio: int = 2) -> None:
        super().__init__()
        if hidden_dim <= 0 or mlp_ratio <= 0:
            raise ValueError("hidden_dim and mlp_ratio must be positive")
        inner_dim = hidden_dim * mlp_ratio
        self.hidden_dim = hidden_dim
        self.register_buffer(
            "normalization_scales",
            torch.tensor(NORMALIZATION_SCALES, dtype=torch.float32),
            persistent=True,
        )
        self.value_projector = nn.Sequential(
            nn.Linear(RAW_WIDTH * 2, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, hidden_dim),
        )
        self.type_embedding = nn.Embedding(TOKEN_COUNT, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, batch: StructuredBatch) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (batch.values.shape[0], TOKEN_COUNT, RAW_WIDTH)
        if tuple(batch.values.shape) != expected:
            raise ValueError(f"structured values must have shape {expected}")
        if batch.feature_mask.shape != batch.values.shape:
            raise ValueError("feature mask shape must match values")
        if tuple(batch.type_ids.shape) != expected[:2]:
            raise ValueError("type ids shape must match token axes")
        if tuple(batch.token_attention_mask.shape) != expected[:2]:
            raise ValueError("attention mask shape must match token axes")
        scales = self.normalization_scales.to(dtype=batch.values.dtype)
        masked_values = torch.where(
            batch.feature_mask,
            batch.values / scales.unsqueeze(0),
            torch.zeros_like(batch.values),
        )
        projector_input = torch.cat(
            (masked_values, batch.feature_mask.to(dtype=batch.values.dtype)), dim=-1
        )
        tokens = self.value_projector(projector_input) + self.type_embedding(batch.type_ids)
        tokens = self.output_norm(tokens)
        tokens = tokens * batch.token_attention_mask.unsqueeze(-1).to(tokens.dtype)
        return tokens, batch.token_attention_mask


def parameter_count(module: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return total, trainable
