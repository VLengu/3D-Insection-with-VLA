"""Small end-to-end VLA used only to validate the Stage-2 training contract."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Sequence

import torch
from torch import nn

from stage2.data.schema import TASK_VARIANTS
from stage2.models.injection_boundary import BoundaryOutput, InjectionBoundaryModel, InjectionSite
from stage2.models.structured_projector import tokenize_oracle_2d, tokenize_oracle_3d


TINY_CONTRACT_VLA_VERSION = "tiny-contract-vla-v1"
OracleModality = Literal["2d", "3d"]


class TaskTokenizer:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, max_length: int = 16) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        vocabulary = sorted(
            {
                token
                for task in TASK_VARIANTS
                for token in self._split(task)
            }
        )
        self.token_to_id = {self.PAD: 0, self.UNK: 1}
        self.token_to_id.update(
            {token: index + 2 for index, token in enumerate(vocabulary)}
        )
        self.max_length = max_length

    @staticmethod
    def _split(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, tasks: Sequence[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.zeros((len(tasks), self.max_length), dtype=torch.long, device=device)
        mask = torch.zeros((len(tasks), self.max_length), dtype=torch.bool, device=device)
        for row, task in enumerate(tasks):
            tokens = self._split(task)[: self.max_length]
            encoded = [self.token_to_id.get(token, 1) for token in tokens]
            if encoded:
                ids[row, : len(encoded)] = torch.tensor(encoded, device=device)
                mask[row, : len(encoded)] = True
        return ids, mask


class ImageTokenEncoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=8),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=4),
            nn.GELU(),
            nn.Conv2d(64, hidden_dim, kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1:] != (3, 256, 256):
            raise ValueError("images must have shape [batch,3,256,256]")
        features = self.encoder(images)
        return features.flatten(2).transpose(1, 2)


@dataclass(frozen=True)
class TinyVLAOutput:
    actions: torch.Tensor
    boundary: BoundaryOutput


class TinyContractVLA(nn.Module):
    def __init__(
        self,
        *,
        tokenizer: TaskTokenizer,
        hidden_dim: int = 128,
        heads: int = 4,
        context_layers: int = 2,
        action_chunk: int = 8,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.hidden_dim = hidden_dim
        self.action_chunk = action_chunk
        self.image_encoder = ImageTokenEncoder(hidden_dim)
        self.language_embedding = nn.Embedding(
            tokenizer.vocab_size, hidden_dim, padding_idx=0
        )
        self.language_position = nn.Parameter(
            torch.zeros(1, tokenizer.max_length, hidden_dim)
        )
        nn.init.normal_(self.language_position, std=0.02)
        self.boundary_model = InjectionBoundaryModel(
            hidden_dim=hidden_dim,
            heads=heads,
            context_layers=context_layers,
            action_chunk=action_chunk,
        )

    def forward(
        self,
        *,
        images: torch.Tensor,
        tasks: Sequence[str],
        robot_state: torch.Tensor,
        oracle: torch.Tensor,
        modality: OracleModality,
        injection_site: InjectionSite,
    ) -> TinyVLAOutput:
        image_tokens = self.image_encoder(images)
        token_ids, language_mask = self.tokenizer.encode(tasks, images.device)
        language_tokens = self.language_embedding(token_ids) + self.language_position
        if modality == "2d":
            structured = tokenize_oracle_2d(oracle)
        elif modality == "3d":
            structured = tokenize_oracle_3d(oracle)
        else:
            raise ValueError("modality must be '2d' or '3d'")
        boundary = self.boundary_model(
            image_tokens,
            language_tokens,
            robot_state,
            structured,
            injection_site=injection_site,
            language_attention_mask=language_mask,
        )
        return TinyVLAOutput(boundary.actions, boundary)


def masked_action_mse(
    predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if predictions.shape != targets.shape:
        raise ValueError("prediction and target action shapes must match")
    if mask.shape != predictions.shape[:2]:
        raise ValueError("action mask must match batch and chunk axes")
    weights = mask.unsqueeze(-1).to(dtype=predictions.dtype)
    denominator = weights.sum() * predictions.shape[-1]
    if denominator.item() == 0:
        raise ValueError("action mask must contain at least one valid step")
    return ((predictions - targets).square() * weights).sum() / denominator
