"""Auditable VL-side and action-side structured-token injection boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from stage2.models.structured_projector import StructuredBatch, StructuredProjector, TOKEN_COUNT


INJECTION_BOUNDARY_VERSION = "vl-action-injection-boundary-v1"
InjectionSite = Literal["vl", "action"]


@dataclass(frozen=True)
class InjectionTrace:
    injection_site: InjectionSite
    structured_first_seen_at: str
    visual_language_tokens: int
    contextualizer_input_tokens: int
    contextual_features_tokens: int
    action_memory_tokens: int
    structured_context_span: tuple[int, int] | None
    structured_action_span: tuple[int, int]
    structured_attention_tokens: int


@dataclass(frozen=True)
class BoundaryOutput:
    actions: torch.Tensor
    contextual_features: torch.Tensor
    contextual_attention_mask: torch.Tensor
    trace: InjectionTrace


class Contextualizer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, layers: int) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        contextual = self.encoder(tokens, src_key_padding_mask=~attention_mask)
        return self.output_norm(contextual)


class ActionExpert(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, action_chunk: int) -> None:
        super().__init__()
        self.action_queries = nn.Parameter(torch.empty(action_chunk, hidden_dim))
        nn.init.normal_(self.action_queries, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=0.0, batch_first=True
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, 7),
        )

    def forward(self, memory: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        queries = self.action_queries.unsqueeze(0).expand(memory.shape[0], -1, -1)
        attended, _ = self.cross_attention(
            queries,
            memory,
            memory,
            key_padding_mask=~attention_mask,
            need_weights=False,
        )
        return self.output(self.query_norm(queries + attended))


class InjectionBoundaryModel(nn.Module):
    """Same parameters for VL/A; only the structured token insertion point changes."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        heads: int = 4,
        context_layers: int = 2,
        action_chunk: int = 4,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.hidden_dim = hidden_dim
        self.structured_projector = StructuredProjector(hidden_dim=hidden_dim)
        self.image_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.language_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.state_projector = nn.Linear(8, hidden_dim)
        self.contextualizer = Contextualizer(hidden_dim, heads, context_layers)
        self.action_expert = ActionExpert(hidden_dim, heads, action_chunk)

    def forward(
        self,
        image_tokens: torch.Tensor,
        language_tokens: torch.Tensor,
        robot_state: torch.Tensor,
        structured: StructuredBatch,
        *,
        injection_site: InjectionSite,
        image_attention_mask: torch.Tensor | None = None,
        language_attention_mask: torch.Tensor | None = None,
    ) -> BoundaryOutput:
        if injection_site not in {"vl", "action"}:
            raise ValueError("injection_site must be 'vl' or 'action'")
        if image_tokens.ndim != 3 or language_tokens.ndim != 3:
            raise ValueError("image and language tokens must have shape [batch,tokens,hidden]")
        batch_size = image_tokens.shape[0]
        if language_tokens.shape[0] != batch_size or robot_state.shape != (batch_size, 8):
            raise ValueError("batch dimensions or robot state shape do not match")
        if image_tokens.shape[-1] != self.hidden_dim or language_tokens.shape[-1] != self.hidden_dim:
            raise ValueError("input token hidden dimension does not match model")
        if structured.values.shape[0] != batch_size:
            raise ValueError("structured batch size does not match token batch")

        device = image_tokens.device
        if language_tokens.device != device or robot_state.device != device:
            raise ValueError("all model inputs must use the same device")
        if structured.values.device != device:
            raise ValueError("structured inputs must use the same device as image tokens")
        if image_attention_mask is None:
            image_attention_mask = torch.ones(
                image_tokens.shape[:2], dtype=torch.bool, device=device
            )
        if language_attention_mask is None:
            language_attention_mask = torch.ones(
                language_tokens.shape[:2], dtype=torch.bool, device=device
            )

        image_tokens = image_tokens + self.image_type
        language_tokens = language_tokens + self.language_type
        visual_language = torch.cat((image_tokens, language_tokens), dim=1)
        visual_language_mask = torch.cat(
            (image_attention_mask, language_attention_mask), dim=1
        )
        visual_language_count = visual_language.shape[1]
        structured_tokens, structured_mask = self.structured_projector(structured)

        if injection_site == "vl":
            context_input = torch.cat((visual_language, structured_tokens), dim=1)
            context_mask = torch.cat((visual_language_mask, structured_mask), dim=1)
            contextual_all = self.contextualizer(context_input, context_mask)
            contextual_features = contextual_all[:, :visual_language_count]
            action_memory = contextual_all
            action_mask_without_state = context_mask
            context_span: tuple[int, int] | None = (
                visual_language_count,
                visual_language_count + TOKEN_COUNT,
            )
            first_seen = "contextualizer_input"
        else:
            contextual_features = self.contextualizer(
                visual_language, visual_language_mask
            )
            contextual_all = contextual_features
            action_memory = torch.cat((contextual_features, structured_tokens), dim=1)
            action_mask_without_state = torch.cat(
                (visual_language_mask, structured_mask), dim=1
            )
            context_span = None
            first_seen = "action_expert_input"

        state_token = self.state_projector(robot_state).unsqueeze(1)
        state_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
        action_memory = torch.cat((action_memory, state_token), dim=1)
        action_mask = torch.cat((action_mask_without_state, state_mask), dim=1)
        actions = self.action_expert(action_memory, action_mask)
        structured_action_start = visual_language_count
        trace = InjectionTrace(
            injection_site=injection_site,
            structured_first_seen_at=first_seen,
            visual_language_tokens=visual_language_count,
            contextualizer_input_tokens=contextual_all.shape[1],
            contextual_features_tokens=contextual_features.shape[1],
            action_memory_tokens=action_memory.shape[1],
            structured_context_span=context_span,
            structured_action_span=(
                structured_action_start,
                structured_action_start + TOKEN_COUNT,
            ),
            structured_attention_tokens=int(structured_mask.sum().detach().cpu()),
        )
        return BoundaryOutput(actions, contextual_features, visual_language_mask, trace)
