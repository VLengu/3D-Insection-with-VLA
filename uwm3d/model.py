"""A compact, time-conditioned action diffusion Transformer with 3D residual correction."""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import ModelConfig


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(-math.log(10_000) * torch.arange(half, device=timesteps.device) / max(half - 1, 1))
        values = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat((values.sin(), values.cos()), dim=1)


class AdaLNBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, geometry_cross_attention: bool = False):
        super().__init__()
        self.norm_attention = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.geometry_cross_attention = geometry_cross_attention
        if geometry_cross_attention:
            self.norm_geometry_query = nn.LayerNorm(hidden_dim)
            self.norm_geometry_key_value = nn.LayerNorm(hidden_dim)
            self.geometry_attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
            # Start with a small but nonzero geometry residual. A zero gate
            # would initially block gradients into the attention projection.
            self.geometry_gate_logit = nn.Parameter(torch.tensor(-2.0))

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
        geometry_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(condition).chunk(6, dim=-1)
        attention_input = self._modulate(self.norm_attention(tokens), shift_a, scale_a)
        attended, _ = self.attention(attention_input, attention_input, attention_input, need_weights=False)
        tokens = tokens + torch.sigmoid(gate_a).unsqueeze(1) * attended
        if self.geometry_cross_attention:
            if geometry_tokens is None:
                raise ValueError("Geometry cross-attention requires geometry tokens")
            # The image token remains in the self-attention stream. Only the
            # action tokens query explicit world geometry.
            action_query = self.norm_geometry_query(tokens[:, 1:])
            geometry_key_value = self.norm_geometry_key_value(geometry_tokens)
            geometry_update, _ = self.geometry_attention(
                action_query, geometry_key_value, geometry_key_value, need_weights=False
            )
            tokens = torch.cat((
                tokens[:, :1],
                tokens[:, 1:] + torch.sigmoid(self.geometry_gate_logit) * geometry_update,
            ), dim=1)
        mlp_input = self._modulate(self.norm_mlp(tokens), shift_m, scale_m)
        return tokens + torch.sigmoid(gate_m).unsqueeze(1) * self.mlp(mlp_input)


class HeightGeometryTokenizer(nn.Module):
    """Turn the auditable 18-D height-world record into typed geometry tokens."""

    def __init__(self, hidden_dim: int):
        super().__init__()

        def projector(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )

        self.start_projector = projector(3)
        self.goal_projector = projector(3)
        self.obstacle_projector = projector(6)
        self.relation_projector = projector(10)
        self.camera_projector = projector(3)
        self.type_embedding = nn.Parameter(torch.zeros(1, 5, hidden_dim))
        nn.init.normal_(self.type_embedding, std=0.02)

    def forward(self, z3d: torch.Tensor) -> torch.Tensor:
        if z3d.shape[-1] != 18:
            raise ValueError(f"Height geometry tokenizer expects 18 features, got {z3d.shape[-1]}")
        start = z3d[:, 0:3]
        goal = z3d[:, 3:6]
        relative = z3d[:, 6:9]
        center = z3d[:, 9:12]
        half = z3d[:, 12:15]
        max_lift = z3d[:, 15:16]
        yaw = z3d[:, 16:17]
        valid = z3d[:, 17:18]

        # Encode the actual collision volume used by the environment rather
        # than distance to the obstacle center. Constants are fixed properties
        # of HeightWorldConfig: object radius .035, safety margin .025, and
        # vertical margin .04.
        expanded_xy = half[:, :2] + 0.060
        clearance_z = center[:, 2:3] + half[:, 2:3] + 0.040
        obstacle = torch.cat((center, expanded_xy, clearance_z), dim=-1)

        center_from_start = center - start
        planar_relative = relative[:, :2]
        planar_length = torch.linalg.vector_norm(planar_relative, dim=-1, keepdim=True).clamp_min(1e-6)
        direction = planar_relative / planar_length
        longitudinal = (center_from_start[:, :2] * direction).sum(dim=-1, keepdim=True)
        lateral = (
            center_from_start[:, 1:2] * direction[:, 0:1]
            - center_from_start[:, 0:1] * direction[:, 1:2]
        )
        feasible_gap = max_lift - clearance_z
        relation = torch.cat((
            relative,
            center_from_start,
            longitudinal,
            lateral,
            planar_length,
            feasible_gap,
        ), dim=-1)
        camera = torch.cat((yaw.sin(), yaw.cos(), valid), dim=-1)
        tokens = torch.stack((
            self.start_projector(start),
            self.goal_projector(goal),
            self.obstacle_projector(obstacle),
            self.relation_projector(relation),
            self.camera_projector(camera),
        ), dim=1)
        return tokens + self.type_embedding


class VisualEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.GELU(),
            # The 64x64 input is now an 8x8 feature grid. Do not globally
            # average it: the policy must retain where the cube and target are.
            nn.Flatten(), nn.Linear(128 * 8 * 8, hidden_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image)


class ThreeDCorrectionAdapter(nn.Module):
    def __init__(self, hidden_dim: int, z_dim: int, horizon: int, action_dim: int):
        super().__init__()
        self.z_encoder = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, horizon * action_dim)
        )
        self.horizon = horizon
        self.action_dim = action_dim

    def forward(self, action_tokens: torch.Tensor, z3d: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        pooled_actions = action_tokens.mean(dim=1)
        z_embedding = self.z_encoder(z3d)
        residual = self.head(torch.cat((pooled_actions, z_embedding, time_embedding), dim=-1))
        return residual.view(-1, self.horizon, self.action_dim).tanh()


class ActionDiffusionPolicy(nn.Module):
    """UWM-Lite action branch with optional residual and AdaLN 3D conditioning."""

    def __init__(self, config: ModelConfig, correction_mode: str = "base"):
        super().__init__()
        if correction_mode not in {"base", "early", "middle", "middle_wide", "late", "all"}:
            raise ValueError(f"Unknown correction mode: {correction_mode}")
        self.horizon = config.horizon
        self.action_dim = config.action_dim
        self.diffusion_steps = config.diffusion_steps
        self.correction_mode = correction_mode
        if config.prediction_type not in {"epsilon", "v"}:
            raise ValueError(f"Unknown prediction type: {config.prediction_type}")
        self.prediction_type = config.prediction_type
        self.use_3d_adaln = config.use_3d_adaln
        self.route_classes = config.route_classes
        # getattr keeps checkpoints created before Experiment A-prime
        # loadable after the config schema gains this optional field.
        self.geometry_tokens = getattr(config, "geometry_tokens", False)
        self.correction_strength = config.correction_strength
        self.visual_encoder = VisualEncoder(config.hidden_dim)
        # Training-only auxiliary decoder. It forces the shared visual encoder
        # to retain the cube-to-goal displacement even at high diffusion noise.
        self.visual_goal_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.SiLU(), nn.Linear(config.hidden_dim, 2)
        )
        self.action_encoder = nn.Linear(config.action_dim, config.hidden_dim)
        self.time_encoder = nn.Sequential(SinusoidalTimeEmbedding(config.hidden_dim), nn.Linear(config.hidden_dim, config.hidden_dim), nn.SiLU(), nn.Linear(config.hidden_dim, config.hidden_dim))
        self.adaln_z_encoder = nn.Sequential(
            nn.Linear(config.z_dim, config.hidden_dim), nn.SiLU(), nn.Linear(config.hidden_dim, config.hidden_dim), nn.LayerNorm(config.hidden_dim)
        )
        if self.geometry_tokens:
            self.geometry_tokenizer = HeightGeometryTokenizer(config.hidden_dim)
        else:
            self.geometry_tokenizer = None
        if config.route_classes > 0:
            if self.geometry_tokens:
                self.route_pool_query = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
                nn.init.normal_(self.route_pool_query, std=0.02)
                self.route_pool_attention = nn.MultiheadAttention(
                    config.hidden_dim, config.heads, batch_first=True
                )
                self.route_encoder = nn.Sequential(
                    nn.LayerNorm(config.hidden_dim),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.SiLU(),
                )
            else:
                self.route_pool_query = None
                self.route_pool_attention = None
                self.route_encoder = nn.Sequential(
                    nn.Linear(config.z_dim, config.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                )
            self.route_head = nn.Linear(config.hidden_dim, config.route_classes)
            self.route_embedding = nn.Embedding(config.route_classes, config.hidden_dim)
        else:
            self.route_encoder = None
            self.route_head = None
            self.route_embedding = None
        self.position = nn.Parameter(torch.zeros(1, config.horizon + 1, config.hidden_dim))
        self.blocks = nn.ModuleList(
            AdaLNBlock(config.hidden_dim, config.heads, self.geometry_tokens)
            for _ in range(config.depth)
        )
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.noise_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.correction = ThreeDCorrectionAdapter(config.hidden_dim, config.z_dim, config.horizon, config.action_dim)

    def _gate(self, timesteps: torch.Tensor) -> torch.Tensor:
        high_noise = timesteps.float().div(float(self.diffusion_steps)).view(-1, 1, 1)
        if self.correction_mode == "early":
            return high_noise
        if self.correction_mode == "middle":
            # Bell-shaped in raw diffusion time: zero at both high and low
            # noise, maximum near alpha_bar ~= 0.5. pi/4 normalizes its mean
            # over [0, 1] to 0.5, matching early/late/all correction budgets.
            return (torch.pi / 4.0) * torch.sin(torch.pi * high_noise)
        if self.correction_mode == "middle_wide":
            # A wide middle window: active after the pure-noise stage and
            # before local low-noise refinement. Normalize discretely so its
            # mean gate is exactly 0.5 for the active diffusion schedule.
            softness = 0.07
            raw_gate = torch.sigmoid((high_noise - 0.2) / softness) * torch.sigmoid((0.8 - high_noise) / softness)
            all_steps = torch.arange(1, self.diffusion_steps + 1, device=timesteps.device, dtype=torch.float32)
            all_r = all_steps / float(self.diffusion_steps)
            all_raw = torch.sigmoid((all_r - 0.2) / softness) * torch.sigmoid((0.8 - all_r) / softness)
            return raw_gate * (0.5 / all_raw.mean())
        if self.correction_mode == "late":
            return 1.0 - high_noise
        if self.correction_mode == "all":
            # Uniform conditioning otherwise has twice the time-integrated
            # correction budget of linear early/late gates (mean 1 vs 0.5).
            # Match their expected gate magnitude for a fair timing ablation.
            return 0.5 * torch.ones_like(high_noise)
        return torch.zeros_like(high_noise)

    def _adaln_gate(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Weak geometry modulation at high noise; strong at low noise."""
        high_noise = timesteps.float().div(float(self.diffusion_steps)).view(-1, 1)
        return 0.15 + 0.85 * (1.0 - high_noise)

    def predict_goal_delta(self, image: torch.Tensor) -> torch.Tensor:
        """Predict planar goal-minus-cube displacement from RGB alone."""
        return self.visual_goal_head(self.visual_encoder(image))

    def encode_geometry(self, z3d: torch.Tensor) -> torch.Tensor:
        if self.geometry_tokenizer is None:
            raise ValueError("Geometry tokenization is not enabled in this model")
        return self.geometry_tokenizer(z3d)

    def predict_route(
        self,
        z3d: torch.Tensor,
        geometry_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict global route topology from geometry alone."""
        if self.route_encoder is None or self.route_head is None:
            raise ValueError("Route prediction is not enabled in this model")
        if self.geometry_tokens:
            geometry_tokens = self.encode_geometry(z3d) if geometry_tokens is None else geometry_tokens
            query = self.route_pool_query.expand(len(z3d), -1, -1)
            pooled, _ = self.route_pool_attention(query, geometry_tokens, geometry_tokens, need_weights=False)
            route_features = self.route_encoder(pooled[:, 0])
        else:
            route_features = self.route_encoder(z3d)
        return self.route_head(route_features)

    def forward(
        self,
        image: torch.Tensor,
        noisy_action: torch.Tensor,
        timesteps: torch.Tensor,
        z3d: torch.Tensor | None = None,
        return_diagnostics: bool = False,
        route_override: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        time_embedding = self.time_encoder(timesteps)
        geometry_tokens = None
        if self.geometry_tokens:
            if z3d is None:
                raise ValueError("Geometry cross-attention requires z3d")
            geometry_tokens = self.encode_geometry(z3d)
        if self.route_classes > 0:
            if z3d is None:
                raise ValueError("Route conditioning requires z3d")
            route_logits = self.predict_route(z3d, geometry_tokens)
            if route_override is None:
                route_probabilities = route_logits.softmax(dim=-1)
            elif route_override.ndim == 1:
                route_probabilities = nn.functional.one_hot(
                    route_override.long(), num_classes=self.route_classes
                ).to(dtype=route_logits.dtype)
            else:
                if route_override.shape != route_logits.shape:
                    raise ValueError(
                        f"route_override must have shape {tuple(route_logits.shape)}, "
                        f"got {tuple(route_override.shape)}"
                    )
                route_probabilities = route_override.to(dtype=route_logits.dtype)
            # The route topology is a global decision and must remain strong at
            # high diffusion noise. Use a soft predicted route embedding so
            # diffusion gradients also improve the route encoder.
            route_condition = route_probabilities @ self.route_embedding.weight
            time_embedding = time_embedding + route_condition
        if self.use_3d_adaln:
            if z3d is None:
                raise ValueError("3D AdaLN requires z3d")
            # This is injected before every block's modulation network, not
            # merely added to the final denoising vector.
            time_embedding = time_embedding + self._adaln_gate(timesteps) * self.adaln_z_encoder(z3d)
        image_token = self.visual_encoder(image).unsqueeze(1)
        action_tokens = self.action_encoder(noisy_action)
        tokens = torch.cat((image_token, action_tokens), dim=1) + self.position
        for block in self.blocks:
            tokens = block(tokens, time_embedding, geometry_tokens)
        action_tokens = self.final_norm(tokens[:, 1:])
        base_noise = self.noise_head(action_tokens)
        if self.correction_mode == "base":
            if not return_diagnostics:
                return base_noise
            zeros = torch.zeros(len(base_noise), device=base_noise.device, dtype=base_noise.dtype)
            return base_noise, {
                "correction_ratio": zeros,
                "correction_saturation": zeros,
                "correction_gate": zeros,
            }
        if z3d is None:
            raise ValueError("A 3D correction mode requires z3d")
        correction = self.correction(action_tokens, z3d, time_embedding)
        gate = self._gate(timesteps)
        applied_correction = self.correction_strength * gate * correction
        prediction = base_noise + applied_correction
        if not return_diagnostics:
            return prediction
        base_norm = torch.linalg.vector_norm(base_noise.flatten(1), dim=1)
        correction_norm = torch.linalg.vector_norm(applied_correction.flatten(1), dim=1)
        return prediction, {
            "correction_ratio": correction_norm / base_norm.clamp_min(1e-8),
            "correction_saturation": (correction.abs() > 0.95).float().mean(dim=(1, 2)),
            "correction_gate": gate.flatten(),
        }
