"""Small DDPM/DDIM utilities for continuous action chunks."""

from __future__ import annotations

import torch


class DiffusionSchedule:
    def __init__(self, steps: int, device: torch.device):
        self.steps = steps
        # With only 50 training steps, the usual linear [1e-4, 0.02] schedule
        # leaves alpha_bar(T) around 0.6. Starting inference from N(0, I) then
        # mismatches the training distribution. A cosine schedule reaches an
        # effectively fully noised terminal state even for this compact setup.
        offsets = torch.linspace(0, steps, steps + 1, device=device)
        cosine_s = 0.008
        alpha_curve = torch.cos(((offsets / steps + cosine_s) / (1 + cosine_s)) * torch.pi / 2).square()
        alpha_curve = alpha_curve / alpha_curve[0]
        betas = (1.0 - alpha_curve[1:] / alpha_curve[:-1]).clamp(max=0.999)
        alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(alphas, dim=0)

    def alpha_bar(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.alpha_bars[timesteps - 1]

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bar(timesteps).view(-1, 1, 1)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

    def prediction_target(
        self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor, prediction_type: str
    ) -> torch.Tensor:
        """Return epsilon or velocity targets for the selected parameterization."""
        alpha_bar = self.alpha_bar(timesteps).view(-1, 1, 1)
        if prediction_type == "epsilon":
            return noise
        if prediction_type == "v":
            return alpha_bar.sqrt() * noise - (1.0 - alpha_bar).sqrt() * clean
        raise ValueError(f"Unknown prediction type: {prediction_type}")

    def predict_clean(
        self, noisy: torch.Tensor, prediction: torch.Tensor, timesteps: torch.Tensor, prediction_type: str
    ) -> torch.Tensor:
        """Recover x0 from epsilon or velocity parameterization."""
        alpha_bar = self.alpha_bar(timesteps).view(-1, 1, 1)
        if prediction_type == "epsilon":
            return (noisy - (1.0 - alpha_bar).sqrt() * prediction) / alpha_bar.sqrt()
        if prediction_type == "v":
            return alpha_bar.sqrt() * noisy - (1.0 - alpha_bar).sqrt() * prediction
        raise ValueError(f"Unknown prediction type: {prediction_type}")

    def predict_noise(
        self, noisy: torch.Tensor, prediction: torch.Tensor, timesteps: torch.Tensor, prediction_type: str
    ) -> torch.Tensor:
        """Convert a model output to epsilon for the DDIM update."""
        alpha_bar = self.alpha_bar(timesteps).view(-1, 1, 1)
        if prediction_type == "epsilon":
            return prediction
        if prediction_type == "v":
            return (1.0 - alpha_bar).sqrt() * noisy + alpha_bar.sqrt() * prediction
        raise ValueError(f"Unknown prediction type: {prediction_type}")

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        image: torch.Tensor,
        z3d: torch.Tensor | None,
        inference_steps: int = 10,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = image.shape[0]
        device = image.device
        if initial_noise is None:
            action = torch.randn(batch, model.horizon, model.action_dim, device=device)
        else:
            expected_shape = (batch, model.horizon, model.action_dim)
            if tuple(initial_noise.shape) != expected_shape:
                raise ValueError(f"initial_noise must have shape {expected_shape}, got {tuple(initial_noise.shape)}")
            action = initial_noise.to(device=device, dtype=image.dtype)
        selected = torch.linspace(self.steps, 1, inference_steps, device=device).round().long().unique_consecutive()
        for index, timestep in enumerate(selected):
            t = torch.full((batch,), int(timestep.item()), device=device, dtype=torch.long)
            prediction = model(image=image, noisy_action=action, timesteps=t, z3d=z3d)
            alpha_bar = self.alpha_bar(t).view(-1, 1, 1)
            clean = self.predict_clean(action, prediction, t, model.prediction_type)
            eps = self.predict_noise(action, prediction, t, model.prediction_type)
            if index + 1 < len(selected):
                prev_t = torch.full((batch,), int(selected[index + 1].item()), device=device, dtype=torch.long)
                prev_alpha_bar = self.alpha_bar(prev_t).view(-1, 1, 1)
                action = prev_alpha_bar.sqrt() * clean + (1.0 - prev_alpha_bar).sqrt() * eps
            else:
                action = clean
        return action.clamp(-1.0, 1.0)
