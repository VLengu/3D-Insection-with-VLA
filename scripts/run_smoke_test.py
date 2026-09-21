"""Fast CPU/MPS-independent test of data, forward pass, loss, and diffusion sampling."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from uwm3d.config import ModelConfig
from uwm3d.data import PushToGoalDataset, write_dataset
from uwm3d.diffusion import DiffusionSchedule
from uwm3d.model import ActionDiffusionPolicy


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tiny.npz"
        write_dataset(path, count=8, split="train", seed=1)
        batch = next(iter(DataLoader(PushToGoalDataset(path), batch_size=4)))
        config = ModelConfig(hidden_dim=64, depth=2, heads=4, prediction_type="v", use_3d_adaln=True)
        model = ActionDiffusionPolicy(config, correction_mode="early")
        schedule = DiffusionSchedule(config.diffusion_steps, torch.device("cpu"))
        timestep = torch.tensor([1, 10, 25, 50])
        noise = torch.randn_like(batch["action"])
        noisy = schedule.add_noise(batch["action"], noise, timestep)
        # At the terminal cosine step alpha_bar is about 1e-6; recovering x0
        # involves subtracting nearly equal float32 values, so use a practical
        # numerical tolerance rather than an exact identity check.
        target = schedule.prediction_target(batch["action"], noise, timestep, "v")
        assert torch.allclose(schedule.predict_clean(noisy, target, timestep, "v"), batch["action"], atol=2e-3)
        assert torch.allclose(schedule.predict_noise(noisy, target, timestep, "v"), noise, atol=2e-3)
        predicted = model(batch["image"], noisy, timestep, batch["z3d"])
        assert predicted.shape == noise.shape
        assert model.predict_goal_delta(batch["image"]).shape == (4, 2)
        sample = schedule.sample(model, batch["image"], batch["z3d"], inference_steps=4)
        assert sample.shape == noise.shape
        assert torch.isfinite(sample).all()
    print("smoke test passed")


if __name__ == "__main__":
    main()
