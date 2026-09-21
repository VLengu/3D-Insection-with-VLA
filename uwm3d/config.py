from dataclasses import dataclass


@dataclass(frozen=True)
class WorldConfig:
    image_size: int = 64
    horizon: int = 8
    action_dim: int = 2
    action_scale: float = 0.12
    goal_radius: float = 0.075
    max_distractors: int = 3
    cube_height: float = 0.04

    @property
    def z_dim(self) -> int:
        # cube xyz, goal xyz, relative xyz, distance, and xyz+valid for 3 distractors
        return 3 + 3 + 3 + 1 + self.max_distractors * 4


@dataclass(frozen=True)
class ModelConfig:
    horizon: int = 8
    action_dim: int = 2
    z_dim: int = 22
    hidden_dim: int = 256
    depth: int = 4
    heads: int = 4
    diffusion_steps: int = 50
    correction_strength: float = 0.35
    prediction_type: str = "v"
    use_3d_adaln: bool = False
    route_classes: int = 0
    geometry_tokens: bool = False
