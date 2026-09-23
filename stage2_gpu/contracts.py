"""Strict, dependency-light validation for the Gate M5 experiment configs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any


CONFIG_VERSION = "smolvla-structured-injection-v1"
CONDITIONS = {
    "vl_2d": ("vl", "2d"),
    "vl_3d": ("vl", "3d"),
    "a_2d": ("action", "2d"),
    "a_3d": ("action", "3d"),
}
TRAIN_SPLITS = ("train", "val")
LOCKED_SPLITS = (
    "paired_test",
    "height_ood",
    "layout_ood",
    "camera_ood",
    "compound_ood",
)
TINY_MARKERS = ("tiny", "lerobot_tiny", "fixtures/tiny")


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def condition(self) -> str:
        return str(self.raw["condition"])

    @property
    def run_name(self) -> str:
        return str(self.raw["run"]["name"])

    def canonical_sha256(self) -> str:
        payload = json.dumps(self.raw, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


def _required(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ContractError(f"missing {where}.{key}")
    return mapping[key]


def load_config(path: str | Path, *, resolve_environment: bool = False) -> ExperimentConfig:
    config_path = Path(path)
    raw = json.loads(config_path.read_text())
    if raw.get("config_version") != CONFIG_VERSION:
        raise ContractError("config_version mismatch")
    condition = _required(raw, "condition", "config")
    if condition not in CONDITIONS:
        raise ContractError(f"unknown condition: {condition!r}")
    expected_site, expected_modality = CONDITIONS[condition]
    injection = _required(raw, "injection", "config")
    if injection.get("site") != expected_site or injection.get("modality") != expected_modality:
        raise ContractError(f"condition {condition} does not match injection fields")

    dataset = _required(raw, "dataset", "config")
    if dataset.get("role") != "formal":
        raise ContractError("GPU configs must use dataset.role='formal'")
    if tuple(dataset.get("train_visible_splits", ())) != TRAIN_SPLITS:
        raise ContractError("training may see only train and val")
    if tuple(dataset.get("locked_splits", ())) != LOCKED_SPLITS:
        raise ContractError("locked split declaration mismatch")
    root = str(_required(dataset, "root", "dataset"))
    if any(marker in root.lower() for marker in TINY_MARKERS):
        raise ContractError("Tiny validation data cannot be a formal GPU dataset")
    if resolve_environment:
        resolve_path(root)

    policy = _required(raw, "policy", "config")
    for key in ("lerobot_version", "lerobot_revision", "model_id", "model_revision"):
        value = str(_required(policy, key, "policy"))
        if not value or value in {"main", "latest", "HEAD"}:
            raise ContractError(f"policy.{key} must be immutable")
    run = _required(raw, "run", "config")
    if run.get("allow_overwrite") is not False:
        raise ContractError("run.allow_overwrite must be false")
    if int(run.get("seed", -1)) < 0:
        raise ContractError("run.seed must be non-negative")
    return ExperimentConfig(config_path, raw)


def resolve_path(value: str) -> Path:
    if value.startswith("${") and value.endswith("}"):
        variable = value[2:-1]
        resolved = os.environ.get(variable)
        if not resolved:
            raise ContractError(f"required environment variable {variable} is unset")
        return Path(resolved).expanduser().resolve()
    return Path(value).expanduser().resolve()


def assert_four_way_equivalence(configs: list[ExperimentConfig]) -> None:
    if {config.condition for config in configs} != set(CONDITIONS):
        raise ContractError("exactly one config for each 2x2 condition is required")
    reference: dict[str, Any] | None = None
    for config in configs:
        comparable = json.loads(json.dumps(config.raw))
        comparable.pop("condition")
        comparable["run"].pop("name")
        comparable["injection"].pop("site")
        comparable["injection"].pop("modality")
        if reference is None:
            reference = comparable
        elif comparable != reference:
            raise ContractError("four configs differ outside condition/name/site/modality")


def reserve_output(config: ExperimentConfig, root: Path) -> Path:
    output = root / config.run_name
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.snapshot.json").write_text(
        json.dumps(config.raw, indent=2, sort_keys=True) + "\n"
    )
    return output
