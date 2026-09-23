"""One-way audit guard for locked paired/OOD evaluation splits."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from stage2_gpu.contracts import LOCKED_SPLITS, ContractError


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class EvaluationUnlock:
    checkpoint_sha256: str
    dataset_manifest_sha256: str
    allowed_splits: tuple[str, ...]


def create_unlock(
    destination: Path, *, checkpoint: Path, dataset_manifest: Path
) -> EvaluationUnlock:
    if destination.exists():
        raise FileExistsError(f"unlock record already exists: {destination}")
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "dataset_manifest": str(dataset_manifest.resolve()),
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "allowed_splits": list(LOCKED_SPLITS),
        "purpose": "one-shot locked evaluation after checkpoint selection",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return read_unlock(destination)


def read_unlock(path: Path) -> EvaluationUnlock:
    raw = json.loads(path.read_text())
    allowed = tuple(raw.get("allowed_splits", ()))
    if allowed != LOCKED_SPLITS:
        raise ContractError("unlock split set has been altered")
    return EvaluationUnlock(
        checkpoint_sha256=str(raw["checkpoint_sha256"]),
        dataset_manifest_sha256=str(raw["dataset_manifest_sha256"]),
        allowed_splits=allowed,
    )


def require_split_access(split: str, *, phase: str, unlock: Path | None = None) -> None:
    if split not in LOCKED_SPLITS:
        if split not in {"train", "val"}:
            raise ContractError(f"unknown split: {split}")
        return
    if phase != "locked_evaluate":
        raise PermissionError(f"split {split} is locked during phase {phase}")
    if unlock is None or not unlock.is_file():
        raise PermissionError("locked evaluation requires an immutable unlock record")
    read_unlock(unlock)
