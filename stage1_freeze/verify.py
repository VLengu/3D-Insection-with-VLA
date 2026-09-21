#!/usr/bin/env python3
"""Create or verify the immutable Stage-1 file manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).resolve().with_name("manifest.json")

FROZEN_DIRS = (
    "uwm3d",
    "scripts",
    "outputs",
    "data",
    "data_height",
    "checkpoints",
    "checkpoints_height",
)

FROZEN_FILES = (
    "docs/NEXT_WINDOW_PROMPT_ZH.md",
    "docs/V1_FREEZE.md",
    "docs/HEIGHT_WORLD_V2.md",
    "docs/EXPERIMENT_A_ANALYSIS.md",
    "docs/PROJECT_STATUS_HANDOFF_ZH.md",
    "docs/METRICS_GLOSSARY_ZH.md",
)

IGNORED_NAMES = {".DS_Store"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}
IGNORED_PARTS = {"__pycache__"}


def ignored(path: Path) -> bool:
    return (
        path.name in IGNORED_NAMES
        or path.suffix in IGNORED_SUFFIXES
        or any(part in IGNORED_PARTS for part in path.parts)
    )


def frozen_paths() -> list[Path]:
    paths: list[Path] = []
    for directory in FROZEN_DIRS:
        root = PROJECT_ROOT / directory
        if not root.is_dir():
            raise FileNotFoundError(f"Missing frozen directory: {directory}")
        paths.extend(path for path in root.rglob("*") if path.is_file() and not ignored(path))
    for relative in FROZEN_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen file: {relative}")
        paths.append(path)
    return sorted(set(paths), key=lambda path: path.relative_to(PROJECT_ROOT).as_posix())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot() -> list[dict[str, object]]:
    records = []
    for path in frozen_paths():
        stat = path.stat()
        records.append(
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": stat.st_size,
                "sha256": sha256(path),
            }
        )
    return records


def write_manifest() -> int:
    if MANIFEST_PATH.exists():
        print(f"Refusing to overwrite existing manifest: {MANIFEST_PATH}", file=sys.stderr)
        return 2
    records = snapshot()
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hash_algorithm": "sha256",
        "project_root_name": PROJECT_ROOT.name,
        "frozen_directories": list(FROZEN_DIRS),
        "frozen_files": list(FROZEN_FILES),
        "ignored_names": sorted(IGNORED_NAMES),
        "ignored_suffixes": sorted(IGNORED_SUFFIXES),
        "ignored_parts": sorted(IGNORED_PARTS),
        "file_count": len(records),
        "total_bytes": sum(int(record["size_bytes"]) for record in records),
        "files": records,
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(
        f"Wrote {MANIFEST_PATH.relative_to(PROJECT_ROOT)}: "
        f"{payload['file_count']} files, {payload['total_bytes']} bytes"
    )
    return 0


def verify_manifest() -> int:
    if not MANIFEST_PATH.is_file():
        print(f"Manifest not found: {MANIFEST_PATH}", file=sys.stderr)
        return 2
    manifest = json.loads(MANIFEST_PATH.read_text())
    expected = {record["path"]: record for record in manifest["files"]}
    current_paths = {
        path.relative_to(PROJECT_ROOT).as_posix(): path for path in frozen_paths()
    }

    missing = sorted(set(expected) - set(current_paths))
    unexpected = sorted(set(current_paths) - set(expected))
    size_mismatches: list[str] = []
    hash_mismatches: list[str] = []

    for relative in sorted(set(expected) & set(current_paths)):
        path = current_paths[relative]
        expected_size = int(expected[relative]["size_bytes"])
        if path.stat().st_size != expected_size:
            size_mismatches.append(relative)
            continue
        if sha256(path) != expected[relative]["sha256"]:
            hash_mismatches.append(relative)

    if missing or unexpected or size_mismatches or hash_mismatches:
        print("STAGE1_FREEZE_VERIFY=FAIL")
        for label, values in (
            ("missing", missing),
            ("unexpected", unexpected),
            ("size_mismatch", size_mismatches),
            ("hash_mismatch", hash_mismatches),
        ):
            for value in values:
                print(f"{label}: {value}")
        return 1

    print(
        "STAGE1_FREEZE_VERIFY=PASS "
        f"files={manifest['file_count']} bytes={manifest['total_bytes']}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Create the manifest once; refuses to overwrite an existing manifest.",
    )
    args = parser.parse_args()
    return write_manifest() if args.write else verify_manifest()


if __name__ == "__main__":
    raise SystemExit(main())
