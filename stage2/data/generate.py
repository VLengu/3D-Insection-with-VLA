"""Generate deterministic geometry-only Stage-2 dataset records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from stage2.data.schema import DataContract, record_from_scene
from stage2.envs.dual_clearance.geometry import generate_pair


def split_seed(master_seed: int, split: str) -> int:
    digest = hashlib.sha256(f"{master_seed}:{split}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def generate_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    master_seed = int(config["master_seed"])
    for split, pair_count_value in config["splits"].items():
        pair_count = int(pair_count_value)
        base_seed = split_seed(master_seed, split)
        split_records = []
        for index in range(pair_count):
            pair = generate_pair(index=index, split=split, seed=base_seed + index)
            task_variant = index % 4
            split_records.extend(
                record_from_scene(scene, task_variant=task_variant) for scene in pair.members
            )
        random.Random(base_seed ^ 0x5EED).shuffle(split_records)
        records.extend(split_records)
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    splits: dict[str, dict[str, Any]] = {}
    for record in records:
        split = record["split"]
        entry = splits.setdefault(
            split,
            {"episodes": 0, "pairs": set(), "over": 0, "side": 0},
        )
        entry["episodes"] += 1
        entry["pairs"].add(record["evaluation"]["pair_id"])
        entry[record["evaluation"]["route_label"]] += 1
    serializable = {
        split: {
            "episodes": values["episodes"],
            "pairs": len(values["pairs"]),
            "over": values["over"],
            "side": values["side"],
        }
        for split, values in sorted(splits.items())
    }
    return {
        "records": len(records),
        "pairs": sum(values["pairs"] for values in serializable.values()),
        "splits": serializable,
    }


def write_dataset(config_path: Path, output: Path) -> None:
    config = json.loads(config_path.read_text())
    records = generate_records(config)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    (output / "contract.json").write_text(
        json.dumps(DataContract().to_dict(), indent=2, sort_keys=True) + "\n"
    )
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    summary = summarize(records)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"generated records={summary['records']} pairs={summary['pairs']} "
        f"output={output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_dataset(args.config, args.output)


if __name__ == "__main__":
    main()
