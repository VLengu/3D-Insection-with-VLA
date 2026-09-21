"""Validate Stage-2 records, counterfactual pairs, and split isolation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from stage2.data.schema import ORACLE_3D_NAMES, validate_record


ALLOWED_PAIRWISE_3D_CHANGES = frozenset(
    {
        "central_center_z",
        "central_half_z",
        "side_gate_clearance_z",
    }
)


def _changed_fields(first: list[float], second: list[float]) -> set[str]:
    return {
        name
        for name, left, right in zip(ORACLE_3D_NAMES, first, second, strict=True)
        if abs(float(left) - float(right)) > 1e-12
    }


def validate_records(records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    episode_ids: set[str] = set()
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    layouts_by_split: dict[str, set[str]] = defaultdict(set)

    for record in records:
        record_errors = validate_record(record)
        errors.extend(f"{record.get('episode_id', '?')}: {error}" for error in record_errors)
        episode_id = record.get("episode_id")
        if episode_id in episode_ids:
            errors.append(f"duplicate episode_id: {episode_id}")
        episode_ids.add(episode_id)
        evaluation = record.get("evaluation", {})
        pair_id = evaluation.get("pair_id")
        if pair_id is not None:
            pairs[str(pair_id)].append(record)
        layout_id = evaluation.get("base_layout_id")
        if layout_id is not None:
            layouts_by_split[str(record.get("split"))].add(str(layout_id))

    for pair_id, members in sorted(pairs.items()):
        if len(members) != 2:
            errors.append(f"{pair_id}: expected 2 members, got {len(members)}")
            continue
        members = sorted(members, key=lambda item: item["evaluation"]["pair_member"])
        first, second = members
        eval_first, eval_second = first["evaluation"], second["evaluation"]
        labels = {eval_first["route_label"], eval_second["route_label"]}
        if labels != {"over", "side"}:
            errors.append(f"{pair_id}: route labels are not balanced: {labels}")
        if first["model_inputs"]["task"] != second["model_inputs"]["task"]:
            errors.append(f"{pair_id}: task text differs inside pair")
        if first["model_inputs"]["observation_state"] != second["model_inputs"]["observation_state"]:
            errors.append(f"{pair_id}: initial robot state differs inside pair")
        if first["model_inputs"]["oracle_2d"] != second["model_inputs"]["oracle_2d"]:
            errors.append(f"{pair_id}: oracle_2d differs inside pair")
        changed = _changed_fields(
            first["model_inputs"]["oracle_3d"], second["model_inputs"]["oracle_3d"]
        )
        if changed != ALLOWED_PAIRWISE_3D_CHANGES:
            errors.append(
                f"{pair_id}: changed 3D fields={sorted(changed)} "
                f"expected={sorted(ALLOWED_PAIRWISE_3D_CHANGES)}"
            )
        for member in members:
            evaluation = member["evaluation"]
            if bool(evaluation["over_feasible"]) == bool(evaluation["side_feasible"]):
                errors.append(f"{member['episode_id']}: routes are not mutually exclusive")
            expected = "over" if evaluation["over_feasible"] else "side"
            if evaluation["route_label"] != expected:
                errors.append(f"{member['episode_id']}: route label contradicts feasibility")

    split_names = sorted(layouts_by_split)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = layouts_by_split[left] & layouts_by_split[right]
            if overlap:
                errors.append(f"layout leakage between {left} and {right}: {sorted(overlap)[:3]}")

    route_counts = Counter(record["evaluation"]["route_label"] for record in records)
    if route_counts["over"] != route_counts["side"]:
        errors.append(f"route imbalance: {dict(route_counts)}")
    return errors


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    records = load_records(args.data)
    errors = validate_records(records)
    if errors:
        print("STAGE2_DATA_VALIDATE=FAIL")
        for error in errors:
            print(error)
        raise SystemExit(1)
    pairs = len({record["evaluation"]["pair_id"] for record in records})
    print(f"STAGE2_DATA_VALIDATE=PASS records={len(records)} pairs={pairs}")


if __name__ == "__main__":
    main()
