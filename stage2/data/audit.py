"""Audit shortcut leakage in a generated Stage-2 dataset."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from stage2.data.schema import ORACLE_2D_NAMES, ORACLE_3D_NAMES, STATE_NAMES
from stage2.data.validate import load_records


def _labels(records: list[dict[str, Any]]) -> list[int]:
    return [1 if record["evaluation"]["route_label"] == "side" else 0 for record in records]


def best_stump_accuracy(values: list[float], labels: list[int]) -> float:
    """Best accuracy of a one-dimensional threshold with either orientation."""
    unique = sorted(set(float(value) for value in values))
    if not unique:
        return 0.0
    thresholds = [unique[0] - 1.0]
    thresholds.extend((left + right) / 2.0 for left, right in zip(unique, unique[1:]))
    thresholds.append(unique[-1] + 1.0)
    best = 0.0
    for threshold in thresholds:
        positive_high = sum((value >= threshold) == bool(label) for value, label in zip(values, labels))
        positive_low = sum((value < threshold) == bool(label) for value, label in zip(values, labels))
        best = max(best, positive_high / len(labels), positive_low / len(labels))
    return best


def _feature_stumps(
    records: list[dict[str, Any]], key: str, names: tuple[str, ...]
) -> dict[str, float]:
    labels = _labels(records)
    return {
        name: best_stump_accuracy(
            [float(record["model_inputs"][key][index]) for record in records], labels
        )
        for index, name in enumerate(names)
    }


def _categorical_lookup_accuracy(records: list[dict[str, Any]], key: str) -> float:
    groups: dict[str, list[int]] = defaultdict(list)
    for record, label in zip(records, _labels(records)):
        groups[str(record["model_inputs"][key])].append(label)
    correct = sum(max(sum(labels), len(labels) - sum(labels)) for labels in groups.values())
    return correct / len(records)


def _relational_route_accuracy(records: list[dict[str, Any]]) -> float:
    index = {name: position for position, name in enumerate(ORACLE_3D_NAMES)}
    correct = 0
    for record in records:
        values = record["model_inputs"]["oracle_3d"]
        central_top = values[index["central_center_z"]] + values[index["central_half_z"]]
        object_half_z = values[index["object_half_z"]]
        margin = values[index["safety_margin"]]
        max_carry = values[index["robot_max_carry_z"]]
        gate_clearance = values[index["side_gate_clearance_z"]]
        over_feasible = central_top + object_half_z + margin <= max_carry + 1e-9
        side_feasible = gate_clearance + 1e-9 >= 2.0 * object_half_z + margin
        predicted = "over" if over_feasible and not side_feasible else "side"
        correct += predicted == record["evaluation"]["route_label"]
    return correct / len(records)


def audit_records(records: list[dict[str, Any]], split: str = "train") -> dict[str, Any]:
    selected = [record for record in records if record["split"] == split]
    if not selected:
        raise ValueError(f"No records found for split={split}")
    state = _feature_stumps(selected, "observation_state", STATE_NAMES)
    two_d = _feature_stumps(selected, "oracle_2d", ORACLE_2D_NAMES)
    three_d = _feature_stumps(selected, "oracle_3d", ORACLE_3D_NAMES)
    result = {
        "split": split,
        "records": len(selected),
        "task_lookup_accuracy": _categorical_lookup_accuracy(selected, "task"),
        "state_best_single_field_accuracy": max(state.values()),
        "oracle_2d_best_single_field_accuracy": max(two_d.values()),
        "oracle_3d_best_single_field_accuracy": max(three_d.values()),
        "oracle_3d_best_single_field": max(three_d, key=three_d.get),
        "oracle_3d_relational_rule_accuracy": _relational_route_accuracy(selected),
        "per_field": {
            "observation_state": state,
            "oracle_2d": two_d,
            "oracle_3d": three_d,
        },
    }
    failures = []
    if result["task_lookup_accuracy"] != 0.5:
        failures.append("task text predicts route above chance")
    if result["state_best_single_field_accuracy"] != 0.5:
        failures.append("initial robot state predicts route above chance")
    if result["oracle_2d_best_single_field_accuracy"] != 0.5:
        failures.append("oracle_2d predicts route above chance")
    if result["oracle_3d_best_single_field_accuracy"] >= 0.95:
        failures.append("a single raw oracle_3d field nearly reveals the route")
    if result["oracle_3d_relational_rule_accuracy"] != 1.0:
        failures.append("registered 3D relational rule does not recover all route labels")
    result["status"] = "PASS" if not failures else "FAIL"
    result["failures"] = failures
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_records(load_records(args.data), split=args.split)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(
        f"STAGE2_SHORTCUT_AUDIT={result['status']} "
        f"2d={result['oracle_2d_best_single_field_accuracy']:.3f} "
        f"3d_single={result['oracle_3d_best_single_field_accuracy']:.3f} "
        f"3d_relational={result['oracle_3d_relational_rule_accuracy']:.3f}"
    )
    if result["failures"]:
        for failure in result["failures"]:
            print(failure)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
