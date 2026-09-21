from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from stage2.data.generate import generate_records, summarize
from stage2.data.validate import validate_records


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "dataset_tiny.json"


class DatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text())
        cls.records = generate_records(cls.config)

    def test_tiny_profile_size_and_balance(self) -> None:
        summary = summarize(self.records)
        expected_pairs = sum(self.config["splits"].values())
        self.assertEqual(summary["pairs"], expected_pairs)
        self.assertEqual(summary["records"], expected_pairs * 2)
        for values in summary["splits"].values():
            self.assertEqual(values["over"], values["side"])

    def test_full_validation_passes(self) -> None:
        self.assertEqual(validate_records(self.records), [])

    def test_generation_is_byte_deterministic(self) -> None:
        regenerated = generate_records(self.config)
        first = json.dumps(self.records, sort_keys=True, separators=(",", ":"))
        second = json.dumps(regenerated, sort_keys=True, separators=(",", ":"))
        self.assertEqual(first, second)

    def test_pairwise_xy_change_is_rejected(self) -> None:
        corrupted = deepcopy(self.records)
        target_pair = corrupted[0]["evaluation"]["pair_id"]
        target = next(
            record for record in corrupted if record["evaluation"]["pair_id"] == target_pair
        )
        target["model_inputs"]["oracle_2d"][0] += 0.01
        errors = validate_records(corrupted)
        self.assertTrue(any("oracle_2d differs" in error for error in errors))

    def test_split_layout_ids_are_disjoint(self) -> None:
        by_split: dict[str, set[str]] = {}
        for record in self.records:
            by_split.setdefault(record["split"], set()).add(
                record["evaluation"]["base_layout_id"]
            )
        names = sorted(by_split)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                self.assertFalse(by_split[left] & by_split[right])


if __name__ == "__main__":
    unittest.main()
