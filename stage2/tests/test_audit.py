from __future__ import annotations

import json
from pathlib import Path
import unittest

from stage2.data.audit import audit_records
from stage2.data.generate import generate_records


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "dataset_tiny.json"


class ShortcutAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = json.loads(CONFIG_PATH.read_text())
        cls.result = audit_records(generate_records(config), split="train")

    def test_audit_passes(self) -> None:
        self.assertEqual(self.result["status"], "PASS")

    def test_non_3d_inputs_are_at_chance(self) -> None:
        self.assertEqual(self.result["task_lookup_accuracy"], 0.5)
        self.assertEqual(self.result["state_best_single_field_accuracy"], 0.5)
        self.assertEqual(self.result["oracle_2d_best_single_field_accuracy"], 0.5)

    def test_3d_requires_relations(self) -> None:
        self.assertLess(self.result["oracle_3d_best_single_field_accuracy"], 0.95)
        self.assertEqual(self.result["oracle_3d_relational_rule_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
