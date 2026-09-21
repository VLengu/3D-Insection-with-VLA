from __future__ import annotations

from copy import deepcopy
import unittest

from stage2.data.schema import (
    ACTION_NAMES,
    FORBIDDEN_MODEL_KEYS,
    MODEL_INPUT_KEYS,
    ORACLE_2D_NAMES,
    ORACLE_3D_NAMES,
    STATE_NAMES,
    record_from_scene,
    validate_record,
)
from stage2.envs.dual_clearance.geometry import generate_pair


class SchemaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = generate_pair(0, "train", seed=42).over_only
        self.record = record_from_scene(self.scene, task_variant=0)

    def test_dimensions(self) -> None:
        inputs = self.record["model_inputs"]
        self.assertEqual(len(inputs["observation_state"]), len(STATE_NAMES))
        self.assertEqual(len(inputs["oracle_2d"]), len(ORACLE_2D_NAMES))
        self.assertEqual(len(inputs["oracle_3d"]), len(ORACLE_3D_NAMES))
        self.assertEqual(len(ACTION_NAMES), 7)

    def test_model_inputs_are_an_exact_allowlist(self) -> None:
        keys = set(self.record["model_inputs"])
        self.assertEqual(keys, MODEL_INPUT_KEYS)
        self.assertFalse(keys & FORBIDDEN_MODEL_KEYS)

    def test_valid_record(self) -> None:
        self.assertEqual(validate_record(self.record), [])

    def test_route_label_leak_is_rejected(self) -> None:
        corrupted = deepcopy(self.record)
        corrupted["model_inputs"]["route_label"] = "over"
        errors = validate_record(corrupted)
        self.assertTrue(any("leaked" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
