from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from stage2_gpu.contracts import (
    ContractError,
    assert_four_way_equivalence,
    load_config,
    reserve_output,
)
from stage2_gpu.test_lock import create_unlock, require_split_access


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "stage2_gpu" / "configs"


class ConfigContractTests(unittest.TestCase):
    def configs(self):
        return [load_config(path) for path in sorted(CONFIG_DIR.glob("*.json"))]

    def test_four_configs_are_equivalent_except_treatment(self):
        assert_four_way_equivalence(self.configs())

    def test_tiny_is_rejected_as_formal_data(self):
        source = json.loads((CONFIG_DIR / "vl_2d.json").read_text())
        source["dataset"]["root"] = "stage2/generated_data/lerobot_tiny"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            path.write_text(json.dumps(source))
            with self.assertRaises(ContractError):
                load_config(path)

    def test_output_cannot_be_overwritten(self):
        config = self.configs()[0]
        with tempfile.TemporaryDirectory() as temp:
            reserve_output(config, Path(temp))
            with self.assertRaises(FileExistsError):
                reserve_output(config, Path(temp))

    def test_locked_split_requires_unlock(self):
        with self.assertRaises(PermissionError):
            require_split_access("paired_test", phase="train")
        with self.assertRaises(PermissionError):
            require_split_access("paired_test", phase="locked_evaluate")

    def test_unlock_binds_checkpoint_and_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "checkpoint.bin"
            manifest = root / "dataset.sha256"
            checkpoint.write_bytes(b"checkpoint")
            manifest.write_text("dataset\n")
            unlock = root / "unlock.json"
            record = create_unlock(unlock, checkpoint=checkpoint, dataset_manifest=manifest)
            self.assertEqual(len(record.checkpoint_sha256), 64)
            require_split_access("paired_test", phase="locked_evaluate", unlock=unlock)


if __name__ == "__main__":
    unittest.main()
