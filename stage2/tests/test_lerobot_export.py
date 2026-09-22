from __future__ import annotations

import unittest

from stage2.data.lerobot_export import lerobot_features


class LeRobotExportContractTests(unittest.TestCase):
    def test_feature_shapes_and_dtypes(self) -> None:
        features = lerobot_features()
        self.assertEqual(
            set(features),
            {
                "observation.images.image",
                "observation.state",
                "observation.oracle_2d",
                "observation.oracle_3d",
                "action",
            },
        )
        self.assertEqual(features["observation.images.image"]["shape"], (3, 256, 256))
        self.assertEqual(features["observation.state"]["shape"], (8,))
        self.assertEqual(features["observation.oracle_2d"]["shape"], (15,))
        self.assertEqual(features["observation.oracle_3d"]["shape"], (27,))
        self.assertEqual(features["action"]["shape"], (7,))
        self.assertTrue(
            all(
                value["dtype"] in {"image", "float32"}
                for value in features.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
