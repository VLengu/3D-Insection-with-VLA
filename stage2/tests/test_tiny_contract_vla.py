from __future__ import annotations

import unittest

import torch

from stage2.data.action_chunks import PairedBatchSampler, build_action_chunk
from stage2.data.schema import TASK_VARIANTS, oracle_2d, oracle_3d
from stage2.envs.dual_clearance.geometry import generate_pair
from stage2.models.tiny_contract_vla import TaskTokenizer, TinyContractVLA, masked_action_mse


class TinyContractVLATests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.tokenizer = TaskTokenizer(max_length=16)
        self.model = TinyContractVLA(
            tokenizer=self.tokenizer,
            hidden_dim=32,
            heads=4,
            context_layers=1,
            action_chunk=4,
        )
        pair = generate_pair(0, "train", seed=42)
        self.images = torch.rand(2, 3, 256, 256)
        self.tasks = [TASK_VARIANTS[0], TASK_VARIANTS[1]]
        self.state = torch.rand(2, 8)
        self.oracles = {
            "2d": torch.tensor(
                [oracle_2d(pair.over_only), oracle_2d(pair.side_only)], dtype=torch.float32
            ),
            "3d": torch.tensor(
                [oracle_3d(pair.over_only), oracle_3d(pair.side_only)], dtype=torch.float32
            ),
        }

    def test_four_configurations_forward_and_backward(self) -> None:
        for site in ("vl", "action"):
            for modality in ("2d", "3d"):
                self.model.zero_grad(set_to_none=True)
                output = self.model(
                    images=self.images,
                    tasks=self.tasks,
                    robot_state=self.state,
                    oracle=self.oracles[modality],
                    modality=modality,
                    injection_site=site,
                )
                self.assertEqual(output.actions.shape, (2, 4, 7))
                target = torch.zeros_like(output.actions)
                mask = torch.ones(2, 4, dtype=torch.bool)
                masked_action_mse(output.actions, target, mask).backward()
                self.assertIsNotNone(self.model.image_encoder.encoder[0].weight.grad)
                self.assertIsNotNone(self.model.language_embedding.weight.grad)

    def test_masked_loss_ignores_padded_steps(self) -> None:
        prediction = torch.zeros(1, 3, 7)
        target = torch.zeros_like(prediction)
        target[:, 2] = 1000.0
        mask = torch.tensor([[True, True, False]])
        self.assertEqual(float(masked_action_mse(prediction, target, mask)), 0.0)

    def test_action_chunk_padding(self) -> None:
        actions = [torch.full((7,), float(index)) for index in range(3)]
        chunk, mask = build_action_chunk(actions, 1, 4)
        self.assertEqual(chunk.shape, (4, 7))
        self.assertEqual(mask.tolist(), [True, True, False, False])
        self.assertTrue(torch.equal(chunk[2:], torch.zeros(2, 7)))

    def test_tokenizer_is_fixed_and_unknown_safe(self) -> None:
        ids, mask = self.tokenizer.encode(
            ["zyxqv", TASK_VARIANTS[0]], torch.device("cpu")
        )
        self.assertEqual(ids[0, 0].item(), 1)
        self.assertTrue(mask[0, 0])
        self.assertFalse(mask[0, 1:].any())
        forbidden = {"over", "side", "height", "clearance", "route"}
        self.assertFalse(forbidden & set(self.tokenizer.token_to_id))

    def test_paired_batch_sampler_preserves_order(self) -> None:
        sampler = PairedBatchSampler([0, 1, 2, 3, 4, 5], batch_size=4)
        self.assertEqual(list(sampler), [[0, 1, 2, 3], [4, 5]])


if __name__ == "__main__":
    unittest.main()
