from __future__ import annotations

import io
import unittest

import torch

from stage2.data.schema import oracle_2d, oracle_3d
from stage2.envs.dual_clearance.geometry import generate_pair
from stage2.models.structured_projector import (
    RAW_WIDTH,
    TOKEN_COUNT,
    StructuredProjector,
    parameter_count,
    tokenize_oracle_2d,
    tokenize_oracle_3d,
)


class StructuredProjectorTests(unittest.TestCase):
    def setUp(self) -> None:
        pair = generate_pair(0, "train", seed=42)
        self.two_d = torch.tensor([oracle_2d(pair.over_only)], dtype=torch.float32)
        self.three_d = torch.tensor([oracle_3d(pair.over_only)], dtype=torch.float32)

    def test_modalities_have_identical_output_contract(self) -> None:
        model = StructuredProjector(hidden_dim=32)
        output_2d, attention_2d = model(tokenize_oracle_2d(self.two_d))
        output_3d, attention_3d = model(tokenize_oracle_3d(self.three_d))
        self.assertEqual(output_2d.shape, (1, TOKEN_COUNT, 32))
        self.assertEqual(output_2d.shape, output_3d.shape)
        self.assertEqual(attention_2d.shape, (1, TOKEN_COUNT))
        self.assertTrue(torch.equal(attention_2d, attention_3d))
        self.assertEqual(parameter_count(model)[0], parameter_count(model)[1])

    def test_2d_masked_slots_cannot_change_output(self) -> None:
        torch.manual_seed(7)
        model = StructuredProjector(hidden_dim=32).eval()
        batch = tokenize_oracle_2d(self.two_d)
        baseline = model(batch)[0]
        changed_values = batch.values.clone()
        changed_values[~batch.feature_mask] = 100000.0
        changed = type(batch)(
            changed_values,
            batch.feature_mask,
            batch.type_ids,
            batch.token_attention_mask,
            batch.modality,
        )
        self.assertTrue(torch.equal(baseline, model(changed)[0]))

    def test_3d_height_change_changes_output(self) -> None:
        torch.manual_seed(8)
        model = StructuredProjector(hidden_dim=32).eval()
        changed = self.three_d.clone()
        changed[:, 19] += 0.1  # side gate clearance
        baseline_tokens = model(tokenize_oracle_3d(self.three_d))[0]
        changed_tokens = model(tokenize_oracle_3d(changed))[0]
        self.assertFalse(torch.equal(baseline_tokens[:, 3], changed_tokens[:, 3]))

    def test_invalid_record_masks_every_token(self) -> None:
        invalid = self.two_d.clone()
        invalid[:, -1] = 0.0
        model = StructuredProjector(hidden_dim=32)
        tokens, attention = model(tokenize_oracle_2d(invalid))
        self.assertFalse(attention.any())
        self.assertEqual(torch.count_nonzero(tokens).item(), 0)

    def test_gradient_and_checkpoint_roundtrip(self) -> None:
        torch.manual_seed(9)
        model = StructuredProjector(hidden_dim=32)
        tokens, _ = model(tokenize_oracle_3d(self.three_d.repeat(2, 1)))
        tokens.square().mean().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        self.assertGreater(
            sum(float(parameter.grad.abs().sum()) for parameter in model.parameters()), 0.0
        )
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = StructuredProjector(hidden_dim=32)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        model.eval()
        restored.eval()
        with torch.no_grad():
            original = model(tokenize_oracle_3d(self.three_d))[0]
            loaded = restored(tokenize_oracle_3d(self.three_d))[0]
        self.assertTrue(torch.equal(original, loaded))

    def test_schema_width_and_dtype_are_strict(self) -> None:
        self.assertEqual(tokenize_oracle_2d(self.two_d).values.shape[-1], RAW_WIDTH)
        with self.assertRaisesRegex(ValueError, "width"):
            tokenize_oracle_2d(torch.zeros(1, 16))
        with self.assertRaisesRegex(TypeError, "floating"):
            tokenize_oracle_3d(torch.zeros(1, 27, dtype=torch.int64))


if __name__ == "__main__":
    unittest.main()
