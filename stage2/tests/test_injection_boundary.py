from __future__ import annotations

import io
import unittest

import torch

from stage2.data.schema import oracle_2d, oracle_3d
from stage2.envs.dual_clearance.geometry import generate_pair
from stage2.models.injection_boundary import InjectionBoundaryModel
from stage2.models.structured_projector import (
    parameter_count,
    tokenize_oracle_2d,
    tokenize_oracle_3d,
)


class InjectionBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(123)
        self.model = InjectionBoundaryModel(
            hidden_dim=32, heads=4, context_layers=1, action_chunk=3
        ).eval()
        self.pair = generate_pair(0, "train", seed=42)
        self.images = torch.randn(1, 4, 32)
        self.language = torch.randn(1, 3, 32)
        self.state = torch.randn(1, 8)
        self.two_d = tokenize_oracle_2d(
            torch.tensor([oracle_2d(self.pair.over_only)], dtype=torch.float32)
        )
        self.three_d_a = tokenize_oracle_3d(
            torch.tensor([oracle_3d(self.pair.over_only)], dtype=torch.float32)
        )
        self.three_d_b = tokenize_oracle_3d(
            torch.tensor([oracle_3d(self.pair.side_only)], dtype=torch.float32)
        )

    def _forward(self, structured, site):
        return self.model(
            self.images, self.language, self.state, structured, injection_site=site
        )

    def test_action_side_context_is_oracle_invariant_but_actions_change(self) -> None:
        with torch.no_grad():
            first = self._forward(self.three_d_a, "action")
            second = self._forward(self.three_d_b, "action")
        self.assertTrue(torch.equal(first.contextual_features, second.contextual_features))
        self.assertFalse(torch.equal(first.actions, second.actions))
        self.assertEqual(first.trace.structured_first_seen_at, "action_expert_input")
        self.assertIsNone(first.trace.structured_context_span)

    def test_vl_side_context_and_actions_change_with_oracle(self) -> None:
        with torch.no_grad():
            first = self._forward(self.three_d_a, "vl")
            second = self._forward(self.three_d_b, "vl")
        self.assertFalse(torch.equal(first.contextual_features, second.contextual_features))
        self.assertFalse(torch.equal(first.actions, second.actions))
        self.assertEqual(first.trace.structured_first_seen_at, "contextualizer_input")
        self.assertEqual(first.trace.structured_context_span, (7, 13))

    def test_modes_have_equal_action_memory_and_parameter_budget(self) -> None:
        vl = self._forward(self.two_d, "vl")
        action = self._forward(self.two_d, "action")
        self.assertEqual(vl.trace.action_memory_tokens, action.trace.action_memory_tokens)
        self.assertEqual(vl.trace.structured_action_span, action.trace.structured_action_span)
        self.assertEqual(parameter_count(self.model)[0], parameter_count(self.model)[1])

    def test_hooks_observe_the_expected_first_boundary(self) -> None:
        context_lengths: list[int] = []
        action_lengths: list[int] = []
        context_hook = self.model.contextualizer.register_forward_pre_hook(
            lambda _module, args: context_lengths.append(args[0].shape[1])
        )
        action_hook = self.model.action_expert.register_forward_pre_hook(
            lambda _module, args: action_lengths.append(args[0].shape[1])
        )
        try:
            self._forward(self.three_d_a, "vl")
            self._forward(self.three_d_a, "action")
        finally:
            context_hook.remove()
            action_hook.remove()
        self.assertEqual(context_lengths, [13, 7])
        self.assertEqual(action_lengths, [14, 14])

    def test_all_four_configurations_backpropagate_to_projector(self) -> None:
        self.model.train()
        for site, structured in (
            ("vl", self.two_d),
            ("vl", self.three_d_a),
            ("action", self.two_d),
            ("action", self.three_d_a),
        ):
            self.model.zero_grad(set_to_none=True)
            output = self._forward(structured, site)
            output.actions.square().mean().backward()
            gradients = [
                parameter.grad
                for parameter in self.model.structured_projector.parameters()
            ]
            self.assertTrue(all(gradient is not None for gradient in gradients))
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0)

    def test_checkpoint_roundtrip(self) -> None:
        with torch.no_grad():
            expected = self._forward(self.three_d_a, "vl").actions
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        restored = InjectionBoundaryModel(
            hidden_dim=32, heads=4, context_layers=1, action_chunk=3
        ).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        with torch.no_grad():
            actual = restored(
                self.images,
                self.language,
                self.state,
                self.three_d_a,
                injection_site="vl",
            ).actions
        self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
