"""Tests for the OmniSafe convergence-pilot utilities."""

from __future__ import annotations

import unittest

import torch

from neural.omnisafe_pilot import (
    build_custom_configuration,
    episode_cost_limit,
    evaluate_actor,
)


class ConstantSafeActor(
    torch.nn.Module
):
    def predict(
        self,
        observation: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        del deterministic

        return torch.full(
            (
                observation.shape[0],
                1,
            ),
            -1.0,
            dtype=torch.float32,
        )


def configuration() -> dict:
    return {
        "steps_per_epoch": 128,
        "update_iters": 2,
        "batch_size": 32,
        "cost_rate_limit": 0.02,
        "episode_length": 64,
        "save_model_freq": 1000,
    }


class OmniSafePilotTests(
    unittest.TestCase
):
    def test_episode_cost_limit(
        self,
    ) -> None:
        self.assertAlmostEqual(
            episode_cost_limit(
                configuration()
            ),
            1.28,
        )

    def test_ppolag_budget_location(
        self,
    ) -> None:
        custom = (
            build_custom_configuration(
                "PPOLag",
                11,
                configuration(),
            )
        )

        self.assertEqual(
            custom[
                "lagrange_cfgs"
            ][
                "cost_limit"
            ],
            1.28,
        )

    def test_cpo_budget_location(
        self,
    ) -> None:
        custom = (
            build_custom_configuration(
                "CPO",
                11,
                configuration(),
            )
        )

        self.assertEqual(
            custom[
                "algo_cfgs"
            ][
                "cost_limit"
            ],
            1.28,
        )

    def test_safe_actor_has_zero_monitoring_violations(
        self,
    ) -> None:
        results = evaluate_actor(
            actor=ConstantSafeActor(),
            environment_id=(
                "RewardInvariantMonitoringShift-v0"
            ),
            seed=11,
            episodes=2,
            episode_length=64,
        )

        self.assertEqual(
            results[
                "heldout_violation_rate"
            ],
            0.0,
        )

        self.assertEqual(
            results[
                "heldout_true_utility"
            ],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()