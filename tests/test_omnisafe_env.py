"""Tests for the OmniSafe reward-misspecification environments."""

from __future__ import annotations

import unittest

import torch
from gymnasium.spaces import Box

import neural.omnisafe_env
from neural.omnisafe_env import (
    RewardInvariantOmniSafeEnv,
)
from omnisafe.envs.core import make, support_envs


class OmniSafeEnvironmentTests(unittest.TestCase):
    def test_environments_are_registered(
        self,
    ) -> None:
        registered = support_envs()

        expected = {
            "RewardInvariantAdversarial-v0",
            "RewardInvariantTruthProxy-v0",
            "RewardInvariantMonitoringShift-v0",
        }

        self.assertTrue(
            expected.issubset(
                set(registered)
            )
        )

    def test_spaces_are_continuous_boxes(
        self,
    ) -> None:
        environment = (
            RewardInvariantOmniSafeEnv(
                "RewardInvariantAdversarial-v0"
            )
        )

        self.assertIsInstance(
            environment.action_space,
            Box,
        )

        self.assertIsInstance(
            environment.observation_space,
            Box,
        )

        self.assertEqual(
            environment.action_space.shape,
            (1,),
        )

        self.assertEqual(
            environment.observation_space.shape,
            (8,),
        )

    def test_reset_is_reproducible(
        self,
    ) -> None:
        environment = (
            RewardInvariantOmniSafeEnv(
                "RewardInvariantTruthProxy-v0"
            )
        )

        first_observation, _ = (
            environment.reset(seed=11)
        )

        second_observation, _ = (
            environment.reset(seed=11)
        )

        self.assertTrue(
            torch.equal(
                first_observation,
                second_observation,
            )
        )

    def test_step_signature_and_types(
        self,
    ) -> None:
        environment = (
            RewardInvariantOmniSafeEnv(
                "RewardInvariantAdversarial-v0"
            )
        )

        environment.reset(seed=11)

        transition = environment.step(
            torch.tensor(
                [0.5],
                dtype=torch.float32,
            )
        )

        self.assertEqual(
            len(transition),
            6,
        )

        (
            observation,
            reward,
            cost,
            terminated,
            truncated,
            information,
        ) = transition

        self.assertEqual(
            observation.shape,
            (8,),
        )

        self.assertEqual(
            reward.ndim,
            0,
        )

        self.assertEqual(
            cost.ndim,
            0,
        )

        self.assertEqual(
            terminated.ndim,
            0,
        )

        self.assertEqual(
            truncated.ndim,
            0,
        )

        self.assertIn(
            "true_utility",
            information,
        )

        self.assertIn(
            "proxy_true_gap",
            information,
        )

        self.assertIn(
            "binary_action",
            information,
        )

    def test_adversarial_cost(
        self,
    ) -> None:
        environment = (
            RewardInvariantOmniSafeEnv(
                "RewardInvariantAdversarial-v0",
                corruption_probability=1.0,
                corruption_magnitude=5.0,
            )
        )

        environment._state = 1

        _, reward, cost, _, _, information = (
            environment.step(
                torch.tensor([0.5])
            )
        )

        self.assertEqual(
            float(cost.item()),
            1.0,
        )

        self.assertGreater(
            float(reward.item()),
            information["true_utility"],
        )

    def test_truth_cost_uses_ground_truth(
        self,
    ) -> None:
        environment = (
            RewardInvariantOmniSafeEnv(
                "RewardInvariantTruthProxy-v0",
                corruption_probability=1.0,
                corruption_magnitude=5.0,
            )
        )

        environment._state = 1

        _, _, cost, _, _, information = (
            environment.step(
                torch.tensor([0.5])
            )
        )

        self.assertEqual(
            information["ground_truth"],
            0,
        )

        self.assertEqual(
            information["spurious_cue"],
            1,
        )

        self.assertEqual(
            information["binary_action"],
            1,
        )

        self.assertEqual(
            float(cost.item()),
            1.0,
        )

    def test_monitoring_is_observable(
        self,
    ) -> None:
        environment = (
            RewardInvariantOmniSafeEnv(
                "RewardInvariantMonitoringShift-v0"
            )
        )

        environment._state = 2
        unmonitored = environment._observation(
            environment._state
        )

        environment._state = 3
        monitored = environment._observation(
            environment._state
        )

        self.assertFalse(
            torch.equal(
                unmonitored,
                monitored,
            )
        )

    def test_registry_can_construct_environment(
        self,
    ) -> None:
        environment = make(
            "RewardInvariantAdversarial-v0"
        )

        observation, information = (
            environment.reset(seed=23)
        )

        self.assertEqual(
            observation.shape,
            (8,),
        )

        self.assertEqual(
            information["environment"],
            "adversarial",
        )

        environment.close()


if __name__ == "__main__":
    unittest.main()