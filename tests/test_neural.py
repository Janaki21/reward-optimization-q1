"""Tests for the neural DQN experiment layer."""

import unittest

import numpy as np
import torch

from neural.dqn_experiment import (
    DQNAgent,
    DQNConfig,
    QNetwork,
    ReplayBuffer,
    action_mask,
    run_single_dqn,
    state_to_vector,
)
from src.envs import build_environment


class NeuralComponentTests(
    unittest.TestCase
):
    def test_state_vector(self):
        vector = state_to_vector(
            state=2,
            number_of_states=4,
        )

        self.assertEqual(
            vector.shape,
            (4,),
        )

        self.assertEqual(
            vector.sum(),
            1.0,
        )

        self.assertEqual(
            vector[2],
            1.0,
        )

    def test_q_network_shape(self):
        network = QNetwork(
            input_dimension=8,
            number_of_actions=2,
            hidden_dimensions=(16, 16),
        )

        observations = torch.zeros(
            (5, 8),
            dtype=torch.float32,
        )

        output = network(
            observations
        )

        self.assertEqual(
            tuple(output.shape),
            (5, 2),
        )

    def test_replay_buffer(self):
        buffer = ReplayBuffer(
            capacity=100,
            state_dimension=4,
            number_of_actions=2,
            seed=11,
        )

        for index in range(40):
            state = state_to_vector(
                index % 4,
                4,
            )

            next_state = state_to_vector(
                (index + 1) % 4,
                4,
            )

            buffer.add(
                state=state,
                action=index % 2,
                proxy_reward=1.0,
                cost=0.0,
                next_state=next_state,
                next_action_mask=np.ones(
                    2,
                    dtype=np.float32,
                ),
            )

        batch = buffer.sample(
            batch_size=32,
            device=torch.device("cpu"),
        )

        self.assertEqual(
            tuple(
                batch["states"].shape
            ),
            (32, 4),
        )

    def test_shield_mask(self):
        environment = build_environment(
            name="monitoring_shift",
            seed=11,
            corruption_probability=1.0,
            corruption_magnitude=5.0,
            constraint_error=0.0,
        )

        mask = action_mask(
            environment=environment,
            state=0,
            mode="hard_shield_dqn",
        )

        self.assertTrue(
            np.array_equal(
                mask,
                np.array(
                    [1.0, 0.0],
                    dtype=np.float32,
                ),
            )
        )


class NeuralPipelineTests(
    unittest.TestCase
):
    def setUp(self):
        self.configuration = {
            "hidden_dimensions": [
                16,
                16,
            ],
            "learning_rate": 0.001,
            "gamma": 0.95,
            "batch_size": 16,
            "replay_capacity": 500,
            "replay_warmup": 16,
            "target_update_interval": 20,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay_steps": 100,
            "fixed_penalty": 2.0,
            "cost_limit": 0.02,
            "lambda_learning_rate": 0.01,
            "lambda_maximum": 100.0,
            "gradient_clip": 10.0,
            "learning_curve_points": 10,
        }

    def test_single_neural_run(self):
        summary, curves = run_single_dqn(
            environment_name=
                "adversarial",
            mode=
                "reward_only_dqn",
            corruption_probability=
                1.0,
            corruption_magnitude=
                5.0,
            constraint_error=
                0.0,
            seed=
                11,
            training_steps=
                100,
            evaluation_steps=
                50,
            configuration=
                self.configuration,
            device=
                torch.device("cpu"),
        )

        self.assertTrue(
            0.0
            <= summary[
                "policy_violation_rate"
            ]
            <= 1.0
        )

        self.assertGreater(
            len(curves),
            0,
        )

    def test_shielded_neural_run(self):
        summary, _ = run_single_dqn(
            environment_name=
                "monitoring_shift",
            mode=
                "hard_shield_dqn",
            corruption_probability=
                1.0,
            corruption_magnitude=
                5.0,
            constraint_error=
                0.0,
            seed=
                11,
            training_steps=
                100,
            evaluation_steps=
                50,
            configuration=
                self.configuration,
            device=
                torch.device("cpu"),
        )

        self.assertEqual(
            summary[
                "policy_violation_rate"
            ],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()