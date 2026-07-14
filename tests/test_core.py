import unittest

import numpy as np

from src.agents import AgentConfig, build_agent
from src.analysis import holm_adjustment
from src.envs import build_environment
from src.simulate import run_single_experiment


class EnvironmentTests(unittest.TestCase):
    def test_truth_depends_on_ground_truth(self):
        environment = build_environment(
            "truth_proxy",
            1,
            0.0,
            5.0,
        )

        self.assertEqual(
            environment.step(0, 0).correct,
            1,
        )

        self.assertEqual(
            environment.step(2, 0).correct,
            0,
        )

    def test_monitoring_is_observable(self):
        environment = build_environment(
            "monitoring_shift",
            1,
            1.0,
            5.0,
        )

        self.assertEqual(
            environment.monitored_state(0),
            0,
        )

        self.assertEqual(
            environment.monitored_state(1),
            1,
        )

    def test_hard_shield_respects_admissibility(self):
        agent = build_agent(
            "hard_shield",
            AgentConfig(8),
            1,
        )

        for _ in range(100):
            self.assertEqual(
                agent.select_action(
                    0,
                    [0],
                ),
                0,
            )

    def test_constraint_error_changes_prediction(self):
        environment = build_environment(
            "monitoring_shift",
            1,
            0.0,
            1.0,
            constraint_error=1.0,
        )

        self.assertEqual(
            environment.predicted_admissible_actions(0),
            [1],
        )


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "alpha": 0.1,
            "gamma": 0.95,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay": 0.99,
            "fixed_penalty": 2.0,
            "cost_limit": 0.02,
            "lambda_lr": 0.02,
            "lambda_max": 100.0,
            "learning_curve_points": 10,
        }

    def test_evaluation_is_bounded(self):
        summary, curves = run_single_experiment(
            environment_name="adversarial",
            agent_name="reward_only",
            corruption_probability=1.0,
            corruption_magnitude=5.0,
            constraint_error=0.0,
            seed=11,
            train_steps=500,
            eval_steps=200,
            config=self.config,
        )

        self.assertEqual(
            summary["eval_steps"],
            200,
        )

        self.assertTrue(
            0.0
            <= summary["policy_violation_rate"]
            <= 1.0
        )

        self.assertGreater(
            len(curves),
            1,
        )

    def test_perfect_shield_has_zero_violations(self):
        summary, _ = run_single_experiment(
            environment_name="monitoring_shift",
            agent_name="hard_shield",
            corruption_probability=1.0,
            corruption_magnitude=10.0,
            constraint_error=0.0,
            seed=11,
            train_steps=200,
            eval_steps=200,
            config=self.config,
        )

        self.assertEqual(
            summary["policy_violation_rate"],
            0.0,
        )

    def test_holm_adjustment(self):
        adjusted = holm_adjustment(
            [0.01, 0.03, 0.20]
        )

        self.assertTrue(
            np.all(np.diff(adjusted) >= 0)
        )


if __name__ == "__main__":
    unittest.main()