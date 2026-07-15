"""Automated tests for the controlled experimental pipeline."""

import unittest

import numpy as np
import pandas as pd

from src.agents import (
    AgentConfig,
    build_agent,
)
from src.analysis import (
    bootstrap_confidence_interval,
    calculate_perturbation_auc,
    holm_adjustment,
    run_full_analysis,
)
from src.envs import build_environment
from src.simulate import run_single_experiment


class EnvironmentTests(unittest.TestCase):
    def test_truth_depends_on_ground_truth(self):
        environment = build_environment(
            name="truth_proxy",
            seed=1,
            corruption_probability=0.0,
            corruption_magnitude=5.0,
        )

        truth_zero_transition = (
            environment.step(
                state=0,
                action=0,
            )
        )

        truth_one_transition = (
            environment.step(
                state=2,
                action=0,
            )
        )

        self.assertEqual(
            truth_zero_transition.correct,
            1,
        )

        self.assertEqual(
            truth_one_transition.correct,
            0,
        )

    def test_monitoring_is_observable(self):
        environment = build_environment(
            name="monitoring_shift",
            seed=1,
            corruption_probability=1.0,
            corruption_magnitude=5.0,
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
            name="hard_shield",
            config=AgentConfig(
                n_states=8
            ),
            seed=1,
        )

        for _ in range(100):
            selected_action = (
                agent.select_action(
                    state=0,
                    admissible_actions=[0],
                    explore=True,
                )
            )

            self.assertEqual(
                selected_action,
                0,
            )

    def test_constraint_error_changes_prediction(self):
        environment = build_environment(
            name="monitoring_shift",
            seed=1,
            corruption_probability=0.0,
            corruption_magnitude=1.0,
            constraint_error=1.0,
        )

        predicted_actions = (
            environment
            .predicted_admissible_actions(
                state=0
            )
        )

        self.assertEqual(
            predicted_actions,
            [1],
        )


class StatisticalTests(unittest.TestCase):
    def test_bootstrap_interval_is_ordered(self):
        lower_bound, upper_bound = (
            bootstrap_confidence_interval(
                [0.1, 0.2, 0.3, 0.4]
            )
        )

        self.assertLessEqual(
            lower_bound,
            upper_bound,
        )

    def test_holm_adjustment_is_monotonic(self):
        adjusted_values = holm_adjustment(
            [0.01, 0.03, 0.20]
        )

        self.assertTrue(
            np.all(
                np.diff(
                    adjusted_values
                ) >= 0
            )
        )

    def test_perturbation_auc_is_calculated(self):
        sample_data = pd.DataFrame(
            {
                "environment": [
                    "adversarial",
                    "adversarial",
                ],
                "agent": [
                    "reward_only",
                    "reward_only",
                ],
                "seed": [
                    11,
                    11,
                ],
                "corruption_probability": [
                    0.0,
                    1.0,
                ],
                "corruption_magnitude": [
                    5.0,
                    5.0,
                ],
                "constraint_error": [
                    0.0,
                    0.0,
                ],
                "policy_violation_rate": [
                    0.0,
                    1.0,
                ],
                "false_answer_rate": [
                    0.0,
                    1.0,
                ],
                "monitoring_gap": [
                    0.0,
                    1.0,
                ],
            }
        )

        results = (
            calculate_perturbation_auc(
                sample_data
            )
        )

        self.assertEqual(
            len(results),
            1,
        )

        self.assertAlmostEqual(
            results[
                "failure_auc"
            ].iloc[0],
            0.5,
        )

    def test_full_analysis_handles_small_smoke_sample(self):
        rows = []

        for agent in [
            "reward_only",
            "hard_shield",
        ]:
            for probability in [
                0.0,
                1.0,
            ]:
                rows.append(
                    {
                        "environment":
                            "adversarial",
                        "agent":
                            agent,
                        "seed":
                            11,
                        "corruption_probability":
                            probability,
                        "corruption_magnitude":
                            5.0,
                        "constraint_error":
                            0.0,
                        "proxy_return":
                            1.0,
                        "true_utility":
                            1.0 - probability,
                        "policy_violation_rate":
                            (
                                probability
                                if agent
                                == "reward_only"
                                else 0.0
                            ),
                        "false_answer_rate":
                            0.0,
                        "monitoring_gap":
                            np.nan,
                        "proxy_true_gap":
                            probability,
                    }
                )

        sample_data = pd.DataFrame(
            rows
        )

        outputs = run_full_analysis(
            sample_data
        )

        self.assertIn(
            "aggregated_statistics",
            outputs,
        )

        self.assertIn(
            "paired_tests",
            outputs,
        )

        self.assertIn(
            "perturbation_auc",
            outputs,
        )

        self.assertIn(
            "auc_summary",
            outputs,
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
        summary, learning_curves = (
            run_single_experiment(
                environment_name=
                    "adversarial",
                agent_name=
                    "reward_only",
                corruption_probability=
                    1.0,
                corruption_magnitude=
                    5.0,
                constraint_error=
                    0.0,
                seed=
                    11,
                train_steps=
                    500,
                eval_steps=
                    200,
                config=
                    self.config,
            )
        )

        self.assertEqual(
            summary["eval_steps"],
            200,
        )

        self.assertTrue(
            0.0
            <= summary[
                "policy_violation_rate"
            ]
            <= 1.0
        )

        self.assertGreater(
            len(learning_curves),
            1,
        )

    def test_perfect_shield_has_zero_violations(self):
        summary, _ = run_single_experiment(
            environment_name=
                "monitoring_shift",
            agent_name=
                "hard_shield",
            corruption_probability=
                1.0,
            corruption_magnitude=
                10.0,
            constraint_error=
                0.0,
            seed=
                11,
            train_steps=
                200,
            eval_steps=
                200,
            config=
                self.config,
        )

        self.assertEqual(
            summary[
                "policy_violation_rate"
            ],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()