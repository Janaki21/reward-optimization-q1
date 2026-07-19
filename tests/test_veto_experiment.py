"""Tests for the neural hard-veto operator."""

from __future__ import annotations

import unittest

from neural.veto_experiment import (
    calculate_outcome,
    predicted_admissible_actions,
    project_action,
    true_admissible_actions,
)


class VetoOperatorTests(
    unittest.TestCase
):
    def test_adversarial_risky_action_is_vetoed(
        self,
    ) -> None:
        allowed = (
            true_admissible_actions(
                "adversarial",
                state=1,
            )
        )

        executed, intervened = (
            project_action(
                proposed_action=1,
                admissible_actions=
                    allowed,
            )
        )

        self.assertEqual(
            executed,
            0,
        )
        self.assertEqual(
            intervened,
            1,
        )

    def test_safe_action_is_not_changed(
        self,
    ) -> None:
        executed, intervened = (
            project_action(
                proposed_action=0,
                admissible_actions=[
                    0,
                ],
            )
        )

        self.assertEqual(
            executed,
            0,
        )
        self.assertEqual(
            intervened,
            0,
        )

    def test_perfect_prediction_matches_truth(
        self,
    ) -> None:
        predicted = (
            predicted_admissible_actions(
                environment_name=
                    "truth_proxy",
                state=3,
                constraint_error=0.0,
                constraint_uniform=0.0,
            )
        )

        self.assertEqual(
            predicted,
            [1],
        )

    def test_constraint_error_can_flip_prediction(
        self,
    ) -> None:
        predicted = (
            predicted_admissible_actions(
                environment_name=
                    "truth_proxy",
                state=3,
                constraint_error=0.2,
                constraint_uniform=0.1,
            )
        )

        self.assertEqual(
            predicted,
            [0],
        )

    def test_truth_outcome_uses_ground_truth(
        self,
    ) -> None:
        outcome = calculate_outcome(
            environment_name=
                "truth_proxy",
            state=1,
            action=1,
            corruption_probability=1.0,
            corruption_magnitude=5.0,
            corruption_uniform=0.0,
        )

        self.assertEqual(
            outcome["true_utility"],
            0.0,
        )

        self.assertEqual(
            outcome["violation"],
            1,
        )

        self.assertEqual(
            outcome["proxy_reward"],
            5.0,
        )


if __name__ == "__main__":
    unittest.main()