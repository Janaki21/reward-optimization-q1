"""Tests for the controlled RLHF proxy experiment."""

from __future__ import annotations

import unittest

import torch

from neural.rlhf_proxy_experiment import (
    generate_contexts,
    perturbed_preference_score,
    response_features,
    true_utility,
    violation_indicator,
)


class RLHFProxyTests(
    unittest.TestCase
):
    def test_context_count(
        self,
    ) -> None:
        contexts = (
            generate_contexts()
        )

        self.assertEqual(
            contexts.shape,
            (16, 4),
        )

    def test_response_shape(
        self,
    ) -> None:
        contexts = (
            generate_contexts()
        )

        features = (
            response_features(
                contexts
            )
        )

        self.assertEqual(
            features.shape,
            (16, 4, 6),
        )

    def test_truthful_response_is_safe(
        self,
    ) -> None:
        contexts = (
            generate_contexts()
        )

        features = (
            response_features(
                contexts
            )
        )

        costs = (
            violation_indicator(
                features
            )
        )

        self.assertTrue(
            torch.all(
                costs[:, 0]
                == 0.0
            )
        )

    def test_exploit_is_violation(
        self,
    ) -> None:
        contexts = (
            generate_contexts()
        )

        features = (
            response_features(
                contexts
            )
        )

        costs = (
            violation_indicator(
                features
            )
        )

        self.assertTrue(
            torch.all(
                costs[:, 3]
                == 1.0
            )
        )

    def test_truthful_utility_is_one(
        self,
    ) -> None:
        contexts = (
            generate_contexts()
        )

        features = (
            response_features(
                contexts
            )
        )

        utility = true_utility(
            features
        )

        self.assertTrue(
            torch.allclose(
                utility[:, 0],
                torch.ones(16),
            )
        )

    def test_perturbation_changes_preferences(
        self,
    ) -> None:
        contexts = (
            generate_contexts()
        )

        features = (
            response_features(
                contexts
            )
        )

        clean = (
            perturbed_preference_score(
                contexts,
                features,
                perturbation=0.0,
            )
        )

        corrupted = (
            perturbed_preference_score(
                contexts,
                features,
                perturbation=1.0,
            )
        )

        self.assertFalse(
            torch.allclose(
                clean,
                corrupted,
            )
        )


if __name__ == "__main__":
    unittest.main()
    