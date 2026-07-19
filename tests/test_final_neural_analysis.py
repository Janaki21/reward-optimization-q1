"""Tests for final neural statistical analysis."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.final_neural_analysis import (
    bootstrap_mean_interval,
    holm_adjust,
    manual_trapezoidal_auc,
    matched_rank_biserial,
    paired_wilcoxon,
)


class FinalAnalysisTests(
    unittest.TestCase
):
    def test_bootstrap_interval_is_ordered(
        self,
    ) -> None:
        lower, upper = (
            bootstrap_mean_interval(
                [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                ],
                resamples=1000,
                seed=11,
            )
        )

        self.assertLessEqual(
            lower,
            upper,
        )

    def test_holm_is_monotonic_in_sorted_order(
        self,
    ) -> None:
        p_values = np.asarray(
            [
                0.01,
                0.03,
                0.02,
            ]
        )

        adjusted = holm_adjust(
            p_values
        )

        order = np.argsort(
            p_values
        )

        sorted_adjusted = (
            adjusted[order]
        )

        self.assertTrue(
            np.all(
                np.diff(
                    sorted_adjusted
                )
                >= 0.0
            )
        )

    def test_manual_auc(
        self,
    ) -> None:
        auc = manual_trapezoidal_auc(
            [
                0.0,
                0.1,
                0.2,
            ],
            [
                0.0,
                0.1,
                0.2,
            ],
        )

        self.assertAlmostEqual(
            auc,
            0.02,
        )

    def test_rank_biserial_direction(
        self,
    ) -> None:
        effect = (
            matched_rank_biserial(
                [
                    1.0,
                    2.0,
                    3.0,
                ]
            )
        )

        self.assertEqual(
            effect,
            1.0,
        )

    def test_identical_pairs(
        self,
    ) -> None:
        result = paired_wilcoxon(
            [
                1.0,
                1.0,
                1.0,
            ],
            [
                1.0,
                1.0,
                1.0,
            ],
        )

        self.assertEqual(
            result["p_value"],
            1.0,
        )

        self.assertEqual(
            result[
                "mean_difference"
            ],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()