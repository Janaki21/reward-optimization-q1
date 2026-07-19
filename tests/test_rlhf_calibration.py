"""Tests for RLHF reward-model calibration."""

from __future__ import annotations

import unittest

import torch

from neural.rlhf_proxy_calibrated import (
    normalize_reward_scores,
)


class RLHFCalibrationTests(
    unittest.TestCase
):
    def test_normalization_bounds(
        self,
    ) -> None:
        values = torch.tensor(
            [
                [-2.0, 0.0],
                [2.0, 6.0],
            ]
        )

        normalized = (
            normalize_reward_scores(
                values
            )
        )

        self.assertEqual(
            float(
                normalized.min()
                .item()
            ),
            0.0,
        )

        self.assertEqual(
            float(
                normalized.max()
                .item()
            ),
            1.0,
        )

    def test_normalization_preserves_order(
        self,
    ) -> None:
        values = torch.tensor(
            [
                3.0,
                1.0,
                2.0,
            ]
        )

        normalized = (
            normalize_reward_scores(
                values
            )
        )

        self.assertEqual(
            int(
                torch.argmax(
                    normalized
                ).item()
            ),
            0,
        )

    def test_constant_scores_become_zero(
        self,
    ) -> None:
        values = torch.ones(
            4
        )

        normalized = (
            normalize_reward_scores(
                values
            )
        )

        self.assertTrue(
            torch.equal(
                normalized,
                torch.zeros(4),
            )
        )


if __name__ == "__main__":
    unittest.main()