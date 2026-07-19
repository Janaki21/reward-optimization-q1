"""Tests for the OmniSafe experiment runner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from neural.omnisafe_experiment import (
    SUPPORTED_ALGORITHMS,
    build_custom_configuration,
    build_terminal_configuration,
    expected_run_count,
    load_configuration,
    validate_configuration,
)


def valid_configuration() -> dict:
    return {
        "project_name":
            "test",
        "algorithms": [
            "PPO",
            "PPOLag",
            "CPO",
            "RCPO",
        ],
        "environments": [
            "RewardInvariantAdversarial-v0",
        ],
        "random_seeds": [
            11,
        ],
        "total_steps":
            512,
        "steps_per_epoch":
            128,
        "update_iters":
            2,
        "batch_size":
            32,
        "device":
            "cpu",
        "torch_threads":
            1,
        "save_model_freq":
            1,
        "output_dir":
            "outputs/test",
    }


class OmniSafeRunnerTests(
    unittest.TestCase
):
    def test_supported_algorithms(
        self,
    ) -> None:
        self.assertEqual(
            SUPPORTED_ALGORITHMS,
            (
                "PPO",
                "PPOLag",
                "CPO",
                "RCPO",
            ),
        )

    def test_configuration_is_valid(
        self,
    ) -> None:
        validate_configuration(
            valid_configuration()
        )

    def test_unknown_algorithm_fails(
        self,
    ) -> None:
        configuration = (
            valid_configuration()
        )

        configuration[
            "algorithms"
        ] = [
            "UnknownAlgorithm"
        ]

        with self.assertRaises(
            ValueError
        ):
            validate_configuration(
                configuration
            )

    def test_unknown_environment_fails(
        self,
    ) -> None:
        configuration = (
            valid_configuration()
        )

        configuration[
            "environments"
        ] = [
            "UnknownEnvironment-v0"
        ]

        with self.assertRaises(
            ValueError
        ):
            validate_configuration(
                configuration
            )

    def test_invalid_step_division_fails(
        self,
    ) -> None:
        configuration = (
            valid_configuration()
        )

        configuration[
            "total_steps"
        ] = 500

        with self.assertRaises(
            ValueError
        ):
            validate_configuration(
                configuration
            )

    def test_expected_run_count(
        self,
    ) -> None:
        configuration = (
            valid_configuration()
        )

        configuration[
            "random_seeds"
        ] = [
            11,
            17,
        ]

        configuration[
            "environments"
        ] = [
            "RewardInvariantAdversarial-v0",
            "RewardInvariantTruthProxy-v0",
        ]

        self.assertEqual(
            expected_run_count(
                configuration
            ),
            16,
        )

    def test_terminal_configuration(
        self,
    ) -> None:
        terminal = (
            build_terminal_configuration(
                valid_configuration()
            )
        )

        self.assertEqual(
            terminal["total_steps"],
            512,
        )

        self.assertEqual(
            terminal[
                "vector_env_nums"
            ],
            1,
        )

    def test_custom_configuration(
        self,
    ) -> None:
        custom = (
            build_custom_configuration(
                valid_configuration(),
                seed=23,
            )
        )

        self.assertEqual(
            custom["seed"],
            23,
        )

        self.assertEqual(
            custom[
                "algo_cfgs"
            ][
                "steps_per_epoch"
            ],
            128,
        )

        self.assertFalse(
            custom[
                "logger_cfgs"
            ][
                "use_wandb"
            ]
        )

    def test_load_configuration(
        self,
    ) -> None:
        configuration = (
            valid_configuration()
        )

        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "config.yaml"
            )

            with path.open(
                "w",
                encoding="utf-8",
            ) as file:
                yaml.safe_dump(
                    configuration,
                    file,
                )

            loaded = (
                load_configuration(
                    path
                )
            )

        self.assertEqual(
            loaded[
                "project_name"
            ],
            "test",
        )


if __name__ == "__main__":
    unittest.main()