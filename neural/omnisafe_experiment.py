"""Run reproducible OmniSafe constrained-RL experiments.

Supported algorithms:
    PPO
    PPOLag
    CPO
    RCPO

This runner is first used for integration smoke tests. Smoke results are not
intended for statistical inference or manuscript claims.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

import neural.omnisafe_env
import omnisafe
from neural.omnisafe_env import ENVIRONMENT_IDS


SUPPORTED_ALGORITHMS = (
    "PPO",
    "PPOLag",
    "CPO",
    "RCPO",
)


def load_configuration(
    path: str | Path,
) -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as configuration_file:
        configuration = yaml.safe_load(
            configuration_file
        )

    if not isinstance(
        configuration,
        dict,
    ):
        raise ValueError(
            "The configuration must contain a YAML mapping."
        )

    validate_configuration(
        configuration
    )

    return configuration


def validate_configuration(
    configuration: dict[str, Any],
) -> None:
    required_fields = {
        "project_name",
        "algorithms",
        "environments",
        "random_seeds",
        "total_steps",
        "steps_per_epoch",
        "update_iters",
        "batch_size",
        "device",
        "torch_threads",
        "output_dir",
    }

    missing_fields = sorted(
        required_fields.difference(
            configuration
        )
    )

    if missing_fields:
        raise ValueError(
            "Missing configuration fields: "
            + ", ".join(missing_fields)
        )

    algorithms = list(
        configuration["algorithms"]
    )

    unknown_algorithms = sorted(
        set(algorithms).difference(
            SUPPORTED_ALGORITHMS
        )
    )

    if unknown_algorithms:
        raise ValueError(
            "Unsupported algorithms: "
            + ", ".join(
                unknown_algorithms
            )
        )

    environments = list(
        configuration["environments"]
    )

    unknown_environments = sorted(
        set(environments).difference(
            ENVIRONMENT_IDS
        )
    )

    if unknown_environments:
        raise ValueError(
            "Unsupported environments: "
            + ", ".join(
                unknown_environments
            )
        )

    if not configuration["random_seeds"]:
        raise ValueError(
            "At least one random seed is required."
        )

    positive_integer_fields = [
        "total_steps",
        "steps_per_epoch",
        "update_iters",
        "batch_size",
        "torch_threads",
    ]

    for field in positive_integer_fields:
        value = int(
            configuration[field]
        )

        if value <= 0:
            raise ValueError(
                f"{field} must be positive."
            )

    total_steps = int(
        configuration["total_steps"]
    )

    steps_per_epoch = int(
        configuration[
            "steps_per_epoch"
        ]
    )

    if (
        total_steps
        % steps_per_epoch
        != 0
    ):
        raise ValueError(
            "total_steps must be divisible by steps_per_epoch."
        )


def build_terminal_configuration(
    configuration: dict[str, Any],
) -> dict[str, Any]:
    return {
        "parallel": 1,
        "total_steps": int(
            configuration[
                "total_steps"
            ]
        ),
        "device": str(
            configuration["device"]
        ),
        "vector_env_nums": 1,
        "torch_threads": int(
            configuration[
                "torch_threads"
            ]
        ),
    }


def build_custom_configuration(
    configuration: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "algo_cfgs": {
            "steps_per_epoch": int(
                configuration[
                    "steps_per_epoch"
                ]
            ),
            "update_iters": int(
                configuration[
                    "update_iters"
                ]
            ),
            "batch_size": int(
                configuration[
                    "batch_size"
                ]
            ),
        },
        "logger_cfgs": {
            "use_wandb": False,
            "save_model_freq": int(
                configuration.get(
                    "save_model_freq",
                    1,
                )
            ),
        },
    }


def expected_run_count(
    configuration: dict[str, Any],
) -> int:
    return (
        len(
            configuration[
                "algorithms"
            ]
        )
        * len(
            configuration[
                "environments"
            ]
        )
        * len(
            configuration[
                "random_seeds"
            ]
        )
    )


def run_single_experiment(
    algorithm: str,
    environment_id: str,
    seed: int,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    terminal_configuration = (
        build_terminal_configuration(
            configuration
        )
    )

    custom_configuration = (
        build_custom_configuration(
            configuration,
            seed,
        )
    )

    started_at = time.time()

    record: dict[str, Any] = {
        "project_name":
            configuration[
                "project_name"
            ],
        "algorithm":
            algorithm,
        "environment_id":
            environment_id,
        "environment":
            ENVIRONMENT_IDS[
                environment_id
            ],
        "seed":
            int(seed),
        "total_steps":
            int(
                configuration[
                    "total_steps"
                ]
            ),
        "steps_per_epoch":
            int(
                configuration[
                    "steps_per_epoch"
                ]
            ),
        "status":
            "running",
        "mean_episode_return":
            np.nan,
        "mean_episode_cost":
            np.nan,
        "mean_episode_length":
            np.nan,
        "elapsed_seconds":
            np.nan,
        "error_type":
            "",
        "error_message":
            "",
    }

    try:
        agent = omnisafe.Agent(
            algorithm,
            environment_id,
            train_terminal_cfgs=
                terminal_configuration,
            custom_cfgs=
                custom_configuration,
        )

        learning_result = (
            agent.learn()
        )

        if (
            not isinstance(
                learning_result,
                tuple,
            )
            or len(learning_result)
            != 3
        ):
            raise RuntimeError(
                "OmniSafe Agent.learn() did not return "
                "the expected three metrics."
            )

        (
            mean_episode_return,
            mean_episode_cost,
            mean_episode_length,
        ) = learning_result

        record.update(
            {
                "status":
                    "completed",
                "mean_episode_return":
                    float(
                        mean_episode_return
                    ),
                "mean_episode_cost":
                    float(
                        mean_episode_cost
                    ),
                "mean_episode_length":
                    float(
                        mean_episode_length
                    ),
            }
        )

    except Exception as error:
        record.update(
            {
                "status":
                    "failed",
                "error_type":
                    type(error).__name__,
                "error_message":
                    str(error),
            }
        )

        print(
            "\n"
            + "=" * 90
        )

        print(
            "RUN FAILED"
        )

        print(
            "=" * 90
        )

        print(
            "Algorithm:",
            algorithm,
        )

        print(
            "Environment:",
            environment_id,
        )

        print(
            "Seed:",
            seed,
        )

        traceback.print_exc()

    finally:
        record[
            "elapsed_seconds"
        ] = float(
            time.time()
            - started_at
        )

    return record


def save_checkpoint(
    records: list[
        dict[str, Any]
    ],
    output_directory: Path,
) -> None:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = pd.DataFrame(
        records
    )

    frame.to_csv(
        output_directory
        / "omnisafe_run_summary.csv",
        index=False,
    )


def run_all(
    configuration: dict[str, Any],
) -> pd.DataFrame:
    output_directory = Path(
        configuration[
            "output_dir"
        ]
    )

    combinations = [
        (
            algorithm,
            environment_id,
            int(seed),
        )
        for environment_id
        in configuration[
            "environments"
        ]
        for algorithm
        in configuration[
            "algorithms"
        ]
        for seed
        in configuration[
            "random_seeds"
        ]
    ]

    records: list[
        dict[str, Any]
    ] = []

    total_runs = len(
        combinations
    )

    print(
        "=" * 90
    )

    print(
        "OMNISAFE CONSTRAINED-RL EXPERIMENTS"
    )

    print(
        "=" * 90
    )

    print(
        "Total runs:",
        total_runs,
    )

    print(
        "Algorithms:",
        configuration[
            "algorithms"
        ],
    )

    print(
        "Environments:",
        configuration[
            "environments"
        ],
    )

    print(
        "Seeds:",
        configuration[
            "random_seeds"
        ],
    )

    for run_number, (
        algorithm,
        environment_id,
        seed,
    ) in enumerate(
        combinations,
        start=1,
    ):
        print(
            "\n"
            + "=" * 90
        )

        print(
            f"RUN {run_number}/{total_runs}"
        )

        print(
            "=" * 90
        )

        print(
            "Algorithm:",
            algorithm,
        )

        print(
            "Environment:",
            environment_id,
        )

        print(
            "Seed:",
            seed,
        )

        record = (
            run_single_experiment(
                algorithm=
                    algorithm,
                environment_id=
                    environment_id,
                seed=
                    seed,
                configuration=
                    configuration,
            )
        )

        records.append(
            record
        )

        save_checkpoint(
            records,
            output_directory,
        )

        print(
            "Status:",
            record["status"],
        )

        print(
            "Elapsed seconds:",
            round(
                record[
                    "elapsed_seconds"
                ],
                2,
            ),
        )

    results = pd.DataFrame(
        records
    )

    manifest = {
        "configuration":
            configuration,
        "expected_runs":
            expected_run_count(
                configuration
            ),
        "completed_runs":
            int(
                (
                    results["status"]
                    == "completed"
                ).sum()
            ),
        "failed_runs":
            int(
                (
                    results["status"]
                    == "failed"
                ).sum()
            ),
        "python_version":
            platform.python_version(),
        "omnisafe_version":
            omnisafe.__version__,
        "torch_version":
            torch.__version__,
        "numpy_version":
            np.__version__,
        "pandas_version":
            pd.__version__,
    }

    (
        output_directory
        / "omnisafe_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the OmniSafe YAML configuration.",
    )

    arguments = (
        parser.parse_args()
    )

    configuration = (
        load_configuration(
            arguments.config
        )
    )

    results = run_all(
        configuration
    )

    completed = int(
        (
            results["status"]
            == "completed"
        ).sum()
    )

    failed = int(
        (
            results["status"]
            == "failed"
        ).sum()
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "OMNISAFE RUN SUMMARY"
    )

    print(
        "=" * 90
    )

    print(
        results[
            [
                "algorithm",
                "environment",
                "seed",
                "status",
                "mean_episode_return",
                "mean_episode_cost",
                "mean_episode_length",
                "elapsed_seconds",
                "error_type",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nCompleted:",
        completed,
    )

    print(
        "Failed:",
        failed,
    )

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()