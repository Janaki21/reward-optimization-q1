"""Lagrangian learning-rate sensitivity analysis for PPOLag and RCPO."""

from __future__ import annotations

import argparse
import json
import platform
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

import neural.omnisafe_env
import omnisafe
from neural.omnisafe_env import (
    ENVIRONMENT_IDS,
)
from neural.omnisafe_pilot import (
    build_custom_configuration,
    build_terminal_configuration,
    episode_cost_limit,
    evaluate_actor,
)


SUPPORTED_ALGORITHMS = (
    "PPOLag",
    "RCPO",
)


def load_configuration(
    path: str | Path,
) -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing configuration: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        configuration = yaml.safe_load(file)

    required = {
        "project_name",
        "algorithms",
        "environments",
        "random_seeds",
        "lambda_learning_rates",
        "total_steps",
        "steps_per_epoch",
        "update_iters",
        "batch_size",
        "cost_rate_limit",
        "episode_length",
        "evaluation_episodes",
        "device",
        "torch_threads",
        "output_dir",
    }

    missing = sorted(
        required.difference(
            configuration
        )
    )

    if missing:
        raise ValueError(
            "Missing fields: "
            + ", ".join(missing)
        )

    unknown_algorithms = sorted(
        set(
            configuration[
                "algorithms"
            ]
        ).difference(
            SUPPORTED_ALGORITHMS
        )
    )

    if unknown_algorithms:
        raise ValueError(
            "This sensitivity study supports only: "
            + ", ".join(
                SUPPORTED_ALGORITHMS
            )
        )

    unknown_environments = sorted(
        set(
            configuration[
                "environments"
            ]
        ).difference(
            ENVIRONMENT_IDS
        )
    )

    if unknown_environments:
        raise ValueError(
            "Unknown environments: "
            + ", ".join(
                unknown_environments
            )
        )

    for learning_rate in configuration[
        "lambda_learning_rates"
    ]:
        if float(learning_rate) <= 0.0:
            raise ValueError(
                "Every lambda learning rate must be positive."
            )

    if (
        int(
            configuration[
                "total_steps"
            ]
        )
        % int(
            configuration[
                "steps_per_epoch"
            ]
        )
        != 0
    ):
        raise ValueError(
            "total_steps must be divisible by steps_per_epoch."
        )

    return configuration


def build_sensitivity_configuration(
    algorithm: str,
    seed: int,
    lambda_learning_rate: float,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    custom = (
        build_custom_configuration(
            algorithm=algorithm,
            seed=seed,
            configuration=
                configuration,
        )
    )

    custom[
        "lagrange_cfgs"
    ][
        "lambda_lr"
    ] = float(
        lambda_learning_rate
    )

    return custom


def final_lagrange_multiplier(
    agent: omnisafe.Agent,
) -> float:
    algorithm_object = (
        agent.agent
    )

    candidate_names = [
        "_lagrange",
        "_lagrange_multiplier",
        "lagrange",
        "lagrange_multiplier",
    ]

    for name in candidate_names:
        candidate = getattr(
            algorithm_object,
            name,
            None,
        )

        if candidate is None:
            continue

        for value_name in [
            "lagrangian_multiplier",
            "multiplier",
            "value",
        ]:
            value = getattr(
                candidate,
                value_name,
                None,
            )

            if value is None:
                continue

            if isinstance(
                value,
                torch.Tensor,
            ):
                return float(
                    value.detach()
                    .cpu()
                    .reshape(-1)[0]
                    .item()
                )

            try:
                return float(value)
            except (
                TypeError,
                ValueError,
            ):
                continue

    return float("nan")


def run_single(
    algorithm: str,
    environment_id: str,
    seed: int,
    lambda_learning_rate: float,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    started_at = time.time()

    record: dict[str, Any] = {
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
        "lambda_learning_rate":
            float(
                lambda_learning_rate
            ),
        "episode_cost_limit":
            episode_cost_limit(
                configuration
            ),
        "total_steps":
            int(
                configuration[
                    "total_steps"
                ]
            ),
        "status":
            "running",
        "training_episode_return":
            float("nan"),
        "training_episode_cost":
            float("nan"),
        "training_episode_length":
            float("nan"),
        "final_lagrange_multiplier":
            float("nan"),
        "elapsed_seconds":
            float("nan"),
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
                build_terminal_configuration(
                    configuration
                ),
            custom_cfgs=
                build_sensitivity_configuration(
                    algorithm=
                        algorithm,
                    seed=
                        seed,
                    lambda_learning_rate=
                        lambda_learning_rate,
                    configuration=
                        configuration,
                ),
        )

        (
            training_return,
            training_cost,
            training_length,
        ) = agent.learn()

        actor = (
            agent.agent
            ._actor_critic
            .actor
        )

        heldout = evaluate_actor(
            actor=actor,
            environment_id=
                environment_id,
            seed=(
                int(seed)
                + 2_000_000
            ),
            episodes=int(
                configuration[
                    "evaluation_episodes"
                ]
            ),
            episode_length=int(
                configuration[
                    "episode_length"
                ]
            ),
        )

        record.update(
            {
                "status":
                    "completed",
                "training_episode_return":
                    float(
                        training_return
                    ),
                "training_episode_cost":
                    float(
                        training_cost
                    ),
                "training_episode_length":
                    float(
                        training_length
                    ),
                "final_lagrange_multiplier":
                    final_lagrange_multiplier(
                        agent
                    ),
                **heldout,
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

        traceback.print_exc()

    record["elapsed_seconds"] = float(
        time.time()
        - started_at
    )

    return record


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    arguments = parser.parse_args()

    configuration = load_configuration(
        arguments.config
    )

    output_directory = Path(
        configuration[
            "output_dir"
        ]
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    combinations = [
        (
            algorithm,
            environment_id,
            int(seed),
            float(learning_rate),
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
        for learning_rate
        in configuration[
            "lambda_learning_rates"
        ]
    ]

    records: list[
        dict[str, Any]
    ] = []

    print("=" * 100)
    print("LAGRANGIAN SENSITIVITY STUDY")
    print("=" * 100)
    print("Runs:", len(combinations))
    print(
        "Lambda learning rates:",
        configuration[
            "lambda_learning_rates"
        ],
    )
    print(
        "Episode cost limit:",
        episode_cost_limit(
            configuration
        ),
    )

    for run_number, (
        algorithm,
        environment_id,
        seed,
        learning_rate,
    ) in enumerate(
        combinations,
        start=1,
    ):
        print(
            "\n"
            + "=" * 100
        )
        print(
            f"RUN {run_number}/{len(combinations)}"
        )
        print("Algorithm:", algorithm)
        print(
            "Environment:",
            environment_id,
        )
        print("Seed:", seed)
        print(
            "Lambda learning rate:",
            learning_rate,
        )

        record = run_single(
            algorithm=algorithm,
            environment_id=
                environment_id,
            seed=seed,
            lambda_learning_rate=
                learning_rate,
            configuration=
                configuration,
        )

        records.append(record)

        pd.DataFrame(
            records
        ).to_csv(
            output_directory
            / "lagrange_sensitivity_summary.csv",
            index=False,
        )

        print(
            "Status:",
            record["status"],
        )

        if (
            record["status"]
            == "completed"
        ):
            print(
                "Held-out violation:",
                round(
                    record[
                        "heldout_violation_rate"
                    ],
                    4,
                ),
            )
            print(
                "Held-out utility:",
                round(
                    record[
                        "heldout_true_utility"
                    ],
                    4,
                ),
            )

    results = pd.DataFrame(
        records
    )

    manifest = {
        "configuration":
            configuration,
        "python":
            platform.python_version(),
        "omnisafe":
            omnisafe.__version__,
        "torch":
            torch.__version__,
        "completed":
            int(
                (
                    results["status"]
                    == "completed"
                ).sum()
            ),
        "failed":
            int(
                (
                    results["status"]
                    == "failed"
                ).sum()
            ),
    }

    (
        output_directory
        / "lagrange_sensitivity_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 100
    )
    print("SENSITIVITY SUMMARY")
    print("=" * 100)

    print(
        results[
            [
                "algorithm",
                "environment",
                "seed",
                "lambda_learning_rate",
                "status",
                "heldout_true_utility",
                "heldout_violation_rate",
                "heldout_proxy_true_gap",
                "final_lagrange_multiplier",
                "elapsed_seconds",
                "error_type",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    if (
        results["status"]
        == "failed"
    ).any():
        raise SystemExit(1)


if __name__ == "__main__":
    main()