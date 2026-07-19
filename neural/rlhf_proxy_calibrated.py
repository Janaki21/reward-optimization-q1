"""Calibrated controlled RLHF-style reward perturbation experiment.

Pairwise reward models are identifiable only up to scale and offset. This
analysis min-max normalizes each trained reward model over the complete
controlled response set before policy optimization.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from neural.rlhf_proxy_experiment import (
    METHODS,
    evaluate_policy,
    generate_contexts,
    response_features,
    reward_model_scores,
    train_policy,
    train_reward_model,
)


def normalize_reward_scores(
    scores: torch.Tensor,
) -> torch.Tensor:
    minimum = scores.min()
    maximum = scores.max()

    scale = maximum - minimum

    if float(scale.item()) <= 1e-12:
        return torch.zeros_like(
            scores
        )

    return (
        scores - minimum
    ) / scale


def load_configuration(
    path: str | Path,
) -> dict[str, Any]:
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        configuration = yaml.safe_load(
            file
        )

    required = {
        "project_name",
        "random_seeds",
        "perturbations",
        "methods",
        "reward_model_epochs",
        "reward_model_learning_rate",
        "policy_epochs",
        "policy_learning_rate",
        "fixed_penalty",
        "cost_limit",
        "lambda_learning_rate",
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

    unknown_methods = sorted(
        set(
            configuration[
                "methods"
            ]
        ).difference(
            METHODS
        )
    )

    if unknown_methods:
        raise ValueError(
            "Unknown methods: "
            + ", ".join(
                unknown_methods
            )
        )

    return configuration


def run_experiment(
    configuration: dict[str, Any],
) -> pd.DataFrame:
    contexts = generate_contexts()

    features = response_features(
        contexts
    )

    combinations = [
        (
            int(seed),
            float(perturbation),
        )
        for seed
        in configuration[
            "random_seeds"
        ]
        for perturbation
        in configuration[
            "perturbations"
        ]
    ]

    total_runs = (
        len(combinations)
        * len(
            configuration[
                "methods"
            ]
        )
    )

    completed = 0
    records = []

    for seed, perturbation in combinations:
        reward_model = train_reward_model(
            contexts=contexts,
            features=features,
            perturbation=
                perturbation,
            seed=seed,
            epochs=int(
                configuration[
                    "reward_model_epochs"
                ]
            ),
            learning_rate=float(
                configuration[
                    "reward_model_learning_rate"
                ]
            ),
        )

        raw_scores = reward_model_scores(
            reward_model,
            contexts,
            features,
        )

        normalized_scores = (
            normalize_reward_scores(
                raw_scores
            )
        )

        for method in configuration[
            "methods"
        ]:
            (
                policy,
                lagrange_multiplier,
            ) = train_policy(
                contexts=contexts,
                features=features,
                learned_rewards=
                    normalized_scores,
                method=method,
                seed=seed,
                epochs=int(
                    configuration[
                        "policy_epochs"
                    ]
                ),
                learning_rate=float(
                    configuration[
                        "policy_learning_rate"
                    ]
                ),
                fixed_penalty=float(
                    configuration[
                        "fixed_penalty"
                    ]
                ),
                cost_limit=float(
                    configuration[
                        "cost_limit"
                    ]
                ),
                lambda_learning_rate=float(
                    configuration[
                        "lambda_learning_rate"
                    ]
                ),
            )

            metrics = evaluate_policy(
                policy=policy,
                contexts=contexts,
                features=features,
                learned_rewards=
                    normalized_scores,
                method=method,
            )

            records.append(
                {
                    "seed":
                        seed,
                    "perturbation":
                        perturbation,
                    "method":
                        method,
                    "reward_normalization":
                        "global_minmax_0_1",
                    "raw_reward_minimum":
                        float(
                            raw_scores
                            .min()
                            .item()
                        ),
                    "raw_reward_maximum":
                        float(
                            raw_scores
                            .max()
                            .item()
                        ),
                    "final_lagrange_multiplier":
                        lagrange_multiplier,
                    **metrics,
                }
            )

            completed += 1

            if (
                completed % 20 == 0
                or completed
                == total_runs
            ):
                print(
                    f"Completed "
                    f"{completed}/"
                    f"{total_runs}"
                )

    return pd.DataFrame(
        records
    )


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

    results = run_experiment(
        configuration
    )

    result_path = (
        output_directory
        / "rlhf_calibrated_results.csv"
    )

    results.to_csv(
        result_path,
        index=False,
    )

    manifest = {
        "configuration":
            configuration,
        "rows":
            len(results),
        "reward_normalization":
            "global_minmax_0_1",
        "scope": (
            "Controlled RLHF-style "
            "response-selection analogue"
        ),
        "python":
            platform.python_version(),
        "pandas":
            pd.__version__,
        "torch":
            torch.__version__,
    }

    (
        output_directory
        / "rlhf_calibrated_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = (
        results.groupby(
            [
                "perturbation",
                "method",
            ]
        )
        .agg(
            n_seeds=(
                "seed",
                "nunique",
            ),
            mean_utility=(
                "mean_true_utility",
                "mean",
            ),
            sd_utility=(
                "mean_true_utility",
                "std",
            ),
            mean_violation=(
                "violation_rate",
                "mean",
            ),
            sd_violation=(
                "violation_rate",
                "std",
            ),
            mean_sycophancy=(
                "sycophancy_rate",
                "mean",
            ),
            mean_deception=(
                "deception_rate",
                "mean",
            ),
            mean_normalized_reward_gap=(
                "reward_true_gap",
                "mean",
            ),
        )
        .reset_index()
        .round(4)
    )

    print("=" * 120)
    print(
        "CALIBRATED CONTROLLED RLHF RESULTS"
    )
    print("=" * 120)

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nSaved:",
        result_path,
    )


if __name__ == "__main__":
    main()
    