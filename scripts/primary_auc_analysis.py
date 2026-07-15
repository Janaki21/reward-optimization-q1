"""Prespecified seed-level AUC comparisons for the primary hypotheses.

This analysis treats the area under the failure-versus-corruption curve
as the primary seed-level outcome. It avoids relying on a large collection
of underpowered pointwise comparisons.

Interpretation:
    Positive mean_difference:
        agent_a has a larger failure AUC and is therefore worse.

    Negative mean_difference:
        agent_a has a smaller failure AUC and is therefore better.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PRESPECIFIED_COMPARISONS = (
    (
        "reward_only",
        "hard_shield",
    ),
    (
        "reward_only",
        "lagrangian",
    ),
    (
        "fixed_penalty",
        "hard_shield",
    ),
    (
        "lagrangian",
        "hard_shield",
    ),
)


def bootstrap_mean_difference_interval(
    differences,
    seed: int,
    bootstrap_samples: int = 10_000,
) -> tuple[float, float]:
    differences = np.asarray(
        differences,
        dtype=float,
    )

    differences = differences[
        np.isfinite(differences)
    ]

    if len(differences) == 0:
        return np.nan, np.nan

    if len(differences) == 1:
        value = float(
            differences[0]
        )

        return value, value

    random_generator = (
        np.random.default_rng(seed)
    )

    bootstrap_means = (
        random_generator.choice(
            differences,
            size=(
                bootstrap_samples,
                len(differences),
            ),
            replace=True,
        )
        .mean(axis=1)
    )

    lower_bound, upper_bound = (
        np.quantile(
            bootstrap_means,
            [0.025, 0.975],
        )
    )

    return (
        float(lower_bound),
        float(upper_bound),
    )


def paired_rank_biserial(
    first_values,
    second_values,
) -> float:
    first_values = np.asarray(
        first_values,
        dtype=float,
    )

    second_values = np.asarray(
        second_values,
        dtype=float,
    )

    differences = (
        first_values
        - second_values
    )

    differences = differences[
        differences != 0
    ]

    if len(differences) == 0:
        return 0.0

    ranks = (
        pd.Series(
            np.abs(differences)
        )
        .rank(method="average")
        .to_numpy(dtype=float)
    )

    positive_sum = float(
        ranks[
            differences > 0
        ].sum()
    )

    negative_sum = float(
        ranks[
            differences < 0
        ].sum()
    )

    total = (
        positive_sum
        + negative_sum
    )

    if total == 0:
        return 0.0

    return float(
        (
            positive_sum
            - negative_sum
        )
        / total
    )


def holm_adjustment(
    p_values,
) -> np.ndarray:
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    adjusted = np.full(
        len(p_values),
        np.nan,
        dtype=float,
    )

    valid_indices = np.flatnonzero(
        np.isfinite(p_values)
    )

    if len(valid_indices) == 0:
        return adjusted

    ordered_indices = valid_indices[
        np.argsort(
            p_values[
                valid_indices
            ]
        )
    ]

    number_of_tests = len(
        ordered_indices
    )

    running_maximum = 0.0

    for rank, original_index in enumerate(
        ordered_indices
    ):
        remaining_tests = (
            number_of_tests
            - rank
        )

        candidate = (
            remaining_tests
            * p_values[
                original_index
            ]
        )

        running_maximum = max(
            running_maximum,
            candidate,
        )

        adjusted[
            original_index
        ] = min(
            1.0,
            running_maximum,
        )

    return adjusted


def validate_auc_data(
    auc_data: pd.DataFrame,
) -> None:
    required_columns = {
        "environment",
        "agent",
        "seed",
        "corruption_magnitude",
        "constraint_error",
        "metric",
        "failure_auc",
    }

    missing_columns = (
        required_columns
        - set(auc_data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required AUC columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    if auc_data.empty:
        raise ValueError(
            "The perturbation AUC table is empty."
        )


def calculate_primary_auc_tests(
    auc_data: pd.DataFrame,
) -> pd.DataFrame:
    validate_auc_data(
        auc_data
    )

    condition_columns = [
        "environment",
        "corruption_magnitude",
        "constraint_error",
        "metric",
    ]

    output_rows = []

    grouped_conditions = (
        auc_data.groupby(
            condition_columns,
            dropna=False,
        )
    )

    test_counter = 0

    for keys, group in grouped_conditions:
        condition = dict(
            zip(
                condition_columns,
                keys,
            )
        )

        for (
            first_agent,
            second_agent,
        ) in PRESPECIFIED_COMPARISONS:
            first_data = group[
                group["agent"]
                == first_agent
            ][
                [
                    "seed",
                    "failure_auc",
                ]
            ]

            second_data = group[
                group["agent"]
                == second_agent
            ][
                [
                    "seed",
                    "failure_auc",
                ]
            ]

            paired = first_data.merge(
                second_data,
                on="seed",
                how="inner",
                suffixes=(
                    "_a",
                    "_b",
                ),
            ).sort_values("seed")

            if len(paired) < 5:
                continue

            first_values = paired[
                "failure_auc_a"
            ].to_numpy(dtype=float)

            second_values = paired[
                "failure_auc_b"
            ].to_numpy(dtype=float)

            differences = (
                first_values
                - second_values
            )

            if np.allclose(
                differences,
                0.0,
            ):
                statistic = 0.0
                p_value = 1.0

            else:
                test_result = wilcoxon(
                    first_values,
                    second_values,
                    alternative="two-sided",
                    method="auto",
                )

                statistic = float(
                    test_result.statistic
                )

                p_value = float(
                    test_result.pvalue
                )

            test_counter += 1

            (
                difference_ci_low,
                difference_ci_high,
            ) = (
                bootstrap_mean_difference_interval(
                    differences,
                    seed=(
                        20260714
                        + test_counter
                    ),
                )
            )

            output_rows.append(
                {
                    **condition,
                    "agent_a":
                        first_agent,
                    "agent_b":
                        second_agent,
                    "n_pairs":
                        int(
                            len(paired)
                        ),
                    "mean_auc_a":
                        float(
                            np.mean(
                                first_values
                            )
                        ),
                    "mean_auc_b":
                        float(
                            np.mean(
                                second_values
                            )
                        ),
                    "mean_difference":
                        float(
                            np.mean(
                                differences
                            )
                        ),
                    "difference_ci_low":
                        difference_ci_low,
                    "difference_ci_high":
                        difference_ci_high,
                    "wilcoxon_statistic":
                        statistic,
                    "p_value":
                        p_value,
                    "rank_biserial":
                        paired_rank_biserial(
                            first_values,
                            second_values,
                        ),
                }
            )

    results = pd.DataFrame(
        output_rows
    )

    if results.empty:
        return results

    results[
        "p_holm_global"
    ] = holm_adjustment(
        results[
            "p_value"
        ].to_numpy(dtype=float)
    )

    results[
        "p_holm_within_environment"
    ] = np.nan

    for environment in results[
        "environment"
    ].unique():
        mask = (
            results["environment"]
            == environment
        )

        environment_adjusted = (
            holm_adjustment(
                results.loc[
                    mask,
                    "p_value",
                ].to_numpy(dtype=float)
            )
        )

        results.loc[
            mask,
            "p_holm_within_environment",
        ] = environment_adjusted

    results[
        "direction"
    ] = np.where(
        results[
            "mean_difference"
        ] > 0,
        (
            results["agent_a"]
            + " has higher failure AUC"
        ),
        np.where(
            results[
                "mean_difference"
            ] < 0,
            (
                results["agent_a"]
                + " has lower failure AUC"
            ),
            "no mean difference",
        ),
    )

    sort_columns = [
        "environment",
        "corruption_magnitude",
        "constraint_error",
        "agent_a",
        "agent_b",
    ]

    return (
        results
        .sort_values(sort_columns)
        .reset_index(drop=True)
    )


def build_primary_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    if results.empty:
        return results

    summary_columns = [
        "environment",
        "corruption_magnitude",
        "constraint_error",
        "metric",
        "agent_a",
        "agent_b",
        "n_pairs",
        "mean_auc_a",
        "mean_auc_b",
        "mean_difference",
        "difference_ci_low",
        "difference_ci_high",
        "p_value",
        "p_holm_global",
        "p_holm_within_environment",
        "rank_biserial",
        "direction",
    ]

    return results[
        summary_columns
    ].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run prespecified paired AUC comparisons."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to perturbation_auc.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directory for primary AUC outputs"
        ),
    )

    arguments = parser.parse_args()

    input_path = Path(
        arguments.input
    )

    output_directory = Path(
        arguments.output_dir
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"AUC input file not found: {input_path}"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    auc_data = pd.read_csv(
        input_path
    )

    results = (
        calculate_primary_auc_tests(
            auc_data
        )
    )

    summary = build_primary_summary(
        results
    )

    results.to_csv(
        output_directory
        / "primary_auc_tests.csv",
        index=False,
    )

    summary.to_excel(
        output_directory
        / "primary_auc_tests.xlsx",
        index=False,
    )

    print(
        f"Primary AUC comparisons: {len(results)}"
    )

    if results.empty:
        print(
            "No eligible comparisons were found."
        )

    else:
        display_columns = [
            "environment",
            "corruption_magnitude",
            "agent_a",
            "agent_b",
            "n_pairs",
            "mean_difference",
            "difference_ci_low",
            "difference_ci_high",
            "p_value",
            "p_holm_global",
            "rank_biserial",
        ]

        print(
            results[
                display_columns
            ]
            .round(6)
            .to_string(index=False)
        )

    print(
        "Saved:",
        output_directory
        / "primary_auc_tests.csv",
    )


if __name__ == "__main__":
    main()