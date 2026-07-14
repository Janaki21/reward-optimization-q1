"""Statistical analysis for seed-level experimental results."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


METRICS = (
    "proxy_return",
    "true_utility",
    "policy_violation_rate",
    "false_answer_rate",
    "monitoring_gap",
    "proxy_true_gap",
)


def bootstrap_confidence_interval(
    values,
    seed: int = 20260714,
    bootstrap_samples: int = 5000,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    if len(values) == 1:
        value = float(values[0])
        return value, value

    rng = np.random.default_rng(seed)

    bootstrap_means = rng.choice(
        values,
        size=(bootstrap_samples, len(values)),
        replace=True,
    ).mean(axis=1)

    lower, upper = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )

    return float(lower), float(upper)


def aggregate_statistics(
    run_summary: pd.DataFrame,
) -> pd.DataFrame:
    group_columns = [
        "environment",
        "agent",
        "corruption_probability",
        "corruption_magnitude",
        "constraint_error",
    ]

    rows = []

    for keys, group in run_summary.groupby(
        group_columns,
        dropna=False,
    ):
        row = dict(zip(group_columns, keys))

        row["n_seeds"] = int(
            group["seed"].nunique()
        )

        for metric in METRICS:
            values = (
                group[metric]
                .dropna()
                .to_numpy(dtype=float)
            )

            if len(values) == 0:
                mean = np.nan
                standard_deviation = np.nan
            else:
                mean = float(np.mean(values))

                standard_deviation = (
                    float(np.std(values, ddof=1))
                    if len(values) > 1
                    else 0.0
                )

            lower, upper = (
                bootstrap_confidence_interval(values)
            )

            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = (
                standard_deviation
            )
            row[f"{metric}_ci_low"] = lower
            row[f"{metric}_ci_high"] = upper

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(group_columns)
        .reset_index(drop=True)
    )


def rank_biserial_correlation(
    first_values,
    second_values,
) -> float:
    differences = (
        np.asarray(first_values)
        - np.asarray(second_values)
    )

    differences = differences[
        differences != 0
    ]

    if len(differences) == 0:
        return 0.0

    ranks = (
        pd.Series(np.abs(differences))
        .rank(method="average")
        .to_numpy()
    )

    positive_ranks = ranks[
        differences > 0
    ].sum()

    negative_ranks = ranks[
        differences < 0
    ].sum()

    return float(
        (positive_ranks - negative_ranks)
        / ranks.sum()
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
    )

    valid_indices = np.flatnonzero(
        np.isfinite(p_values)
    )

    sorted_indices = valid_indices[
        np.argsort(p_values[valid_indices])
    ]

    running_maximum = 0.0
    number_of_tests = len(sorted_indices)

    for rank, index in enumerate(sorted_indices):
        multiplier = number_of_tests - rank

        corrected_value = (
            multiplier * p_values[index]
        )

        running_maximum = max(
            running_maximum,
            corrected_value,
        )

        adjusted[index] = min(
            1.0,
            running_maximum,
        )

    return adjusted


def paired_statistical_tests(
    run_summary: pd.DataFrame,
) -> pd.DataFrame:
    condition_columns = [
        "environment",
        "corruption_probability",
        "corruption_magnitude",
        "constraint_error",
    ]

    primary_metrics = {
        "adversarial":
            "policy_violation_rate",
        "truth_proxy":
            "false_answer_rate",
        "monitoring_shift":
            "monitoring_gap",
    }

    rows = []

    for keys, group in run_summary.groupby(
        condition_columns
    ):
        environment = keys[0]
        metric = primary_metrics[environment]

        available_agents = sorted(
            group["agent"].unique()
        )

        for first_agent, second_agent in combinations(
            available_agents,
            2,
        ):
            comparison_data = group[
                group["agent"].isin(
                    [first_agent, second_agent]
                )
            ]

            paired_data = comparison_data.pivot(
                index="seed",
                columns="agent",
                values=metric,
            ).dropna()

            if len(paired_data) < 5:
                continue

            first_values = paired_data[
                first_agent
            ].to_numpy(dtype=float)

            second_values = paired_data[
                second_agent
            ].to_numpy(dtype=float)

            if np.allclose(
                first_values,
                second_values,
            ):
                statistic = 0.0
                p_value = 1.0
            else:
                result = wilcoxon(
                    first_values,
                    second_values,
                    alternative="two-sided",
                    method="auto",
                )

                statistic = float(result.statistic)
                p_value = float(result.pvalue)

            differences = (
                first_values - second_values
            )

            lower, upper = (
                bootstrap_confidence_interval(
                    differences
                )
            )

            rows.append(
                {
                    **dict(
                        zip(condition_columns, keys)
                    ),
                    "metric": metric,
                    "agent_a": first_agent,
                    "agent_b": second_agent,
                    "n_pairs": len(paired_data),
                    "mean_difference":
                        float(differences.mean()),
                    "difference_ci_low": lower,
                    "difference_ci_high": upper,
                    "wilcoxon_statistic":
                        statistic,
                    "p_value": p_value,
                    "rank_biserial":
                        rank_biserial_correlation(
                            first_values,
                            second_values,
                        ),
                }
            )

    results = pd.DataFrame(rows)

    if len(results) > 0:
        results["p_holm"] = holm_adjustment(
            results["p_value"].to_numpy()
        )

    return results


def calculate_perturbation_auc(
    run_summary: pd.DataFrame,
) -> pd.DataFrame:
    primary_metrics = {
        "adversarial":
            "policy_violation_rate",
        "truth_proxy":
            "false_answer_rate",
        "monitoring_shift":
            "monitoring_gap",
    }

    group_columns = [
        "environment",
        "agent",
        "seed",
        "corruption_magnitude",
        "constraint_error",
    ]

    rows = []

    for keys, group in run_summary.groupby(
        group_columns
    ):
        environment = keys[0]
        metric = primary_metrics[environment]

        group = group.sort_values(
            "corruption_probability"
        )

        failure_auc = np.trapezoid(
            group[metric],
            group["corruption_probability"],
        )

        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "metric": metric,
                "failure_auc": float(failure_auc),
            }
        )

    return pd.DataFrame(rows)


def run_full_analysis(
    run_summary: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return {
        "aggregated_statistics":
            aggregate_statistics(run_summary),
        "paired_tests":
            paired_statistical_tests(run_summary),
        "perturbation_auc":
            calculate_perturbation_auc(
                run_summary
            ),
    }