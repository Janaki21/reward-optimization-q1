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


PRIMARY_METRICS = {
    "adversarial": "policy_violation_rate",
    "truth_proxy": "false_answer_rate",
    "monitoring_shift": "monitoring_gap",
}


def trapezoidal_auc(
    x_values,
    y_values,
) -> float:
    """
    Calculate trapezoidal area without np.trapz or
    np.trapezoid, ensuring compatibility with NumPy
    1.x and NumPy 2.x.
    """

    x_values = np.asarray(
        x_values,
        dtype=float,
    )

    y_values = np.asarray(
        y_values,
        dtype=float,
    )

    finite_mask = (
        np.isfinite(x_values)
        & np.isfinite(y_values)
    )

    x_values = x_values[finite_mask]
    y_values = y_values[finite_mask]

    if len(x_values) < 2:
        return np.nan

    order = np.argsort(x_values)

    x_values = x_values[order]
    y_values = y_values[order]

    interval_widths = np.diff(
        x_values
    )

    interval_heights = (
        y_values[:-1]
        + y_values[1:]
    ) / 2.0

    area = np.sum(
        interval_widths
        * interval_heights
    )

    return float(area)


def bootstrap_confidence_interval(
    values,
    seed: int = 20260714,
    bootstrap_samples: int = 5000,
) -> tuple[float, float]:
    """
    Calculate a percentile-bootstrap 95% confidence
    interval for a sample mean.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan

    if len(values) == 1:
        value = float(values[0])
        return value, value

    random_generator = (
        np.random.default_rng(seed)
    )

    bootstrap_means = (
        random_generator.choice(
            values,
            size=(
                bootstrap_samples,
                len(values),
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


def aggregate_statistics(
    run_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate seed-level results using means,
    standard deviations and bootstrap confidence
    intervals.
    """

    group_columns = [
        "environment",
        "agent",
        "corruption_probability",
        "corruption_magnitude",
        "constraint_error",
    ]

    rows = []

    grouped_results = run_summary.groupby(
        group_columns,
        dropna=False,
    )

    for keys, group in grouped_results:
        row = dict(
            zip(group_columns, keys)
        )

        row["n_seeds"] = int(
            group["seed"].nunique()
        )

        for metric in METRICS:
            if metric not in group.columns:
                values = np.array(
                    [],
                    dtype=float,
                )
            else:
                values = (
                    group[metric]
                    .dropna()
                    .to_numpy(dtype=float)
                )

            if len(values) == 0:
                metric_mean = np.nan
                metric_standard_deviation = np.nan

            elif len(values) == 1:
                metric_mean = float(
                    values[0]
                )

                metric_standard_deviation = 0.0

            else:
                metric_mean = float(
                    np.mean(values)
                )

                metric_standard_deviation = float(
                    np.std(
                        values,
                        ddof=1,
                    )
                )

            lower_bound, upper_bound = (
                bootstrap_confidence_interval(
                    values
                )
            )

            row[
                f"{metric}_mean"
            ] = metric_mean

            row[
                f"{metric}_sd"
            ] = metric_standard_deviation

            row[
                f"{metric}_ci_low"
            ] = lower_bound

            row[
                f"{metric}_ci_high"
            ] = upper_bound

        rows.append(row)

    aggregated = pd.DataFrame(
        rows
    )

    if len(aggregated) == 0:
        return aggregated

    return (
        aggregated
        .sort_values(group_columns)
        .reset_index(drop=True)
    )


def rank_biserial_correlation(
    first_values,
    second_values,
) -> float:
    """
    Calculate paired rank-biserial correlation.
    """

    first_values = np.asarray(
        first_values,
        dtype=float,
    )

    second_values = np.asarray(
        second_values,
        dtype=float,
    )

    differences = (
        first_values - second_values
    )

    differences = differences[
        differences != 0
    ]

    if len(differences) == 0:
        return 0.0

    absolute_differences = np.abs(
        differences
    )

    ranks = (
        pd.Series(
            absolute_differences
        )
        .rank(method="average")
        .to_numpy(dtype=float)
    )

    positive_rank_sum = float(
        ranks[
            differences > 0
        ].sum()
    )

    negative_rank_sum = float(
        ranks[
            differences < 0
        ].sum()
    )

    total_rank_sum = (
        positive_rank_sum
        + negative_rank_sum
    )

    if total_rank_sum == 0:
        return 0.0

    return float(
        (
            positive_rank_sum
            - negative_rank_sum
        )
        / total_rank_sum
    )


def holm_adjustment(
    p_values,
) -> np.ndarray:
    """
    Apply Holm's step-down multiple-comparison
    correction.
    """

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    adjusted_values = np.full(
        len(p_values),
        np.nan,
        dtype=float,
    )

    valid_indices = np.flatnonzero(
        np.isfinite(p_values)
    )

    if len(valid_indices) == 0:
        return adjusted_values

    sorted_indices = valid_indices[
        np.argsort(
            p_values[
                valid_indices
            ]
        )
    ]

    number_of_tests = len(
        sorted_indices
    )

    running_maximum = 0.0

    for rank, original_index in enumerate(
        sorted_indices
    ):
        remaining_tests = (
            number_of_tests - rank
        )

        corrected_value = (
            remaining_tests
            * p_values[
                original_index
            ]
        )

        running_maximum = max(
            running_maximum,
            corrected_value,
        )

        adjusted_values[
            original_index
        ] = min(
            1.0,
            running_maximum,
        )

    return adjusted_values


def paired_statistical_tests(
    run_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare algorithms using paired Wilcoxon tests,
    bootstrap difference intervals and rank-biserial
    effect sizes.
    """

    condition_columns = [
        "environment",
        "corruption_probability",
        "corruption_magnitude",
        "constraint_error",
    ]

    output_columns = [
        *condition_columns,
        "metric",
        "agent_a",
        "agent_b",
        "n_pairs",
        "mean_difference",
        "difference_ci_low",
        "difference_ci_high",
        "wilcoxon_statistic",
        "p_value",
        "rank_biserial",
        "p_holm",
    ]

    rows = []

    grouped_conditions = (
        run_summary.groupby(
            condition_columns,
            dropna=False,
        )
    )

    for keys, group in grouped_conditions:
        environment = keys[0]

        metric = PRIMARY_METRICS[
            environment
        ]

        available_agents = sorted(
            group["agent"].unique()
        )

        for first_agent, second_agent in combinations(
            available_agents,
            2,
        ):
            comparison_data = group[
                group["agent"].isin(
                    [
                        first_agent,
                        second_agent,
                    ]
                )
            ]

            paired_data = (
                comparison_data.pivot(
                    index="seed",
                    columns="agent",
                    values=metric,
                )
                .dropna()
            )

            if len(paired_data) < 5:
                continue

            first_values = paired_data[
                first_agent
            ].to_numpy(dtype=float)

            second_values = paired_data[
                second_agent
            ].to_numpy(dtype=float)

            differences = (
                first_values
                - second_values
            )

            if np.allclose(
                first_values,
                second_values,
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

            (
                difference_lower,
                difference_upper,
            ) = bootstrap_confidence_interval(
                differences
            )

            result_row = dict(
                zip(
                    condition_columns,
                    keys,
                )
            )

            result_row.update(
                {
                    "metric":
                        metric,
                    "agent_a":
                        first_agent,
                    "agent_b":
                        second_agent,
                    "n_pairs":
                        int(
                            len(
                                paired_data
                            )
                        ),
                    "mean_difference":
                        float(
                            np.mean(
                                differences
                            )
                        ),
                    "difference_ci_low":
                        difference_lower,
                    "difference_ci_high":
                        difference_upper,
                    "wilcoxon_statistic":
                        statistic,
                    "p_value":
                        p_value,
                    "rank_biserial":
                        rank_biserial_correlation(
                            first_values,
                            second_values,
                        ),
                }
            )

            rows.append(
                result_row
            )

    if len(rows) == 0:
        return pd.DataFrame(
            columns=output_columns
        )

    results = pd.DataFrame(
        rows
    )

    results["p_holm"] = (
        holm_adjustment(
            results[
                "p_value"
            ].to_numpy(dtype=float)
        )
    )

    return results[
        output_columns
    ]


def calculate_perturbation_auc(
    run_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate seed-level area under the failure-rate
    versus reward-corruption-probability curve.
    """

    group_columns = [
        "environment",
        "agent",
        "seed",
        "corruption_magnitude",
        "constraint_error",
    ]

    output_columns = [
        *group_columns,
        "metric",
        "failure_auc",
    ]

    rows = []

    grouped_results = run_summary.groupby(
        group_columns,
        dropna=False,
    )

    for keys, group in grouped_results:
        environment = keys[0]

        metric = PRIMARY_METRICS[
            environment
        ]

        ordered_group = (
            group.sort_values(
                "corruption_probability"
            )
        )

        corruption_probabilities = (
            ordered_group[
                "corruption_probability"
            ]
            .to_numpy(dtype=float)
        )

        failure_values = (
            ordered_group[metric]
            .to_numpy(dtype=float)
        )

        failure_auc = (
            trapezoidal_auc(
                corruption_probabilities,
                failure_values,
            )
        )

        result_row = dict(
            zip(
                group_columns,
                keys,
            )
        )

        result_row.update(
            {
                "metric":
                    metric,
                "failure_auc":
                    failure_auc,
            }
        )

        rows.append(
            result_row
        )

    if len(rows) == 0:
        return pd.DataFrame(
            columns=output_columns
        )

    return pd.DataFrame(
        rows,
        columns=output_columns,
    )


def build_auc_summary(
    perturbation_auc: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate seed-level failure AUC results.
    """

    group_columns = [
        "environment",
        "agent",
        "corruption_magnitude",
        "constraint_error",
    ]

    output_columns = [
        *group_columns,
        "n_seeds",
        "failure_auc_mean",
        "failure_auc_sd",
        "failure_auc_ci_low",
        "failure_auc_ci_high",
    ]

    if len(perturbation_auc) == 0:
        return pd.DataFrame(
            columns=output_columns
        )

    rows = []

    grouped_auc = (
        perturbation_auc.groupby(
            group_columns,
            dropna=False,
        )
    )

    for keys, group in grouped_auc:
        values = (
            group["failure_auc"]
            .dropna()
            .to_numpy(dtype=float)
        )

        if len(values) == 0:
            mean_value = np.nan
            standard_deviation = np.nan

        elif len(values) == 1:
            mean_value = float(
                values[0]
            )

            standard_deviation = 0.0

        else:
            mean_value = float(
                np.mean(values)
            )

            standard_deviation = float(
                np.std(
                    values,
                    ddof=1,
                )
            )

        lower_bound, upper_bound = (
            bootstrap_confidence_interval(
                values
            )
        )

        result_row = dict(
            zip(
                group_columns,
                keys,
            )
        )

        result_row.update(
            {
                "n_seeds":
                    int(
                        group[
                            "seed"
                        ].nunique()
                    ),
                "failure_auc_mean":
                    mean_value,
                "failure_auc_sd":
                    standard_deviation,
                "failure_auc_ci_low":
                    lower_bound,
                "failure_auc_ci_high":
                    upper_bound,
            }
        )

        rows.append(
            result_row
        )

    return pd.DataFrame(
        rows,
        columns=output_columns,
    )


def run_full_analysis(
    run_summary: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Execute the complete statistical analysis pipeline.
    """

    aggregated_statistics = (
        aggregate_statistics(
            run_summary
        )
    )

    paired_tests = (
        paired_statistical_tests(
            run_summary
        )
    )

    perturbation_auc = (
        calculate_perturbation_auc(
            run_summary
        )
    )

    auc_summary = (
        build_auc_summary(
            perturbation_auc
        )
    )

    return {
        "aggregated_statistics":
            aggregated_statistics,
        "paired_tests":
            paired_tests,
        "perturbation_auc":
            perturbation_auc,
        "auc_summary":
            auc_summary,
    }