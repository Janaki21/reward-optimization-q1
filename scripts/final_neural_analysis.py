"""Final statistical analysis for the 30-seed neural experiments."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import (
    rankdata,
    wilcoxon,
)


METHOD_ORDER = [
    "PPO",
    "CPO",
    "PPOLag",
    "RCPO",
    "Veto",
]

ENVIRONMENT_ORDER = [
    "adversarial",
    "truth_proxy",
    "monitoring_shift",
]

METHOD_COLOURS = {
    "PPO": "#0072B2",
    "CPO": "#E69F00",
    "PPOLag": "#009E73",
    "RCPO": "#CC79A7",
    "Veto": "#D55E00",
}

DISPLAY_ENVIRONMENTS = {
    "adversarial":
        "Adversarial incentives",
    "truth_proxy":
        "Truth–proxy divergence",
    "monitoring_shift":
        "Monitoring shift",
}


def bootstrap_mean_interval(
    values: Iterable[float],
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 20260719,
) -> tuple[float, float]:
    array = np.asarray(
        list(values),
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    if array.size == 0:
        return (
            float("nan"),
            float("nan"),
        )

    if array.size == 1:
        value = float(
            array[0]
        )

        return (
            value,
            value,
        )

    random_generator = (
        np.random.default_rng(
            seed
        )
    )

    indices = (
        random_generator.integers(
            low=0,
            high=array.size,
            size=(
                resamples,
                array.size,
            ),
        )
    )

    bootstrap_means = (
        array[indices].mean(
            axis=1
        )
    )

    alpha = (
        1.0 - confidence
    )

    lower = float(
        np.quantile(
            bootstrap_means,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            bootstrap_means,
            1.0 - alpha / 2.0,
        )
    )

    return (
        lower,
        upper,
    )


def holm_adjust(
    p_values: Iterable[float],
) -> np.ndarray:
    values = np.asarray(
        list(p_values),
        dtype=float,
    )

    adjusted = np.full(
        values.shape,
        np.nan,
        dtype=float,
    )

    finite_indices = np.flatnonzero(
        np.isfinite(values)
    )

    if finite_indices.size == 0:
        return adjusted

    finite_values = values[
        finite_indices
    ]

    order = np.argsort(
        finite_values
    )

    sorted_values = (
        finite_values[order]
    )

    number_of_tests = (
        len(sorted_values)
    )

    sorted_adjusted = np.empty(
        number_of_tests,
        dtype=float,
    )

    running_maximum = 0.0

    for index, p_value in enumerate(
        sorted_values
    ):
        candidate = (
            (
                number_of_tests
                - index
            )
            * p_value
        )

        running_maximum = max(
            running_maximum,
            candidate,
        )

        sorted_adjusted[index] = min(
            1.0,
            running_maximum,
        )

    inverse_order = np.empty_like(
        order
    )

    inverse_order[
        order
    ] = np.arange(
        number_of_tests
    )

    adjusted[
        finite_indices
    ] = sorted_adjusted[
        inverse_order
    ]

    return adjusted


def matched_rank_biserial(
    differences: Iterable[float],
) -> float:
    difference_array = np.asarray(
        list(differences),
        dtype=float,
    )

    difference_array = difference_array[
        np.isfinite(
            difference_array
        )
    ]

    nonzero = difference_array[
        difference_array != 0.0
    ]

    if nonzero.size == 0:
        return 0.0

    ranks = rankdata(
        np.abs(nonzero)
    )

    positive_rank_sum = float(
        ranks[
            nonzero > 0.0
        ].sum()
    )

    negative_rank_sum = float(
        ranks[
            nonzero < 0.0
        ].sum()
    )

    denominator = (
        positive_rank_sum
        + negative_rank_sum
    )

    if denominator == 0.0:
        return 0.0

    return float(
        (
            positive_rank_sum
            - negative_rank_sum
        )
        / denominator
    )


def paired_wilcoxon(
    method_values: Iterable[float],
    baseline_values: Iterable[float],
) -> dict[str, float]:
    method_array = np.asarray(
        list(method_values),
        dtype=float,
    )

    baseline_array = np.asarray(
        list(baseline_values),
        dtype=float,
    )

    finite = (
        np.isfinite(method_array)
        & np.isfinite(
            baseline_array
        )
    )

    method_array = (
        method_array[finite]
    )

    baseline_array = (
        baseline_array[finite]
    )

    differences = (
        method_array
        - baseline_array
    )

    if differences.size == 0:
        return {
            "n_pairs": 0,
            "mean_difference":
                float("nan"),
            "median_difference":
                float("nan"),
            "cohens_dz":
                float("nan"),
            "rank_biserial":
                float("nan"),
            "wilcoxon_statistic":
                float("nan"),
            "p_value":
                float("nan"),
        }

    mean_difference = float(
        differences.mean()
    )

    median_difference = float(
        np.median(
            differences
        )
    )

    difference_sd = float(
        differences.std(
            ddof=11
        )
    ) if differences.size > 1 else 0.0

    if difference_sd == 0.0:
        if mean_difference == 0.0:
            cohens_dz = 0.0
        else:
            cohens_dz = float(
                np.sign(
                    mean_difference
                )
                * np.inf
            )
    else:
        cohens_dz = float(
            mean_difference
            / difference_sd
        )

    if np.all(
        differences == 0.0
    ):
        statistic = 0.0
        p_value = 1.0
    else:
        test = wilcoxon(
            method_array,
            baseline_array,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )

        statistic = float(
            test.statistic
        )

        p_value = float(
            test.pvalue
        )

    return {
        "n_pairs":
            int(
                differences.size
            ),
        "mean_difference":
            mean_difference,
        "median_difference":
            median_difference,
        "cohens_dz":
            cohens_dz,
        "rank_biserial":
            matched_rank_biserial(
                differences
            ),
        "wilcoxon_statistic":
            statistic,
        "p_value":
            p_value,
    }


def manual_trapezoidal_auc(
    x_values: Iterable[float],
    y_values: Iterable[float],
) -> float:
    x_array = np.asarray(
        list(x_values),
        dtype=float,
    )

    y_array = np.asarray(
        list(y_values),
        dtype=float,
    )

    order = np.argsort(
        x_array
    )

    x_array = x_array[
        order
    ]

    y_array = y_array[
        order
    ]

    if x_array.size < 2:
        return 0.0

    widths = np.diff(
        x_array
    )

    heights = (
        y_array[:-1]
        + y_array[1:]
    ) / 2.0

    return float(
        np.sum(
            widths * heights
        )
    )


def load_primary_results(
    omnisafe_path: Path,
    lagrange_path: Path,
    veto_path: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    omnisafe = pd.read_csv(
        omnisafe_path
    )

    lagrange = pd.read_csv(
        lagrange_path
    )

    veto = pd.read_csv(
        veto_path
    )

    omnisafe_primary = (
        omnisafe[
            omnisafe["status"]
            == "completed"
        ][
            [
                "environment",
                "algorithm",
                "seed",
                "heldout_true_utility",
                "heldout_violation_rate",
                "heldout_proxy_true_gap",
            ]
        ]
        .rename(
            columns={
                "algorithm":
                    "method",
                "heldout_true_utility":
                    "true_utility",
                "heldout_violation_rate":
                    "violation_rate",
                "heldout_proxy_true_gap":
                    "proxy_true_gap",
            }
        )
    )

    lagrange_primary = (
        lagrange[
            lagrange["status"]
            == "completed"
        ][
            [
                "environment",
                "algorithm",
                "seed",
                "heldout_true_utility",
                "heldout_violation_rate",
                "heldout_proxy_true_gap",
            ]
        ]
        .rename(
            columns={
                "algorithm":
                    "method",
                "heldout_true_utility":
                    "true_utility",
                "heldout_violation_rate":
                    "violation_rate",
                "heldout_proxy_true_gap":
                    "proxy_true_gap",
            }
        )
    )

    perfect_veto = veto[
        (
            veto["method"]
            == "ppo_hard_veto"
        )
        & (
            veto["constraint_error"]
            == 0.0
        )
    ][
        [
            "environment",
            "seed",
            "true_utility",
            "policy_violation_rate",
            "proxy_true_gap",
        ]
    ].copy()

    perfect_veto[
        "method"
    ] = "Veto"

    perfect_veto = (
        perfect_veto.rename(
            columns={
                "policy_violation_rate":
                    "violation_rate",
            }
        )
    )

    primary = pd.concat(
        [
            omnisafe_primary,
            lagrange_primary,
            perfect_veto[
                [
                    "environment",
                    "method",
                    "seed",
                    "true_utility",
                    "violation_rate",
                    "proxy_true_gap",
                ]
            ],
        ],
        ignore_index=True,
    )

    primary[
        "method"
    ] = pd.Categorical(
        primary["method"],
        categories=
            METHOD_ORDER,
        ordered=True,
    )

    primary[
        "environment"
    ] = pd.Categorical(
        primary["environment"],
        categories=
            ENVIRONMENT_ORDER,
        ordered=True,
    )

    primary = primary.sort_values(
        [
            "environment",
            "method",
            "seed",
        ]
    ).reset_index(
        drop=True
    )

    veto_rows = veto[
        veto["method"]
        == "ppo_hard_veto"
    ].copy()

    return (
        primary,
        veto_rows,
    )


def summarize_primary(
    primary: pd.DataFrame,
) -> pd.DataFrame:
    records: list[
        dict[str, float | str | int]
    ] = []

    metrics = [
        "true_utility",
        "violation_rate",
        "proxy_true_gap",
    ]

    for (
        environment,
        method,
    ), group in primary.groupby(
        [
            "environment",
            "method",
        ],
        observed=True,
    ):
        for metric_index, metric in enumerate(
            metrics
        ):
            values = (
                group[metric]
                .to_numpy(
                    dtype=float
                )
            )

            lower, upper = (
                bootstrap_mean_interval(
                    values,
                    seed=(
                        20260719
                        + metric_index
                        + len(records)
                    ),
                )
            )

            records.append(
                {
                    "environment":
                        str(environment),
                    "method":
                        str(method),
                    "metric":
                        metric,
                    "n_seeds":
                        int(
                            group[
                                "seed"
                            ].nunique()
                        ),
                    "mean":
                        float(
                            values.mean()
                        ),
                    "sd":
                        float(
                            values.std(
                                ddof=1
                            )
                        ),
                    "median":
                        float(
                            np.median(
                                values
                            )
                        ),
                    "q1":
                        float(
                            np.quantile(
                                values,
                                0.25,
                            )
                        ),
                    "q3":
                        float(
                            np.quantile(
                                values,
                                0.75,
                            )
                        ),
                    "ci95_lower":
                        lower,
                    "ci95_upper":
                        upper,
                }
            )

    return pd.DataFrame(
        records
    )


def calculate_paired_tests(
    primary: pd.DataFrame,
) -> pd.DataFrame:
    records: list[
        dict[str, float | str | int]
    ] = []

    metrics = [
        "violation_rate",
        "true_utility",
    ]

    comparison_methods = [
        "CPO",
        "PPOLag",
        "RCPO",
        "Veto",
    ]

    for environment in ENVIRONMENT_ORDER:
        environment_data = primary[
            primary[
                "environment"
            ]
            == environment
        ]

        baseline = (
            environment_data[
                environment_data[
                    "method"
                ]
                == "PPO"
            ][
                [
                    "seed",
                    *metrics,
                ]
            ]
            .set_index(
                "seed"
            )
        )

        for method in comparison_methods:
            comparison = (
                environment_data[
                    environment_data[
                        "method"
                    ]
                    == method
                ][
                    [
                        "seed",
                        *metrics,
                    ]
                ]
                .set_index(
                    "seed"
                )
            )

            paired = (
                baseline.join(
                    comparison,
                    how="inner",
                    lsuffix="_ppo",
                    rsuffix="_method",
                )
            )

            for metric in metrics:
                result = paired_wilcoxon(
                    paired[
                        f"{metric}_method"
                    ],
                    paired[
                        f"{metric}_ppo"
                    ],
                )

                records.append(
                    {
                        "environment":
                            environment,
                        "comparison":
                            f"{method} vs PPO",
                        "method":
                            method,
                        "baseline":
                            "PPO",
                        "metric":
                            metric,
                        **result,
                    }
                )

    frame = pd.DataFrame(
        records
    )

    frame[
        "p_holm"
    ] = np.nan

    for metric in metrics:
        mask = (
            frame["metric"]
            == metric
        )

        frame.loc[
            mask,
            "p_holm",
        ] = holm_adjust(
            frame.loc[
                mask,
                "p_value",
            ]
        )

    frame[
        "significant_0_05"
    ] = (
        frame["p_holm"]
        < 0.05
    )

    return frame


def calculate_veto_auc(
    veto_rows: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    seed_records: list[
        dict[str, float | str | int]
    ] = []

    for (
        environment,
        seed,
    ), group in veto_rows.groupby(
        [
            "environment",
            "seed",
        ]
    ):
        group = group.sort_values(
            "constraint_error"
        )

        seed_records.append(
            {
                "environment":
                    environment,
                "seed":
                    int(seed),
                "violation_auc":
                    manual_trapezoidal_auc(
                        group[
                            "constraint_error"
                        ],
                        group[
                            "policy_violation_rate"
                        ],
                    ),
                "utility_loss_auc":
                    manual_trapezoidal_auc(
                        group[
                            "constraint_error"
                        ],
                        1.0
                        - group[
                            "true_utility"
                        ],
                    ),
                "false_intervention_auc":
                    manual_trapezoidal_auc(
                        group[
                            "constraint_error"
                        ],
                        group[
                            "false_intervention_rate"
                        ],
                    ),
            }
        )

    seed_frame = pd.DataFrame(
        seed_records
    )

    summary_records: list[
        dict[str, float | str | int]
    ] = []

    for environment, group in (
        seed_frame.groupby(
            "environment"
        )
    ):
        for metric in [
            "violation_auc",
            "utility_loss_auc",
            "false_intervention_auc",
        ]:
            values = group[
                metric
            ].to_numpy(
                dtype=float
            )

            lower, upper = (
                bootstrap_mean_interval(
                    values,
                    seed=(
                        20260719
                        + len(
                            summary_records
                        )
                    ),
                )
            )

            summary_records.append(
                {
                    "environment":
                        environment,
                    "metric":
                        metric,
                    "n_seeds":
                        int(
                            group[
                                "seed"
                            ].nunique()
                        ),
                    "mean":
                        float(
                            values.mean()
                        ),
                    "sd":
                        float(
                            values.std(
                                ddof=1
                            )
                        ),
                    "ci95_lower":
                        lower,
                    "ci95_upper":
                        upper,
                }
            )

    return (
        seed_frame,
        pd.DataFrame(
            summary_records
        ),
    )


def save_figure(
    figure: plt.Figure,
    output_directory: Path,
    name: str,
) -> None:
    figure.savefig(
        output_directory
        / f"{name}.pdf",
        bbox_inches="tight",
    )

    figure.savefig(
        output_directory
        / f"{name}.svg",
        bbox_inches="tight",
    )

    figure.savefig(
        output_directory
        / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def plot_primary_metric(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_directory: Path,
    name: str,
) -> None:
    metric_data = summary[
        summary["metric"]
        == metric
    ]

    figure, axis = plt.subplots(
        figsize=(
            10.5,
            5.8,
        )
    )

    x_positions = np.arange(
        len(
            ENVIRONMENT_ORDER
        ),
        dtype=float,
    )

    width = 0.15

    offsets = (
        np.arange(
            len(METHOD_ORDER)
        )
        - (
            len(METHOD_ORDER)
            - 1
        )
        / 2.0
    ) * width

    for method_index, method in enumerate(
        METHOD_ORDER
    ):
        method_rows = (
            metric_data[
                metric_data[
                    "method"
                ]
                == method
            ]
            .set_index(
                "environment"
            )
        )

        means = np.asarray(
            [
                method_rows.loc[
                    environment,
                    "mean",
                ]
                for environment
                in ENVIRONMENT_ORDER
            ],
            dtype=float,
        )

        lower = np.asarray(
            [
                method_rows.loc[
                    environment,
                    "ci95_lower",
                ]
                for environment
                in ENVIRONMENT_ORDER
            ],
            dtype=float,
        )

        upper = np.asarray(
            [
                method_rows.loc[
                    environment,
                    "ci95_upper",
                ]
                for environment
                in ENVIRONMENT_ORDER
            ],
            dtype=float,
        )

        errors = np.vstack(
            [
                means - lower,
                upper - means,
            ]
        )

        axis.bar(
            x_positions
            + offsets[
                method_index
            ],
            means,
            width=width,
            color=
                METHOD_COLOURS[
                    method
                ],
            label=method,
            yerr=errors,
            capsize=3,
            linewidth=0.7,
            edgecolor="black",
        )

    axis.set_ylabel(
        ylabel
    )

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        [
            DISPLAY_ENVIRONMENTS[
                environment
            ]
            for environment
            in ENVIRONMENT_ORDER
        ]
    )

    axis.set_ylim(
        bottom=0.0
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend(
        ncol=5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            1.14,
        ),
    )

    figure.tight_layout()

    save_figure(
        figure,
        output_directory,
        name,
    )


def plot_veto_sensitivity(
    veto_rows: pd.DataFrame,
    output_directory: Path,
) -> None:
    summary_records = []

    for (
        environment,
        constraint_error,
    ), group in veto_rows.groupby(
        [
            "environment",
            "constraint_error",
        ]
    ):
        values = group[
            "policy_violation_rate"
        ].to_numpy(
            dtype=float
        )

        lower, upper = (
            bootstrap_mean_interval(
                values,
                seed=(
                    20260719
                    + len(
                        summary_records
                    )
                ),
            )
        )

        summary_records.append(
            {
                "environment":
                    environment,
                "constraint_error":
                    float(
                        constraint_error
                    ),
                "mean":
                    float(
                        values.mean()
                    ),
                "lower":
                    lower,
                "upper":
                    upper,
            }
        )

    summary = pd.DataFrame(
        summary_records
    )

    figure, axis = plt.subplots(
        figsize=(
            8.2,
            5.5,
        )
    )

    environment_colours = {
        "adversarial":
            "#0072B2",
        "truth_proxy":
            "#D55E00",
        "monitoring_shift":
            "#009E73",
    }

    for environment in ENVIRONMENT_ORDER:
        rows = (
            summary[
                summary[
                    "environment"
                ]
                == environment
            ]
            .sort_values(
                "constraint_error"
            )
        )

        x = rows[
            "constraint_error"
        ].to_numpy(
            dtype=float
        )

        mean = rows[
            "mean"
        ].to_numpy(
            dtype=float
        )

        lower = rows[
            "lower"
        ].to_numpy(
            dtype=float
        )

        upper = rows[
            "upper"
        ].to_numpy(
            dtype=float
        )

        colour = (
            environment_colours[
                environment
            ]
        )

        axis.plot(
            x,
            mean,
            marker="o",
            linewidth=2.2,
            color=colour,
            label=
                DISPLAY_ENVIRONMENTS[
                    environment
                ],
        )

        axis.fill_between(
            x,
            lower,
            upper,
            color=colour,
            alpha=0.18,
        )

    axis.set_xlabel(
        "Constraint-model error probability"
    )

    axis.set_ylabel(
        "Policy violation rate"
    )

    axis.set_xlim(
        -0.005,
        0.205,
    )

    axis.set_ylim(
        bottom=0.0
    )

    axis.grid(
        alpha=0.25,
    )

    axis.legend(
        frameon=False
    )

    figure.tight_layout()

    save_figure(
        figure,
        output_directory,
        "figure_veto_sensitivity",
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--omnisafe",
        default=(
            "outputs/final/"
            "omnisafe_30seed/"
            "omnisafe_pilot_summary.csv"
        ),
    )

    parser.add_argument(
        "--lagrange",
        default=(
            "outputs/final/"
            "lagrange_30seed/"
            "lagrange_sensitivity_summary.csv"
        ),
    )

    parser.add_argument(
        "--veto",
        default=(
            "outputs/final/"
            "veto_30seed/"
            "veto_run_summary.csv"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "outputs/final/analysis"
        ),
    )

    arguments = parser.parse_args()

    output_directory = Path(
        arguments.output
    )

    figure_directory = (
        output_directory
        / "figures"
    )

    table_directory = (
        output_directory
        / "tables"
    )

    figure_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    primary, veto_rows = (
        load_primary_results(
            Path(arguments.omnisafe),
            Path(arguments.lagrange),
            Path(arguments.veto),
        )
    )

    expected_primary_rows = (
        len(METHOD_ORDER)
        * len(
            ENVIRONMENT_ORDER
        )
        * 30
    )

    if len(primary) != expected_primary_rows:
        raise RuntimeError(
            "Expected "
            f"{expected_primary_rows} "
            "primary rows, found "
            f"{len(primary)}."
        )

    if len(veto_rows) != 360:
        raise RuntimeError(
            "Expected 360 veto sensitivity rows, "
            f"found {len(veto_rows)}."
        )

    primary_summary = (
        summarize_primary(
            primary
        )
    )

    paired_tests = (
        calculate_paired_tests(
            primary
        )
    )

    (
        veto_auc_seed,
        veto_auc_summary,
    ) = calculate_veto_auc(
        veto_rows
    )

    primary.to_csv(
        table_directory
        / "primary_seed_level.csv",
        index=False,
    )

    primary_summary.to_csv(
        table_directory
        / "primary_summary.csv",
        index=False,
    )

    paired_tests.to_csv(
        table_directory
        / "paired_tests_holm.csv",
        index=False,
    )

    veto_auc_seed.to_csv(
        table_directory
        / "veto_auc_seed_level.csv",
        index=False,
    )

    veto_auc_summary.to_csv(
        table_directory
        / "veto_auc_summary.csv",
        index=False,
    )

    plot_primary_metric(
        primary_summary,
        metric="violation_rate",
        ylabel=(
            "Held-out policy violation rate"
        ),
        output_directory=
            figure_directory,
        name=(
            "figure_primary_violations"
        ),
    )

    plot_primary_metric(
        primary_summary,
        metric="true_utility",
        ylabel=(
            "Held-out true utility"
        ),
        output_directory=
            figure_directory,
        name=(
            "figure_primary_utility"
        ),
    )

    plot_primary_metric(
        primary_summary,
        metric="proxy_true_gap",
        ylabel=(
            "Proxy–true objective gap"
        ),
        output_directory=
            figure_directory,
        name=(
            "figure_proxy_true_gap"
        ),
    )

    plot_veto_sensitivity(
        veto_rows,
        figure_directory,
    )

    manifest = {
        "primary_rows":
            len(primary),
        "veto_sensitivity_rows":
            len(veto_rows),
        "random_seeds":
            sorted(
                primary[
                    "seed"
                ].unique()
                .tolist()
            ),
        "methods":
            METHOD_ORDER,
        "environments":
            ENVIRONMENT_ORDER,
        "bootstrap_resamples":
            10_000,
        "confidence_level":
            0.95,
        "multiple_testing":
            "Holm",
        "python":
            platform.python_version(),
        "numpy":
            np.__version__,
        "pandas":
            pd.__version__,
        "scipy":
            scipy.__version__,
        "matplotlib":
            matplotlib.__version__,
    }

    (
        output_directory
        / "analysis_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 100)
    print("FINAL NEURAL ANALYSIS")
    print("=" * 100)
    print(
        "Primary rows:",
        len(primary),
    )
    print(
        "Veto sensitivity rows:",
        len(veto_rows),
    )

    print("\nPRIMARY RESULTS")
    print(
        primary_summary[
            primary_summary[
                "metric"
            ].isin(
                [
                    "violation_rate",
                    "true_utility",
                ]
            )
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\nPAIRED TESTS")
    print(
        paired_tests[
            [
                "environment",
                "comparison",
                "metric",
                "mean_difference",
                "rank_biserial",
                "p_value",
                "p_holm",
                "significant_0_05",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )

    print(
        "\nSaved tables to:",
        table_directory,
    )

    print(
        "Saved figures to:",
        figure_directory,
    )


if __name__ == "__main__":
    main()