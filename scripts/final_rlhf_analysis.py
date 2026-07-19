"""Final statistics and figures for the calibrated RLHF-style study."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from scripts.final_neural_analysis import (
    bootstrap_mean_interval,
    holm_adjust,
    manual_trapezoidal_auc,
    paired_wilcoxon,
)


METHOD_ORDER = [
    "reward_only",
    "fixed_penalty",
    "lagrangian",
    "hard_veto",
]

METHOD_LABELS = {
    "reward_only": "Reward only",
    "fixed_penalty": "Fixed penalty",
    "lagrangian": "Lagrangian",
    "hard_veto": "Hard veto",
}

METHOD_COLOURS = {
    "reward_only": "#0072B2",
    "fixed_penalty": "#E69F00",
    "lagrangian": "#009E73",
    "hard_veto": "#D55E00",
}

METRICS = [
    "mean_true_utility",
    "violation_rate",
    "sycophancy_rate",
    "deception_rate",
    "refusal_rate",
    "reward_true_gap",
]


def validate_results(
    results: pd.DataFrame,
) -> None:
    required = {
        "seed",
        "perturbation",
        "method",
        *METRICS,
    }

    missing = sorted(
        required.difference(
            results.columns
        )
    )

    if missing:
        raise ValueError(
            "Missing columns: "
            + ", ".join(missing)
        )

    expected_rows = 600

    if len(results) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, "
            f"found {len(results)}."
        )

    expected_methods = set(
        METHOD_ORDER
    )

    actual_methods = set(
        results["method"].unique()
    )

    if actual_methods != expected_methods:
        raise ValueError(
            "Unexpected method set: "
            f"{sorted(actual_methods)}"
        )

    seed_counts = (
        results.groupby(
            [
                "perturbation",
                "method",
            ]
        )["seed"]
        .nunique()
    )

    if not seed_counts.eq(30).all():
        raise ValueError(
            "Every perturbation/method cell "
            "must contain 30 seeds."
        )


def summarize_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    for (
        perturbation,
        method,
    ), group in results.groupby(
        [
            "perturbation",
            "method",
        ]
    ):
        for metric_index, metric in enumerate(
            METRICS
        ):
            values = group[
                metric
            ].to_numpy(
                dtype=float
            )

            lower, upper = (
                bootstrap_mean_interval(
                    values,
                    resamples=10_000,
                    seed=(
                        20260719
                        + metric_index
                        + len(records)
                    ),
                )
            )

            records.append(
                {
                    "perturbation":
                        float(
                            perturbation
                        ),
                    "method":
                        method,
                    "metric":
                        metric,
                    "n_seeds":
                        int(
                            group["seed"]
                            .nunique()
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
    results: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    comparison_methods = [
        "fixed_penalty",
        "lagrangian",
        "hard_veto",
    ]

    test_metrics = [
        "violation_rate",
        "mean_true_utility",
    ]

    for perturbation in sorted(
        results[
            "perturbation"
        ].unique()
    ):
        perturbation_rows = results[
            results[
                "perturbation"
            ]
            == perturbation
        ]

        baseline = (
            perturbation_rows[
                perturbation_rows[
                    "method"
                ]
                == "reward_only"
            ][
                [
                    "seed",
                    *test_metrics,
                ]
            ]
            .set_index(
                "seed"
            )
        )

        for method in comparison_methods:
            comparison = (
                perturbation_rows[
                    perturbation_rows[
                        "method"
                    ]
                    == method
                ][
                    [
                        "seed",
                        *test_metrics,
                    ]
                ]
                .set_index(
                    "seed"
                )
            )

            paired = baseline.join(
                comparison,
                how="inner",
                lsuffix=
                    "_reward_only",
                rsuffix=
                    "_method",
            )

            for metric in test_metrics:
                result = paired_wilcoxon(
                    paired[
                        f"{metric}_method"
                    ],
                    paired[
                        (
                            f"{metric}_"
                            "reward_only"
                        )
                    ],
                )

                records.append(
                    {
                        "perturbation":
                            float(
                                perturbation
                            ),
                        "comparison": (
                            f"{method} "
                            "vs reward_only"
                        ),
                        "method":
                            method,
                        "baseline":
                            "reward_only",
                        "metric":
                            metric,
                        **result,
                    }
                )

    frame = pd.DataFrame(
        records
    )

    frame["p_holm"] = np.nan

    for metric in test_metrics:
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


def calculate_seed_auc(
    results: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    seed_records = []

    for (
        method,
        seed,
    ), group in results.groupby(
        [
            "method",
            "seed",
        ]
    ):
        group = group.sort_values(
            "perturbation"
        )

        seed_records.append(
            {
                "method":
                    method,
                "seed":
                    int(seed),
                "violation_auc":
                    manual_trapezoidal_auc(
                        group[
                            "perturbation"
                        ],
                        group[
                            "violation_rate"
                        ],
                    ),
                "utility_loss_auc":
                    manual_trapezoidal_auc(
                        group[
                            "perturbation"
                        ],
                        1.0
                        - group[
                            "mean_true_utility"
                        ],
                    ),
                "sycophancy_auc":
                    manual_trapezoidal_auc(
                        group[
                            "perturbation"
                        ],
                        group[
                            "sycophancy_rate"
                        ],
                    ),
                "deception_auc":
                    manual_trapezoidal_auc(
                        group[
                            "perturbation"
                        ],
                        group[
                            "deception_rate"
                        ],
                    ),
            }
        )

    seed_frame = pd.DataFrame(
        seed_records
    )

    summary_records = []

    auc_metrics = [
        "violation_auc",
        "utility_loss_auc",
        "sycophancy_auc",
        "deception_auc",
    ]

    for method, group in (
        seed_frame.groupby(
            "method"
        )
    ):
        for metric_index, metric in enumerate(
            auc_metrics
        ):
            values = group[
                metric
            ].to_numpy(
                dtype=float
            )

            lower, upper = (
                bootstrap_mean_interval(
                    values,
                    resamples=10_000,
                    seed=(
                        20260719
                        + metric_index
                        + len(
                            summary_records
                        )
                    ),
                )
            )

            summary_records.append(
                {
                    "method":
                        method,
                    "metric":
                        metric,
                    "n_seeds":
                        int(
                            group["seed"]
                            .nunique()
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
    directory: Path,
    name: str,
) -> None:
    figure.savefig(
        directory
        / f"{name}.pdf",
        bbox_inches="tight",
    )

    figure.savefig(
        directory
        / f"{name}.svg",
        bbox_inches="tight",
    )

    figure.savefig(
        directory
        / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def plot_metric(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    figure_directory: Path,
    filename: str,
) -> None:
    metric_rows = summary[
        summary["metric"]
        == metric
    ]

    figure, axis = plt.subplots(
        figsize=(
            8.5,
            5.5,
        )
    )

    for method in METHOD_ORDER:
        rows = (
            metric_rows[
                metric_rows[
                    "method"
                ]
                == method
            ]
            .sort_values(
                "perturbation"
            )
        )

        x_values = rows[
            "perturbation"
        ].to_numpy(
            dtype=float
        )

        means = rows[
            "mean"
        ].to_numpy(
            dtype=float
        )

        lower = rows[
            "ci95_lower"
        ].to_numpy(
            dtype=float
        )

        upper = rows[
            "ci95_upper"
        ].to_numpy(
            dtype=float
        )

        colour = (
            METHOD_COLOURS[
                method
            ]
        )

        axis.plot(
            x_values,
            means,
            marker="o",
            linewidth=2.1,
            markersize=5,
            color=colour,
            label=
                METHOD_LABELS[
                    method
                ],
        )

        axis.fill_between(
            x_values,
            lower,
            upper,
            color=colour,
            alpha=0.16,
        )

    axis.set_xlabel(
        "Reward-model perturbation strength"
    )

    axis.set_ylabel(
        ylabel
    )

    axis.set_xlim(
        -0.02,
        1.02,
    )

    axis.set_ylim(
        bottom=0.0
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend(
        frameon=False,
        ncol=2,
    )

    figure.tight_layout()

    save_figure(
        figure,
        figure_directory,
        filename,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "outputs/final/"
            "rlhf_calibrated_30seed/"
            "rlhf_calibrated_results.csv"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "outputs/final/"
            "rlhf_analysis"
        ),
    )

    arguments = parser.parse_args()

    input_path = Path(
        arguments.input
    )

    output_directory = Path(
        arguments.output
    )

    table_directory = (
        output_directory
        / "tables"
    )

    figure_directory = (
        output_directory
        / "figures"
    )

    table_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = pd.read_csv(
        input_path
    )

    validate_results(
        results
    )

    summary = summarize_results(
        results
    )

    paired_tests = (
        calculate_paired_tests(
            results
        )
    )

    (
        seed_auc,
        auc_summary,
    ) = calculate_seed_auc(
        results
    )

    summary.to_csv(
        table_directory
        / "rlhf_summary.csv",
        index=False,
    )

    paired_tests.to_csv(
        table_directory
        / "rlhf_paired_tests_holm.csv",
        index=False,
    )

    seed_auc.to_csv(
        table_directory
        / "rlhf_auc_seed_level.csv",
        index=False,
    )

    auc_summary.to_csv(
        table_directory
        / "rlhf_auc_summary.csv",
        index=False,
    )

    plot_metric(
        summary,
        metric="violation_rate",
        ylabel="Constraint violation rate",
        figure_directory=
            figure_directory,
        filename=
            "figure_rlhf_violations",
    )

    plot_metric(
        summary,
        metric="mean_true_utility",
        ylabel="Held-out true utility",
        figure_directory=
            figure_directory,
        filename=
            "figure_rlhf_utility",
    )

    plot_metric(
        summary,
        metric="sycophancy_rate",
        ylabel="Sycophantic response rate",
        figure_directory=
            figure_directory,
        filename=
            "figure_rlhf_sycophancy",
    )

    plot_metric(
        summary,
        metric="deception_rate",
        ylabel="Deceptive response rate",
        figure_directory=
            figure_directory,
        filename=
            "figure_rlhf_deception",
    )

    manifest = {
        "input":
            str(input_path),
        "rows":
            len(results),
        "seeds":
            sorted(
                results["seed"]
                .unique()
                .tolist()
            ),
        "perturbations":
            sorted(
                results[
                    "perturbation"
                ]
                .unique()
                .tolist()
            ),
        "methods":
            METHOD_ORDER,
        "bootstrap_resamples":
            10_000,
        "multiple_testing":
            "Holm",
        "scope": (
            "Controlled RLHF-style "
            "response-selection analogue"
        ),
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
        / "rlhf_analysis_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 115)
    print(
        "FINAL CALIBRATED RLHF ANALYSIS"
    )
    print("=" * 115)

    display = summary[
        summary["metric"].isin(
            [
                "violation_rate",
                "mean_true_utility",
                "sycophancy_rate",
            ]
        )
    ]

    print(
        display.round(5)
        .to_string(index=False)
    )

    print("\nAUC SUMMARY")

    print(
        auc_summary.round(5)
        .to_string(index=False)
    )

    print(
        "\nSaved tables:",
        table_directory,
    )

    print(
        "Saved figures:",
        figure_directory,
    )


if __name__ == "__main__":
    main()