"""Publication-ready vector and raster figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "reward_only": "#0072B2",
    "fixed_penalty": "#E69F00",
    "lagrangian": "#009E73",
    "hard_shield": "#CC79A7",
}

MARKERS = {
    "reward_only": "o",
    "fixed_penalty": "s",
    "lagrangian": "^",
    "hard_shield": "D",
}

PRIMARY_METRICS = {
    "adversarial": (
        "policy_violation_rate",
        "Policy violation rate",
    ),
    "truth_proxy": (
        "false_answer_rate",
        "False-answer rate",
    ),
    "monitoring_shift": (
        "monitoring_gap",
        "Monitoring-conditioned unsafe gap",
    ),
}


def save_figure(
    figure,
    base_path: Path,
) -> None:
    figure.tight_layout()

    figure.savefig(
        f"{base_path}.pdf",
        bbox_inches="tight",
    )

    figure.savefig(
        f"{base_path}.svg",
        bbox_inches="tight",
    )

    figure.savefig(
        f"{base_path}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_primary_results(
    aggregated: pd.DataFrame,
    figures_directory: str,
) -> None:
    for environment, specification in (
        PRIMARY_METRICS.items()
    ):
        metric, y_label = specification

        subset = aggregated[
            (aggregated["environment"] == environment)
            & np.isclose(
                aggregated["constraint_error"],
                0.0,
            )
        ]

        magnitudes = sorted(
            subset["corruption_magnitude"].unique()
        )

        figure, axes = plt.subplots(
            1,
            len(magnitudes),
            figsize=(4.2 * len(magnitudes), 3.7),
            sharey=True,
        )

        axes = np.atleast_1d(axes)

        for axis, magnitude in zip(
            axes,
            magnitudes,
        ):
            panel = subset[
                np.isclose(
                    subset["corruption_magnitude"],
                    magnitude,
                )
            ]

            for agent in COLORS:
                agent_data = panel[
                    panel["agent"] == agent
                ].sort_values(
                    "corruption_probability"
                )

                if len(agent_data) == 0:
                    continue

                x_values = agent_data[
                    "corruption_probability"
                ].to_numpy()

                y_values = agent_data[
                    f"{metric}_mean"
                ].to_numpy()

                axis.plot(
                    x_values,
                    y_values,
                    color=COLORS[agent],
                    marker=MARKERS[agent],
                    label=agent.replace("_", " "),
                )

                axis.fill_between(
                    x_values,
                    agent_data[
                        f"{metric}_ci_low"
                    ],
                    agent_data[
                        f"{metric}_ci_high"
                    ],
                    color=COLORS[agent],
                    alpha=0.15,
                )

            axis.set_title(
                f"Corruption magnitude = {magnitude:g}"
            )

            axis.set_xlabel(
                "Reward-corruption probability"
            )

            axis.set_ylim(-0.04, 1.04)
            axis.grid(alpha=0.2)

        axes[0].set_ylabel(y_label)

        axes[-1].legend(
            frameon=False,
            fontsize=8,
        )

        save_figure(
            figure,
            Path(figures_directory)
            / f"{environment}_primary",
        )


def plot_constraint_error(
    aggregated: pd.DataFrame,
    figures_directory: str,
) -> None:
    subset = aggregated[
        (aggregated["environment"] == "adversarial")
        & np.isclose(
            aggregated["corruption_probability"],
            1.0,
        )
    ]

    figure, axis = plt.subplots(
        figsize=(6.2, 4.0)
    )

    for agent in COLORS:
        agent_data = subset[
            subset["agent"] == agent
        ]

        if len(agent_data) == 0:
            continue

        grouped = (
            agent_data
            .groupby("constraint_error")[
                "policy_violation_rate_mean"
            ]
            .mean()
            .reset_index()
        )

        axis.plot(
            grouped["constraint_error"],
            grouped["policy_violation_rate_mean"],
            color=COLORS[agent],
            marker=MARKERS[agent],
            label=agent.replace("_", " "),
        )

    axis.set_xlabel("Constraint-model error")
    axis.set_ylabel("Policy violation rate")

    axis.set_title(
        "Safety sensitivity to constraint misspecification"
    )

    axis.grid(alpha=0.2)
    axis.legend(frameon=False)

    save_figure(
        figure,
        Path(figures_directory)
        / "constraint_error_sensitivity",
    )


def plot_safety_utility(
    aggregated: pd.DataFrame,
    figures_directory: str,
) -> None:
    subset = aggregated[
        (aggregated["environment"] == "adversarial")
        & np.isclose(
            aggregated["constraint_error"],
            0.0,
        )
    ]

    figure, axis = plt.subplots(
        figsize=(6.2, 4.2)
    )

    for agent in COLORS:
        agent_data = subset[
            subset["agent"] == agent
        ]

        axis.scatter(
            agent_data[
                "policy_violation_rate_mean"
            ],
            agent_data["true_utility_mean"],
            s=35,
            color=COLORS[agent],
            marker=MARKERS[agent],
            label=agent.replace("_", " "),
            alpha=0.8,
        )

    axis.set_xlabel("Policy violation rate")
    axis.set_ylabel("True utility")
    axis.set_title("Safety–utility outcomes")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)

    save_figure(
        figure,
        Path(figures_directory)
        / "safety_utility_frontier",
    )


def generate_all_plots(
    aggregated: pd.DataFrame,
    learning_curves: pd.DataFrame,
    figures_directory: str,
) -> None:
    Path(figures_directory).mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    plot_primary_results(
        aggregated,
        figures_directory,
    )

    plot_constraint_error(
        aggregated,
        figures_directory,
    )

    plot_safety_utility(
        aggregated,
        figures_directory,
    )