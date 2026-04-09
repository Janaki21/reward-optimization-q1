from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRIMARY_METRIC_MAP = {
    "adversarial": ("constraint_violation_rate_mean", "constraint_violation_rate_ci_low", "constraint_violation_rate_ci_high", "Constraint Violation Rate"),
    "truth_reward": ("false_answer_rate_mean", "false_answer_rate_ci_low", "false_answer_rate_ci_high", "False Answer Rate"),
    "monitoring": ("deception_index_mean", "deception_index_ci_low", "deception_index_ci_high", "Deception Index"),
}


def _savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_primary_metric_curves(agg_df: pd.DataFrame, figures_dir: str) -> None:
    for env_name in sorted(agg_df["env_name"].unique()):
        latest_scale = int(agg_df[agg_df["env_name"] == env_name]["episode_scale"].max())
        sub = agg_df[(agg_df["env_name"] == env_name) & (agg_df["episode_scale"] == latest_scale)]

        mean_col, low_col, high_col, ylabel = PRIMARY_METRIC_MAP[env_name]

        plt.figure(figsize=(8, 5))
        for agent in ["reward", "soft", "constrained"]:
            a = sub[sub["agent_name"] == agent].sort_values("perturbation")
            x = a["perturbation"].to_numpy(dtype=float)
            y = a[mean_col].to_numpy(dtype=float)
            yerr_low = y - a[low_col].to_numpy(dtype=float)
            yerr_high = a[high_col].to_numpy(dtype=float) - y

            plt.errorbar(x, y, yerr=[yerr_low, yerr_high], marker="o", capsize=3, label=agent)

        plt.xlabel("Perturbation Probability")
        plt.ylabel(ylabel)
        plt.title(f"{env_name.replace('_', ' ').title()} — Primary Metric")
        plt.legend()
        plt.grid(True, alpha=0.3)

        _savefig(str(Path(figures_dir) / f"{env_name}_primary_metric.png"))


def plot_utility_curves(agg_df: pd.DataFrame, figures_dir: str) -> None:
    for env_name in sorted(agg_df["env_name"].unique()):
        latest_scale = int(agg_df[agg_df["env_name"] == env_name]["episode_scale"].max())
        sub = agg_df[(agg_df["env_name"] == env_name) & (agg_df["episode_scale"] == latest_scale)]

        plt.figure(figsize=(8, 5))
        for agent in ["reward", "soft", "constrained"]:
            a = sub[sub["agent_name"] == agent].sort_values("perturbation")
            x = a["perturbation"].to_numpy(dtype=float)
            y = a["mean_utility_mean"].to_numpy(dtype=float)
            yerr_low = y - a["mean_utility_ci_low"].to_numpy(dtype=float)
            yerr_high = a["mean_utility_ci_high"].to_numpy(dtype=float) - y

            plt.errorbar(x, y, yerr=[yerr_low, yerr_high], marker="o", capsize=3, label=agent)

        plt.xlabel("Perturbation Probability")
        plt.ylabel("Mean Utility")
        plt.title(f"{env_name.replace('_', ' ').title()} — Utility Trade-off")
        plt.legend()
        plt.grid(True, alpha=0.3)

        _savefig(str(Path(figures_dir) / f"{env_name}_utility_tradeoff.png"))


def plot_learning_curves(curve_df: pd.DataFrame, figures_dir: str) -> None:
    latest_scale = int(curve_df["episode_scale"].max())

    plot_specs = [
        ("avg_reward_so_far", "Average Reward"),
        ("cvr_so_far", "Constraint Violation Rate"),
        ("far_so_far", "False Answer Rate"),
        ("deception_index_so_far", "Deception Index"),
    ]

    for env_name in sorted(curve_df["env_name"].unique()):
        env_sub = curve_df[(curve_df["env_name"] == env_name) & (curve_df["episode_scale"] == latest_scale)]

        for metric_col, ylabel in plot_specs:
            if metric_col == "far_so_far" and env_name != "truth_reward":
                continue
            if metric_col == "deception_index_so_far" and env_name != "monitoring":
                continue

            plt.figure(figsize=(8, 5))
            for agent in ["reward", "soft", "constrained"]:
                a = env_sub[env_sub["agent_name"] == agent]
                if len(a) == 0:
                    continue

                mean_curve = (
                    a.groupby("episode")[metric_col]
                    .mean()
                    .reset_index()
                    .sort_values("episode")
                )
                plt.plot(mean_curve["episode"], mean_curve[metric_col], label=agent)

            plt.xlabel("Episodes")
            plt.ylabel(ylabel)
            plt.title(f"{env_name.replace('_', ' ').title()} — Learning Curve ({ylabel})")
            plt.legend()
            plt.grid(True, alpha=0.3)

            _savefig(str(Path(figures_dir) / f"{env_name}_learning_curve_{metric_col}.png"))


def plot_ablation_bars(agg_df: pd.DataFrame, figures_dir: str) -> None:
    latest_scale = int(agg_df["episode_scale"].max())

    for env_name in sorted(agg_df["env_name"].unique()):
        sub = agg_df[(agg_df["env_name"] == env_name) & (agg_df["episode_scale"] == latest_scale)]

        chosen_p = 0.5 if 0.5 in set(sub["perturbation"].round(10)) else float(sorted(sub["perturbation"].unique())[len(sub["perturbation"].unique()) // 2])
        psub = sub[np.isclose(sub["perturbation"], chosen_p)]

        if env_name == "adversarial":
            mean_col = "constraint_violation_rate_mean"
            ylabel = "Constraint Violation Rate"
        elif env_name == "truth_reward":
            mean_col = "false_answer_rate_mean"
            ylabel = "False Answer Rate"
        else:
            mean_col = "deception_index_mean"
            ylabel = "Deception Index"

        psub = psub.set_index("agent_name").reindex(["reward", "soft", "constrained"]).reset_index()

        plt.figure(figsize=(7, 5))
        plt.bar(psub["agent_name"], psub[mean_col])
        plt.xlabel("Agent")
        plt.ylabel(ylabel)
        plt.title(f"{env_name.replace('_', ' ').title()} — Ablation at p={chosen_p:.1f}")
        plt.grid(True, axis="y", alpha=0.3)

        _savefig(str(Path(figures_dir) / f"{env_name}_ablation_bar.png"))


def plot_scaling_analysis(agg_df: pd.DataFrame, figures_dir: str) -> None:
    for env_name in sorted(agg_df["env_name"].unique()):
        plt.figure(figsize=(8, 5))
        env_sub = agg_df[agg_df["env_name"] == env_name]

        if env_name == "adversarial":
            metric_col = "constraint_violation_rate_mean"
            ylabel = "Average CVR Across Perturbations"
        elif env_name == "truth_reward":
            metric_col = "false_answer_rate_mean"
            ylabel = "Average FAR Across Perturbations"
        else:
            metric_col = "deception_index_mean"
            ylabel = "Average Deception Index Across Perturbations"

        for agent in ["reward", "soft", "constrained"]:
            a = env_sub[env_sub["agent_name"] == agent]
            scale_mean = (
                a.groupby("episode_scale")[metric_col]
                .mean()
                .reset_index()
                .sort_values("episode_scale")
            )
            plt.plot(scale_mean["episode_scale"], scale_mean[metric_col], marker="o", label=agent)

        plt.xlabel("Episode Scale")
        plt.ylabel(ylabel)
        plt.title(f"{env_name.replace('_', ' ').title()} — Scaling Analysis")
        plt.legend()
        plt.grid(True, alpha=0.3)

        _savefig(str(Path(figures_dir) / f"{env_name}_scaling_analysis.png"))


def plot_integrated_summary(agg_df: pd.DataFrame, figures_dir: str) -> None:
    latest_scale = int(agg_df["episode_scale"].max())

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    env_order = ["adversarial", "truth_reward", "monitoring"]
    for idx, env_name in enumerate(env_order):
        ax = axes[idx]
        sub = agg_df[(agg_df["env_name"] == env_name) & (agg_df["episode_scale"] == latest_scale)]

        if env_name == "adversarial":
            mean_col = "constraint_violation_rate_mean"
            ylabel = "CVR"
        elif env_name == "truth_reward":
            mean_col = "false_answer_rate_mean"
            ylabel = "FAR"
        else:
            mean_col = "deception_index_mean"
            ylabel = "Deception"

        for agent in ["reward", "soft", "constrained"]:
            a = sub[sub["agent_name"] == agent].sort_values("perturbation")
            ax.plot(a["perturbation"], a[mean_col], marker="o", label=agent)

        ax.set_title(env_name.replace("_", " ").title())
        ax.set_xlabel("Perturbation")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    ax = axes[3]
    sub = agg_df[agg_df["episode_scale"] == latest_scale]
    utility_summary = (
        sub.groupby(["env_name", "agent_name"])["mean_utility_mean"]
        .mean()
        .reset_index()
    )

    labels = []
    values = []
    for env_name in env_order:
        for agent in ["reward", "soft", "constrained"]:
            r = utility_summary[
                (utility_summary["env_name"] == env_name)
                & (utility_summary["agent_name"] == agent)
            ]
            if len(r) == 0:
                continue
            labels.append(f"{env_name[:4]}-{agent[:4]}")
            values.append(float(r["mean_utility_mean"].iloc[0]))

    ax.bar(labels, values)
    ax.set_title("Average Utility Across Environments")
    ax.set_xlabel("Environment-Agent")
    ax.set_ylabel("Mean Utility")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="upper center", ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(str(Path(figures_dir) / "integrated_summary_figure.png"), dpi=300, bbox_inches="tight")
    plt.close()


def generate_all_plots(agg_df: pd.DataFrame, curve_df: pd.DataFrame, figures_dir: str) -> None:
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    plot_primary_metric_curves(agg_df, figures_dir)
    plot_utility_curves(agg_df, figures_dir)
    plot_learning_curves(curve_df, figures_dir)
    plot_ablation_bars(agg_df, figures_dir)
    plot_scaling_analysis(agg_df, figures_dir)
    plot_integrated_summary(agg_df, figures_dir)