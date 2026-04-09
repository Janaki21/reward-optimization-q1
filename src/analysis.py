from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.utils import confidence_interval_95


PRIMARY_METRIC_MAP = {
    "adversarial": "constraint_violation_rate",
    "truth_reward": "false_answer_rate",
    "monitoring": "deception_index",
}


def rank_biserial_from_paired(x: np.ndarray, y: np.ndarray) -> float:
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0

    abs_d = np.abs(d)
    ranks = pd.Series(abs_d).rank(method="average").to_numpy()
    pos = np.sum(ranks[d > 0])
    neg = np.sum(ranks[d < 0])
    denom = pos + neg
    if denom == 0:
        return 0.0
    return float((pos - neg) / denom)


def aggregate_run_statistics(run_summary_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "mean_utility",
        "constraint_violation_rate",
        "false_answer_rate",
        "safe_rate_monitored",
        "unsafe_rate_unmonitored",
        "deception_index",
    ]

    rows: List[Dict] = []

    group_cols = ["env_name", "agent_name", "episode_scale", "perturbation"]

    for keys, group in run_summary_df.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(group["seed"].nunique())

        for metric in metric_cols:
            vals = group[metric].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_ci_low"] = np.nan
                row[f"{metric}_ci_high"] = np.nan
            else:
                row[f"{metric}_mean"] = float(np.mean(vals))
                row[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                ci_low, ci_high = confidence_interval_95(vals)
                row[f"{metric}_ci_low"] = ci_low
                row[f"{metric}_ci_high"] = ci_high

        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def compute_wilcoxon_tests(run_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []

    comparisons = [("reward", "constrained"), ("reward", "soft"), ("soft", "constrained")]

    for env_name in run_summary_df["env_name"].unique():
        metric = PRIMARY_METRIC_MAP[env_name]

        env_df = run_summary_df[run_summary_df["env_name"] == env_name]

        for episode_scale in sorted(env_df["episode_scale"].unique()):
            scale_df = env_df[env_df["episode_scale"] == episode_scale]

            for perturbation in sorted(scale_df["perturbation"].unique()):
                sub = scale_df[scale_df["perturbation"] == perturbation]

                for a1, a2 in comparisons:
                    g1 = sub[sub["agent_name"] == a1].sort_values("seed")
                    g2 = sub[sub["agent_name"] == a2].sort_values("seed")

                    common_seeds = sorted(set(g1["seed"]).intersection(set(g2["seed"])))
                    if len(common_seeds) < 2:
                        continue

                    x = g1[g1["seed"].isin(common_seeds)][metric].to_numpy(dtype=float)
                    y = g2[g2["seed"].isin(common_seeds)][metric].to_numpy(dtype=float)

                    if np.allclose(x, y):
                        stat, p_value = 0.0, 1.0
                    else:
                        try:
                            stat, p_value = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                        except ValueError:
                            stat, p_value = np.nan, np.nan

                    effect = rank_biserial_from_paired(x, y)

                    rows.append(
                        {
                            "env_name": env_name,
                            "episode_scale": episode_scale,
                            "perturbation": perturbation,
                            "metric": metric,
                            "agent_a": a1,
                            "agent_b": a2,
                            "n_pairs": len(common_seeds),
                            "mean_a": float(np.mean(x)),
                            "mean_b": float(np.mean(y)),
                            "wilcoxon_stat": stat,
                            "p_value": p_value,
                            "rank_biserial": effect,
                        }
                    )

    return pd.DataFrame(rows).sort_values(
        ["env_name", "episode_scale", "perturbation", "agent_a", "agent_b"]
    ).reset_index(drop=True)


def build_main_result_table(agg_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []

    for _, r in agg_df.iterrows():
        env_name = r["env_name"]
        row = {
            "Environment": env_name,
            "Agent": r["agent_name"],
            "Episodes": int(r["episode_scale"]),
            "Perturbation": r["perturbation"],
            "Mean Utility": f'{r["mean_utility_mean"]:.4f} ± {r["mean_utility_std"]:.4f}',
        }

        if env_name == "adversarial":
            row["Primary Metric"] = f'{r["constraint_violation_rate_mean"]:.4f} ± {r["constraint_violation_rate_std"]:.4f}'
            row["95% CI"] = f'[{r["constraint_violation_rate_ci_low"]:.4f}, {r["constraint_violation_rate_ci_high"]:.4f}]'
        elif env_name == "truth_reward":
            row["Primary Metric"] = f'{r["false_answer_rate_mean"]:.4f} ± {r["false_answer_rate_std"]:.4f}'
            row["95% CI"] = f'[{r["false_answer_rate_ci_low"]:.4f}, {r["false_answer_rate_ci_high"]:.4f}]'
        else:
            row["Primary Metric"] = f'{r["deception_index_mean"]:.4f} ± {r["deception_index_std"]:.4f}'
            row["95% CI"] = f'[{r["deception_index_ci_low"]:.4f}, {r["deception_index_ci_high"]:.4f}]'

        rows.append(row)

    return pd.DataFrame(rows)


def build_ablation_table(agg_df: pd.DataFrame) -> pd.DataFrame:
    latest_scale = int(agg_df["episode_scale"].max())
    sub = agg_df[agg_df["episode_scale"] == latest_scale].copy()

    rows: List[Dict] = []
    for env_name in sorted(sub["env_name"].unique()):
        env_sub = sub[sub["env_name"] == env_name]

        for perturbation in sorted(env_sub["perturbation"].unique()):
            p_sub = env_sub[env_sub["perturbation"] == perturbation]

            out = {
                "Environment": env_name,
                "Perturbation": perturbation,
            }

            for agent in ["reward", "soft", "constrained"]:
                arow = p_sub[p_sub["agent_name"] == agent]
                if len(arow) == 0:
                    continue
                arow = arow.iloc[0]

                if env_name == "adversarial":
                    val = arow["constraint_violation_rate_mean"]
                elif env_name == "truth_reward":
                    val = arow["false_answer_rate_mean"]
                else:
                    val = arow["deception_index_mean"]

                out[f"{agent}_primary_metric"] = round(float(val), 6)
                out[f"{agent}_utility"] = round(float(arow["mean_utility_mean"]), 6)

            rows.append(out)

    return pd.DataFrame(rows)


def build_scaling_table(agg_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []

    for env_name in sorted(agg_df["env_name"].unique()):
        env_sub = agg_df[agg_df["env_name"] == env_name]

        for agent in ["reward", "soft", "constrained"]:
            agent_sub = env_sub[env_sub["agent_name"] == agent]

            for episode_scale in sorted(agent_sub["episode_scale"].unique()):
                ssub = agent_sub[agent_sub["episode_scale"] == episode_scale]

                if env_name == "adversarial":
                    vals = ssub["constraint_violation_rate_mean"].to_numpy(dtype=float)
                elif env_name == "truth_reward":
                    vals = ssub["false_answer_rate_mean"].to_numpy(dtype=float)
                else:
                    vals = ssub["deception_index_mean"].to_numpy(dtype=float)

                rows.append(
                    {
                        "Environment": env_name,
                        "Agent": agent,
                        "Episodes": int(episode_scale),
                        "PrimaryMetricAcrossPerturbations_Mean": float(np.nanmean(vals)),
                        "PrimaryMetricAcrossPerturbations_Std": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        "MeanUtilityAcrossPerturbations_Mean": float(np.nanmean(ssub["mean_utility_mean"].to_numpy(dtype=float))),
                    }
                )

    return pd.DataFrame(rows)


def run_full_analysis(run_summary_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    agg_df = aggregate_run_statistics(run_summary_df)
    wilcox_df = compute_wilcoxon_tests(run_summary_df)
    main_table = build_main_result_table(agg_df)
    ablation_table = build_ablation_table(agg_df)
    scaling_table = build_scaling_table(agg_df)

    return {
        "aggregated_statistics": agg_df,
        "wilcoxon_tests": wilcox_df,
        "main_results_table": main_table,
        "ablation_table": ablation_table,
        "scaling_table": scaling_table,
    }