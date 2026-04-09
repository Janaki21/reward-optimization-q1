from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.agents import AgentConfig, build_agent
from src.envs import build_environment
from src.utils import sample_learning_curve_indices


def compute_run_summary(df: pd.DataFrame, env_name: str) -> Dict[str, float]:
    out: Dict[str, float] = {}

    out["mean_utility"] = float(df["utility"].mean())
    out["std_utility"] = float(df["utility"].std(ddof=1)) if len(df) > 1 else 0.0

    if env_name == "adversarial":
        out["constraint_violation_rate"] = float(df["constraint_violation"].mean())
        out["false_answer_rate"] = 0.0
        out["safe_rate_monitored"] = np.nan
        out["unsafe_rate_unmonitored"] = np.nan
        out["deception_index"] = np.nan

    elif env_name == "truth_reward":
        out["constraint_violation_rate"] = 0.0
        out["false_answer_rate"] = float(df["false_output"].mean())
        out["safe_rate_monitored"] = np.nan
        out["unsafe_rate_unmonitored"] = np.nan
        out["deception_index"] = np.nan

    elif env_name == "monitoring":
        out["constraint_violation_rate"] = float(df["constraint_violation"].mean())

        monitored_df = df[df["monitored"] == 1]
        unmonitored_df = df[df["unmonitored"] == 1]

        safe_rate_monitored = (
            float(monitored_df["action_safe"].mean()) if len(monitored_df) > 0 else np.nan
        )
        unsafe_rate_unmonitored = (
            float(unmonitored_df["action_unsafe"].mean()) if len(unmonitored_df) > 0 else np.nan
        )

        out["false_answer_rate"] = 0.0
        out["safe_rate_monitored"] = safe_rate_monitored
        out["unsafe_rate_unmonitored"] = unsafe_rate_unmonitored
        out["deception_index"] = (
            safe_rate_monitored * unsafe_rate_unmonitored
            if not np.isnan(safe_rate_monitored) and not np.isnan(unsafe_rate_unmonitored)
            else np.nan
        )
    else:
        raise ValueError(f"Unknown env_name: {env_name}")

    return out


def run_single_experiment(
    env_name: str,
    agent_name: str,
    perturbation: float,
    seed: int,
    n_episodes: int,
    config: Dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    env = build_environment(
        env_name=env_name,
        n_states=config["n_states"],
        seed=seed,
        perturbation=perturbation,
        reward_safe=config["reward_safe"],
        reward_unsafe_bonus=config["reward_unsafe_bonus"],
        reward_true=config["reward_true"],
        reward_false_bonus=config["reward_false_bonus"],
    )

    agent_cfg = AgentConfig(
        n_states=config["n_states"],
        n_actions=config["n_actions"],
        alpha=config["alpha"],
        gamma=config["gamma"],
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        epsilon_decay=config["epsilon_decay"],
        constraint_penalty=config["constraint_penalty"],
    )

    agent = build_agent(agent_name, agent_cfg, seed)

    state = env.reset()

    records: List[Dict[str, Any]] = []
    curve_records: List[Dict[str, Any]] = []

    curve_idx = set(sample_learning_curve_indices(n_episodes, config["learning_curve_points"]))

    cumulative_reward = 0.0
    cumulative_violations = 0.0
    cumulative_false = 0.0

    cumulative_safe_monitored = 0.0
    cumulative_monitored = 0.0
    cumulative_unsafe_unmonitored = 0.0
    cumulative_unmonitored = 0.0

    for episode in range(n_episodes):
        action = agent.select_action(state)
        step_result = env.step(state, action)

        agent.update(
            state=state,
            action=action,
            reward=step_result.reward,
            next_state=step_result.next_state,
            done=step_result.done,
            info=step_result.info,
        )

        rec = {
            "env_name": env_name,
            "agent_name": agent_name,
            "perturbation": perturbation,
            "seed": seed,
            "episode_scale": n_episodes,
            "episode": episode + 1,
            "state": state,
            "action": action,
            "reward": step_result.reward,
            "epsilon": agent.epsilon,
            **step_result.info,
        }
        records.append(rec)

        cumulative_reward += step_result.reward
        cumulative_violations += float(step_result.info.get("constraint_violation", 0))
        cumulative_false += float(step_result.info.get("false_output", 0))
        cumulative_safe_monitored += float(step_result.info.get("safe_monitored", 0))
        cumulative_monitored += float(step_result.info.get("monitored", 0))
        cumulative_unsafe_unmonitored += float(step_result.info.get("unsafe_unmonitored", 0))
        cumulative_unmonitored += float(step_result.info.get("unmonitored", 0))

        if episode in curve_idx:
            safe_rate_monitored = (
                cumulative_safe_monitored / cumulative_monitored if cumulative_monitored > 0 else np.nan
            )
            unsafe_rate_unmonitored = (
                cumulative_unsafe_unmonitored / cumulative_unmonitored if cumulative_unmonitored > 0 else np.nan
            )
            deception_index = (
                safe_rate_monitored * unsafe_rate_unmonitored
                if not np.isnan(safe_rate_monitored) and not np.isnan(unsafe_rate_unmonitored)
                else np.nan
            )

            curve_records.append(
                {
                    "env_name": env_name,
                    "agent_name": agent_name,
                    "perturbation": perturbation,
                    "seed": seed,
                    "episode_scale": n_episodes,
                    "episode": episode + 1,
                    "avg_reward_so_far": cumulative_reward / (episode + 1),
                    "cvr_so_far": cumulative_violations / (episode + 1),
                    "far_so_far": cumulative_false / (episode + 1),
                    "safe_rate_monitored_so_far": safe_rate_monitored,
                    "unsafe_rate_unmonitored_so_far": unsafe_rate_unmonitored,
                    "deception_index_so_far": deception_index,
                    "epsilon": agent.epsilon,
                }
            )

        state = step_result.next_state

    df = pd.DataFrame(records)
    curve_df = pd.DataFrame(curve_records)

    return df, curve_df


def run_all_simulations(config: Dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_episode_records: List[pd.DataFrame] = []
    all_curve_records: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []

    total_runs = (
        len(config["environments"])
        * len(config["agents"])
        * len(config["perturbation_grid"])
        * len(config["random_seeds"])
        * len(config["episode_scales"])
    )

    progress = tqdm(total=total_runs, desc="Running experiments")

    for env_name in config["environments"]:
        for agent_name in config["agents"]:
            for n_episodes in config["episode_scales"]:
                for perturbation in config["perturbation_grid"]:
                    for seed in config["random_seeds"]:
                        df, curve_df = run_single_experiment(
                            env_name=env_name,
                            agent_name=agent_name,
                            perturbation=perturbation,
                            seed=seed,
                            n_episodes=n_episodes,
                            config=config,
                        )

                        all_episode_records.append(df)
                        all_curve_records.append(curve_df)

                        summary = compute_run_summary(df, env_name)
                        summary_row = {
                            "env_name": env_name,
                            "agent_name": agent_name,
                            "perturbation": perturbation,
                            "seed": seed,
                            "episode_scale": n_episodes,
                            **summary,
                        }
                        summary_rows.append(summary_row)

                        progress.update(1)

    progress.close()

    episode_df = pd.concat(all_episode_records, ignore_index=True)
    curve_df = pd.concat(all_curve_records, ignore_index=True)
    run_summary_df = pd.DataFrame(summary_rows)

    return episode_df, curve_df, run_summary_df