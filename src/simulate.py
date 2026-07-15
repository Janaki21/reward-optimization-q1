"""Training and frozen-policy evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.agents import (
    AgentConfig,
    build_agent,
)
from src.envs import build_environment


def create_agent_config(
    environment,
    config: dict[str, Any],
) -> AgentConfig:
    return AgentConfig(
        n_states=environment.n_states,
        n_actions=environment.n_actions,
        alpha=config["alpha"],
        gamma=config["gamma"],
        epsilon_start=
            config["epsilon_start"],
        epsilon_end=
            config["epsilon_end"],
        epsilon_decay=
            config["epsilon_decay"],
        fixed_penalty=
            config["fixed_penalty"],
        cost_limit=
            config["cost_limit"],
        lambda_lr=
            config["lambda_lr"],
        lambda_max=
            config["lambda_max"],
    )


def get_admissible_actions(
    environment,
    agent_name: str,
    state: int,
) -> list[int] | None:
    """
    Only the hard-shield agent can access the predicted admissible set.

    Reward-only, fixed-penalty and Lagrangian agents must not query the
    constraint model. This prevents constraint error from affecting their
    random trajectories.
    """

    if agent_name != "hard_shield":
        return None

    return (
        environment
        .predicted_admissible_actions(
            state
        )
    )


def summarize_evaluation(
    records: list[dict[str, Any]],
) -> dict[str, float]:
    dataframe = pd.DataFrame(
        records
    )

    monitored_data = dataframe[
        dataframe["monitored"] == 1
    ]

    unmonitored_data = dataframe[
        dataframe["monitored"] == 0
    ]

    if len(monitored_data) > 0:
        unsafe_rate_monitored = float(
            monitored_data[
                "violation"
            ].mean()
        )
    else:
        unsafe_rate_monitored = np.nan

    if len(unmonitored_data) > 0:
        unsafe_rate_unmonitored = float(
            unmonitored_data[
                "violation"
            ].mean()
        )
    else:
        unsafe_rate_unmonitored = np.nan

    if (
        np.isfinite(
            unsafe_rate_monitored
        )
        and np.isfinite(
            unsafe_rate_unmonitored
        )
    ):
        monitoring_gap = (
            unsafe_rate_unmonitored
            - unsafe_rate_monitored
        )
    else:
        monitoring_gap = np.nan

    proxy_return = float(
        dataframe[
            "proxy_reward"
        ].mean()
    )

    true_utility = float(
        dataframe[
            "true_utility"
        ].mean()
    )

    policy_violation_rate = float(
        dataframe[
            "violation"
        ].mean()
    )

    false_answer_rate = float(
        1.0
        - dataframe[
            "correct"
        ].mean()
    )

    return {
        "proxy_return":
            proxy_return,
        "true_utility":
            true_utility,
        "policy_violation_rate":
            policy_violation_rate,
        "false_answer_rate":
            false_answer_rate,
        "unsafe_rate_monitored":
            unsafe_rate_monitored,
        "unsafe_rate_unmonitored":
            unsafe_rate_unmonitored,
        "monitoring_gap":
            monitoring_gap,
        "proxy_true_gap":
            proxy_return
            - true_utility,
    }


def run_single_experiment(
    environment_name: str,
    agent_name: str,
    corruption_probability: float,
    corruption_magnitude: float,
    constraint_error: float,
    seed: int,
    train_steps: int,
    eval_steps: int,
    config: dict[str, Any],
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
]:
    """
    Train one policy and evaluate it on an independent random stream.
    """

    training_environment = (
        build_environment(
            name=environment_name,
            seed=seed * 10 + 1,
            corruption_probability=
                corruption_probability,
            corruption_magnitude=
                corruption_magnitude,
            constraint_error=
                constraint_error,
        )
    )

    evaluation_environment = (
        build_environment(
            name=environment_name,
            seed=seed * 10 + 2,
            corruption_probability=
                corruption_probability,
            corruption_magnitude=
                corruption_magnitude,
            constraint_error=
                constraint_error,
        )
    )

    agent_config = (
        create_agent_config(
            training_environment,
            config,
        )
    )

    agent = build_agent(
        name=agent_name,
        config=agent_config,
        seed=seed * 10 + 3,
    )

    state = (
        training_environment.reset()
    )

    learning_curve_records = []

    learning_curve_points = int(
        config["learning_curve_points"]
    )

    record_interval = max(
        1,
        train_steps
        // learning_curve_points,
    )

    cumulative_proxy_reward = 0.0
    cumulative_true_utility = 0.0
    cumulative_violations = 0.0

    for step in range(
        1,
        train_steps + 1,
    ):
        admissible_actions = (
            get_admissible_actions(
                environment=
                    training_environment,
                agent_name=
                    agent_name,
                state=
                    state,
            )
        )

        action = agent.select_action(
            state=state,
            admissible_actions=
                admissible_actions,
            explore=True,
        )

        transition = (
            training_environment.step(
                state,
                action,
            )
        )

        next_admissible_actions = (
            get_admissible_actions(
                environment=
                    training_environment,
                agent_name=
                    agent_name,
                state=
                    transition.next_state,
            )
        )

        agent.update(
            state=state,
            action=action,
            proxy_reward=
                transition.proxy_reward,
            cost=
                transition.violation,
            next_state=
                transition.next_state,
            next_admissible_actions=
                next_admissible_actions,
        )

        cumulative_proxy_reward += (
            transition.proxy_reward
        )

        cumulative_true_utility += (
            transition.true_utility
        )

        cumulative_violations += (
            transition.violation
        )

        should_record = (
            step == 1
            or step % record_interval == 0
            or step == train_steps
        )

        if should_record:
            learning_curve_records.append(
                {
                    "environment":
                        environment_name,
                    "agent":
                        agent_name,
                    "seed":
                        seed,
                    "corruption_probability":
                        corruption_probability,
                    "corruption_magnitude":
                        corruption_magnitude,
                    "constraint_error":
                        constraint_error,
                    "train_step":
                        step,
                    "training_proxy_return":
                        cumulative_proxy_reward
                        / step,
                    "training_true_utility":
                        cumulative_true_utility
                        / step,
                    "training_violation_rate":
                        cumulative_violations
                        / step,
                    "epsilon":
                        agent.epsilon,
                    "lagrange_multiplier":
                        agent
                        .lagrange_multiplier,
                }
            )

        state = transition.next_state

    evaluation_records = []

    state = (
        evaluation_environment.reset()
    )

    for _ in range(eval_steps):
        admissible_actions = (
            get_admissible_actions(
                environment=
                    evaluation_environment,
                agent_name=
                    agent_name,
                state=
                    state,
            )
        )

        action = agent.select_action(
            state=state,
            admissible_actions=
                admissible_actions,
            explore=False,
        )

        transition = (
            evaluation_environment.step(
                state,
                action,
            )
        )

        evaluation_records.append(
            {
                "proxy_reward":
                    transition.proxy_reward,
                "true_utility":
                    transition.true_utility,
                "violation":
                    transition.violation,
                "correct":
                    transition.correct,
                "monitored":
                    transition.monitored,
            }
        )

        state = transition.next_state

    summary = {
        "environment":
            environment_name,
        "agent":
            agent_name,
        "seed":
            seed,
        "corruption_probability":
            corruption_probability,
        "corruption_magnitude":
            corruption_magnitude,
        "constraint_error":
            constraint_error,
        "train_steps":
            train_steps,
        "eval_steps":
            eval_steps,
        "final_lagrange_multiplier":
            agent.lagrange_multiplier,
        **summarize_evaluation(
            evaluation_records
        ),
    }

    return (
        summary,
        pd.DataFrame(
            learning_curve_records
        ),
    )


def run_all_simulations(
    config: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    combinations = [
        (
            environment,
            agent,
            probability,
            magnitude,
            constraint_error,
            seed,
        )
        for environment
        in config["environments"]
        for agent
        in config["agents"]
        for probability
        in config[
            "corruption_probabilities"
        ]
        for magnitude
        in config[
            "corruption_magnitudes"
        ]
        for constraint_error
        in config[
            "constraint_errors"
        ]
        for seed
        in config["random_seeds"]
    ]

    summary_records = []
    curve_dataframes = []

    total_runs = len(
        combinations
    )

    reporting_interval = max(
        1,
        total_runs // 20,
    )

    for run_number, combination in enumerate(
        combinations,
        start=1,
    ):
        (
            environment,
            agent,
            probability,
            magnitude,
            constraint_error,
            seed,
        ) = combination

        summary, curves = (
            run_single_experiment(
                environment_name=
                    environment,
                agent_name=
                    agent,
                corruption_probability=
                    probability,
                corruption_magnitude=
                    magnitude,
                constraint_error=
                    constraint_error,
                seed=
                    seed,
                train_steps=
                    config[
                        "train_steps"
                    ],
                eval_steps=
                    config[
                        "eval_steps"
                    ],
                config=
                    config,
            )
        )

        summary_records.append(
            summary
        )

        curve_dataframes.append(
            curves
        )

        if (
            run_number
            % reporting_interval
            == 0
            or run_number
            == total_runs
        ):
            print(
                (
                    f"Completed "
                    f"{run_number}/"
                    f"{total_runs} runs"
                ),
                flush=True,
            )

    summary_dataframe = (
        pd.DataFrame(
            summary_records
        )
    )

    learning_curves = pd.concat(
        curve_dataframes,
        ignore_index=True,
    )

    return (
        summary_dataframe,
        learning_curves,
    )