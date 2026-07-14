"""Tabular baselines for reward and constraint enforcement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AgentConfig:
    n_states: int
    n_actions: int = 2
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.9995
    fixed_penalty: float = 2.0
    cost_limit: float = 0.02
    lambda_lr: float = 0.05
    lambda_max: float = 100.0


class QLearningAgent:
    def __init__(
        self,
        config: AgentConfig,
        seed: int,
        mode: str,
    ) -> None:
        self.config = config
        self.mode = mode
        self.rng = np.random.default_rng(seed)

        self.q_values = np.zeros(
            (config.n_states, config.n_actions),
            dtype=float,
        )

        self.epsilon = config.epsilon_start
        self.lagrange_multiplier = 0.0

    def candidate_actions(
        self,
        admissible_actions: list[int] | None,
    ) -> list[int]:
        if self.mode == "hard_shield":
            if not admissible_actions:
                raise RuntimeError(
                    "Hard shield received an empty admissible action set."
                )

            return admissible_actions

        return list(range(self.config.n_actions))

    def select_action(
        self,
        state: int,
        admissible_actions: list[int] | None = None,
        explore: bool = True,
    ) -> int:
        candidates = self.candidate_actions(
            admissible_actions
        )

        if (
            explore
            and self.rng.random() < self.epsilon
        ):
            return int(
                self.rng.choice(candidates)
            )

        candidate_values = self.q_values[
            state,
            candidates,
        ]

        maximum_value = np.max(
            candidate_values
        )

        best_indices = np.flatnonzero(
            np.isclose(
                candidate_values,
                maximum_value,
            )
        )

        selected_index = int(
            self.rng.choice(best_indices)
        )

        return int(candidates[selected_index])

    def update(
        self,
        state: int,
        action: int,
        proxy_reward: float,
        cost: float,
        next_state: int,
        next_admissible_actions: list[int] | None = None,
    ) -> None:
        shaped_reward = proxy_reward

        if self.mode == "fixed_penalty":
            shaped_reward -= (
                self.config.fixed_penalty
                * cost
            )

        elif self.mode == "lagrangian":
            shaped_reward -= (
                self.lagrange_multiplier
                * cost
            )

            multiplier_update = (
                self.config.lambda_lr
                * (
                    cost
                    - self.config.cost_limit
                )
            )

            self.lagrange_multiplier = float(
                np.clip(
                    self.lagrange_multiplier
                    + multiplier_update,
                    0.0,
                    self.config.lambda_max,
                )
            )

        candidates = self.candidate_actions(
            next_admissible_actions
        )

        next_value = np.max(
            self.q_values[
                next_state,
                candidates,
            ]
        )

        target = (
            shaped_reward
            + self.config.gamma * next_value
        )

        temporal_difference_error = (
            target
            - self.q_values[state, action]
        )

        self.q_values[state, action] += (
            self.config.alpha
            * temporal_difference_error
        )

        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon
            * self.config.epsilon_decay,
        )


AGENT_MODES = (
    "reward_only",
    "fixed_penalty",
    "lagrangian",
    "hard_shield",
)


def build_agent(
    name: str,
    config: AgentConfig,
    seed: int,
) -> QLearningAgent:
    if name not in AGENT_MODES:
        raise ValueError(
            f"Unknown agent: {name}"
        )

    return QLearningAgent(
        config=config,
        seed=seed,
        mode=name,
    )