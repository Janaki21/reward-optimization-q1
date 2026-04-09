from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AgentConfig:
    n_states: int
    n_actions: int
    alpha: float
    gamma: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float
    constraint_penalty: float


class BaseQLearningAgent:
    def __init__(self, cfg: AgentConfig, seed: int) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((cfg.n_states, cfg.n_actions), dtype=float)
        self.epsilon = cfg.epsilon_start

    def valid_actions(self, state: int) -> list[int]:
        return list(range(self.cfg.n_actions))

    def shaped_q_values(self, state: int) -> np.ndarray:
        return self.q[state].copy()

    def select_action(self, state: int) -> int:
        valid = self.valid_actions(state)
        if not valid:
            raise RuntimeError("No valid actions available.")

        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid))

        qvals = self.shaped_q_values(state)
        masked = np.full_like(qvals, fill_value=-1e18, dtype=float)
        for a in valid:
            masked[a] = qvals[a]
        best_actions = np.flatnonzero(masked == np.max(masked))
        return int(self.rng.choice(best_actions))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        info: Optional[dict] = None,
    ) -> None:
        target = reward
        if not done:
            target += self.cfg.gamma * np.max(self.q[next_state])

        td_error = target - self.q[state, action]
        self.q[state, action] += self.cfg.alpha * td_error

        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)


class RewardQLearningAgent(BaseQLearningAgent):
    pass


class ConstrainedQLearningAgent(BaseQLearningAgent):
    """
    Hard veto:
    action 1 is never allowed.
    """

    def valid_actions(self, state: int) -> list[int]:
        return [0]


class SoftPenaltyQLearningAgent(BaseQLearningAgent):
    """
    Soft ablation:
    action 1 allowed, but discouraged by penalty.
    """

    def shaped_q_values(self, state: int) -> np.ndarray:
        qvals = self.q[state].copy()
        qvals[1] -= self.cfg.constraint_penalty
        return qvals

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        info: Optional[dict] = None,
    ) -> None:
        shaped_reward = reward
        if action == 1:
            shaped_reward -= self.cfg.constraint_penalty
        super().update(state, action, shaped_reward, next_state, done, info)


def build_agent(agent_name: str, cfg: AgentConfig, seed: int) -> BaseQLearningAgent:
    if agent_name == "reward":
        return RewardQLearningAgent(cfg, seed)
    if agent_name == "soft":
        return SoftPenaltyQLearningAgent(cfg, seed)
    if agent_name == "constrained":
        return ConstrainedQLearningAgent(cfg, seed)
    raise ValueError(f"Unknown agent: {agent_name}")