"""Neural DQN experiments under controlled reward misspecification.

Implemented algorithms:
    reward_only_dqn
    fixed_penalty_dqn
    lagrangian_dqn
    hard_shield_dqn

The optimized proxy reward is separated from held-out true utility and
constraint-violation evaluators.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.envs import (
    ContextualCMDP,
    build_environment,
)


DQN_MODES = (
    "reward_only_dqn",
    "fixed_penalty_dqn",
    "lagrangian_dqn",
    "hard_shield_dqn",
)


@dataclass
class DQNConfig:
    hidden_dimensions: tuple[int, ...]
    learning_rate: float
    gamma: float
    batch_size: int
    replay_capacity: int
    replay_warmup: int
    target_update_interval: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    fixed_penalty: float
    cost_limit: float
    lambda_learning_rate: float
    lambda_maximum: float
    gradient_clip: float


class QNetwork(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        number_of_actions: int,
        hidden_dimensions: tuple[int, ...],
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []

        previous_dimension = input_dimension

        for hidden_dimension in hidden_dimensions:
            layers.append(
                nn.Linear(
                    previous_dimension,
                    hidden_dimension,
                )
            )

            layers.append(
                nn.ReLU()
            )

            previous_dimension = hidden_dimension

        layers.append(
            nn.Linear(
                previous_dimension,
                number_of_actions,
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(
            observations
        )


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_dimension: int,
        number_of_actions: int,
        seed: int,
    ) -> None:
        self.capacity = int(
            capacity
        )

        self.state_dimension = int(
            state_dimension
        )

        self.number_of_actions = int(
            number_of_actions
        )

        self.random_generator = (
            np.random.default_rng(seed)
        )

        self.states = np.zeros(
            (
                self.capacity,
                self.state_dimension,
            ),
            dtype=np.float32,
        )

        self.actions = np.zeros(
            self.capacity,
            dtype=np.int64,
        )

        self.proxy_rewards = np.zeros(
            self.capacity,
            dtype=np.float32,
        )

        self.costs = np.zeros(
            self.capacity,
            dtype=np.float32,
        )

        self.next_states = np.zeros(
            (
                self.capacity,
                self.state_dimension,
            ),
            dtype=np.float32,
        )

        self.next_action_masks = np.ones(
            (
                self.capacity,
                self.number_of_actions,
            ),
            dtype=np.float32,
        )

        self.current_size = 0
        self.next_index = 0

    def __len__(self) -> int:
        return self.current_size

    def add(
        self,
        state: np.ndarray,
        action: int,
        proxy_reward: float,
        cost: float,
        next_state: np.ndarray,
        next_action_mask: np.ndarray,
    ) -> None:
        index = self.next_index

        self.states[index] = state
        self.actions[index] = action
        self.proxy_rewards[index] = (
            proxy_reward
        )
        self.costs[index] = cost
        self.next_states[index] = (
            next_state
        )
        self.next_action_masks[index] = (
            next_action_mask
        )

        self.next_index = (
            self.next_index + 1
        ) % self.capacity

        self.current_size = min(
            self.current_size + 1,
            self.capacity,
        )

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        if self.current_size < batch_size:
            raise ValueError(
                "Replay buffer does not contain enough samples."
            )

        indices = self.random_generator.choice(
            self.current_size,
            size=batch_size,
            replace=False,
        )

        return {
            "states": torch.as_tensor(
                self.states[indices],
                dtype=torch.float32,
                device=device,
            ),
            "actions": torch.as_tensor(
                self.actions[indices],
                dtype=torch.long,
                device=device,
            ),
            "proxy_rewards": torch.as_tensor(
                self.proxy_rewards[indices],
                dtype=torch.float32,
                device=device,
            ),
            "costs": torch.as_tensor(
                self.costs[indices],
                dtype=torch.float32,
                device=device,
            ),
            "next_states": torch.as_tensor(
                self.next_states[indices],
                dtype=torch.float32,
                device=device,
            ),
            "next_action_masks": torch.as_tensor(
                self.next_action_masks[indices],
                dtype=torch.float32,
                device=device,
            ),
        }


def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except AttributeError:
            pass


def select_device(
    requested_device: str,
) -> torch.device:
    if requested_device == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")

        print(
            "MPS was requested but is unavailable. Falling back to CPU."
        )

        return torch.device("cpu")

    if requested_device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")

        print(
            "CUDA was requested but is unavailable. Falling back to CPU."
        )

        return torch.device("cpu")

    return torch.device("cpu")


def state_to_vector(
    state: int,
    number_of_states: int,
) -> np.ndarray:
    vector = np.zeros(
        number_of_states,
        dtype=np.float32,
    )

    vector[int(state)] = 1.0

    return vector


def action_mask(
    environment: ContextualCMDP,
    state: int,
    mode: str,
) -> np.ndarray:
    mask = np.ones(
        environment.n_actions,
        dtype=np.float32,
    )

    if mode != "hard_shield_dqn":
        return mask

    mask[:] = 0.0

    admissible_actions = (
        environment
        .predicted_admissible_actions(
            state
        )
    )

    if len(admissible_actions) == 0:
        raise RuntimeError(
            "Hard shield produced an empty admissible-action set."
        )

    for action in admissible_actions:
        mask[int(action)] = 1.0

    return mask


def epsilon_at_step(
    step: int,
    config: DQNConfig,
) -> float:
    fraction = min(
        1.0,
        step
        / max(
            1,
            config.epsilon_decay_steps,
        ),
    )

    epsilon = (
        config.epsilon_start
        + fraction
        * (
            config.epsilon_end
            - config.epsilon_start
        )
    )

    return float(
        epsilon
    )


class DQNAgent:
    def __init__(
        self,
        number_of_states: int,
        number_of_actions: int,
        mode: str,
        config: DQNConfig,
        seed: int,
        device: torch.device,
    ) -> None:
        if mode not in DQN_MODES:
            raise ValueError(
                f"Unknown DQN mode: {mode}"
            )

        self.number_of_states = int(
            number_of_states
        )

        self.number_of_actions = int(
            number_of_actions
        )

        self.mode = mode
        self.config = config
        self.device = device

        self.random_generator = (
            np.random.default_rng(seed)
        )

        self.online_network = QNetwork(
            input_dimension=
                self.number_of_states,
            number_of_actions=
                self.number_of_actions,
            hidden_dimensions=
                config.hidden_dimensions,
        ).to(device)

        self.target_network = QNetwork(
            input_dimension=
                self.number_of_states,
            number_of_actions=
                self.number_of_actions,
            hidden_dimensions=
                config.hidden_dimensions,
        ).to(device)

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.online_network.parameters(),
            lr=config.learning_rate,
        )

        self.loss_function = (
            nn.SmoothL1Loss()
        )

        self.lagrange_multiplier = 0.0
        self.optimization_steps = 0

    def select_action(
        self,
        state_vector: np.ndarray,
        mask: np.ndarray,
        epsilon: float,
    ) -> int:
        valid_actions = np.flatnonzero(
            mask > 0.5
        )

        if len(valid_actions) == 0:
            raise RuntimeError(
                "No valid actions are available."
            )

        if (
            epsilon > 0.0
            and self.random_generator.random()
            < epsilon
        ):
            return int(
                self.random_generator.choice(
                    valid_actions
                )
            )

        with torch.no_grad():
            state_tensor = torch.as_tensor(
                state_vector,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            q_values = (
                self.online_network(
                    state_tensor
                )
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )

        masked_q_values = np.full(
            self.number_of_actions,
            -np.inf,
            dtype=np.float32,
        )

        masked_q_values[
            valid_actions
        ] = q_values[
            valid_actions
        ]

        best_actions = np.flatnonzero(
            np.isclose(
                masked_q_values,
                np.max(masked_q_values),
            )
        )

        return int(
            self.random_generator.choice(
                best_actions
            )
        )

    def shaped_rewards(
        self,
        proxy_rewards: torch.Tensor,
        costs: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "fixed_penalty_dqn":
            return (
                proxy_rewards
                - self.config.fixed_penalty
                * costs
            )

        if self.mode == "lagrangian_dqn":
            return (
                proxy_rewards
                - self.lagrange_multiplier
                * costs
            )

        return proxy_rewards

    def update_lagrange_multiplier(
        self,
        observed_cost: float,
    ) -> None:
        if self.mode != "lagrangian_dqn":
            return

        update = (
            self.config.lambda_learning_rate
            * (
                observed_cost
                - self.config.cost_limit
            )
        )

        self.lagrange_multiplier = float(
            np.clip(
                self.lagrange_multiplier
                + update,
                0.0,
                self.config.lambda_maximum,
            )
        )

    def optimize(
        self,
        replay_buffer: ReplayBuffer,
    ) -> float:
        batch = replay_buffer.sample(
            batch_size=
                self.config.batch_size,
            device=
                self.device,
        )

        current_q_values = (
            self.online_network(
                batch["states"]
            )
            .gather(
                1,
                batch["actions"]
                .unsqueeze(1),
            )
            .squeeze(1)
        )

        with torch.no_grad():
            next_q_values = (
                self.target_network(
                    batch["next_states"]
                )
            )

            invalid_actions = (
                batch[
                    "next_action_masks"
                ]
                <= 0.5
            )

            next_q_values = (
                next_q_values.masked_fill(
                    invalid_actions,
                    -1e9,
                )
            )

            maximum_next_q = (
                next_q_values.max(
                    dim=1
                ).values
            )

            rewards = self.shaped_rewards(
                proxy_rewards=
                    batch["proxy_rewards"],
                costs=
                    batch["costs"],
            )

            targets = (
                rewards
                + self.config.gamma
                * maximum_next_q
            )

        loss = self.loss_function(
            current_q_values,
            targets,
        )

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            self.config.gradient_clip,
        )

        self.optimizer.step()

        self.optimization_steps += 1

        if (
            self.optimization_steps
            % self.config.target_update_interval
            == 0
        ):
            self.target_network.load_state_dict(
                self.online_network.state_dict()
            )

        return float(
            loss.detach().cpu().item()
        )


def build_dqn_config(
    configuration: dict[str, Any],
) -> DQNConfig:
    return DQNConfig(
        hidden_dimensions=tuple(
            int(value)
            for value in configuration[
                "hidden_dimensions"
            ]
        ),
        learning_rate=float(
            configuration[
                "learning_rate"
            ]
        ),
        gamma=float(
            configuration["gamma"]
        ),
        batch_size=int(
            configuration["batch_size"]
        ),
        replay_capacity=int(
            configuration[
                "replay_capacity"
            ]
        ),
        replay_warmup=int(
            configuration[
                "replay_warmup"
            ]
        ),
        target_update_interval=int(
            configuration[
                "target_update_interval"
            ]
        ),
        epsilon_start=float(
            configuration[
                "epsilon_start"
            ]
        ),
        epsilon_end=float(
            configuration[
                "epsilon_end"
            ]
        ),
        epsilon_decay_steps=int(
            configuration[
                "epsilon_decay_steps"
            ]
        ),
        fixed_penalty=float(
            configuration[
                "fixed_penalty"
            ]
        ),
        cost_limit=float(
            configuration[
                "cost_limit"
            ]
        ),
        lambda_learning_rate=float(
            configuration[
                "lambda_learning_rate"
            ]
        ),
        lambda_maximum=float(
            configuration[
                "lambda_maximum"
            ]
        ),
        gradient_clip=float(
            configuration[
                "gradient_clip"
            ]
        ),
    )


def evaluate_agent(
    agent: DQNAgent,
    environment: ContextualCMDP,
    mode: str,
    evaluation_steps: int,
) -> dict[str, float]:
    state = environment.reset()

    proxy_rewards = []
    true_utilities = []
    violations = []
    correctness = []
    monitoring_flags = []

    for _ in range(evaluation_steps):
        state_vector = state_to_vector(
            state,
            environment.n_states,
        )

        mask = action_mask(
            environment,
            state,
            mode,
        )

        action = agent.select_action(
            state_vector=state_vector,
            mask=mask,
            epsilon=0.0,
        )

        transition = environment.step(
            state,
            action,
        )

        proxy_rewards.append(
            transition.proxy_reward
        )

        true_utilities.append(
            transition.true_utility
        )

        violations.append(
            transition.violation
        )

        correctness.append(
            transition.correct
        )

        monitoring_flags.append(
            transition.monitored
        )

        state = transition.next_state

    proxy_rewards_array = np.asarray(
        proxy_rewards,
        dtype=float,
    )

    true_utilities_array = np.asarray(
        true_utilities,
        dtype=float,
    )

    violations_array = np.asarray(
        violations,
        dtype=float,
    )

    correctness_array = np.asarray(
        correctness,
        dtype=float,
    )

    monitoring_array = np.asarray(
        monitoring_flags,
        dtype=int,
    )

    monitored_mask = (
        monitoring_array == 1
    )

    unmonitored_mask = (
        monitoring_array == 0
    )

    unsafe_monitored = (
        float(
            violations_array[
                monitored_mask
            ].mean()
        )
        if monitored_mask.any()
        else np.nan
    )

    unsafe_unmonitored = (
        float(
            violations_array[
                unmonitored_mask
            ].mean()
        )
        if unmonitored_mask.any()
        else np.nan
    )

    monitoring_gap = (
        unsafe_unmonitored
        - unsafe_monitored
        if (
            np.isfinite(
                unsafe_unmonitored
            )
            and np.isfinite(
                unsafe_monitored
            )
        )
        else np.nan
    )

    mean_proxy_reward = float(
        proxy_rewards_array.mean()
    )

    mean_true_utility = float(
        true_utilities_array.mean()
    )

    return {
        "proxy_return":
            mean_proxy_reward,
        "true_utility":
            mean_true_utility,
        "policy_violation_rate":
            float(
                violations_array.mean()
            ),
        "false_answer_rate":
            float(
                1.0
                - correctness_array.mean()
            ),
        "unsafe_rate_monitored":
            unsafe_monitored,
        "unsafe_rate_unmonitored":
            unsafe_unmonitored,
        "monitoring_gap":
            monitoring_gap,
        "proxy_true_gap":
            mean_proxy_reward
            - mean_true_utility,
    }


def run_single_dqn(
    environment_name: str,
    mode: str,
    corruption_probability: float,
    corruption_magnitude: float,
    constraint_error: float,
    seed: int,
    training_steps: int,
    evaluation_steps: int,
    configuration: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    set_seed(seed)

    training_environment = build_environment(
        name=environment_name,
        seed=seed * 100 + 1,
        corruption_probability=
            corruption_probability,
        corruption_magnitude=
            corruption_magnitude,
        constraint_error=
            constraint_error,
    )

    evaluation_environment = build_environment(
        name=environment_name,
        seed=seed * 100 + 2,
        corruption_probability=
            corruption_probability,
        corruption_magnitude=
            corruption_magnitude,
        constraint_error=
            constraint_error,
    )

    dqn_config = build_dqn_config(
        configuration
    )

    agent = DQNAgent(
        number_of_states=
            training_environment.n_states,
        number_of_actions=
            training_environment.n_actions,
        mode=mode,
        config=dqn_config,
        seed=seed * 100 + 3,
        device=device,
    )

    replay_buffer = ReplayBuffer(
        capacity=
            dqn_config.replay_capacity,
        state_dimension=
            training_environment.n_states,
        number_of_actions=
            training_environment.n_actions,
        seed=seed * 100 + 4,
    )

    state = training_environment.reset()

    curve_records = []

    cumulative_proxy_reward = 0.0
    cumulative_true_utility = 0.0
    cumulative_cost = 0.0
    latest_loss = np.nan

    record_interval = max(
        1,
        training_steps
        // int(
            configuration[
                "learning_curve_points"
            ]
        ),
    )

    for step in range(
        1,
        training_steps + 1,
    ):
        state_vector = state_to_vector(
            state,
            training_environment.n_states,
        )

        current_mask = action_mask(
            training_environment,
            state,
            mode,
        )

        epsilon = epsilon_at_step(
            step,
            dqn_config,
        )

        action = agent.select_action(
            state_vector=state_vector,
            mask=current_mask,
            epsilon=epsilon,
        )

        transition = training_environment.step(
            state,
            action,
        )

        next_state_vector = state_to_vector(
            transition.next_state,
            training_environment.n_states,
        )

        next_mask = action_mask(
            training_environment,
            transition.next_state,
            mode,
        )

        replay_buffer.add(
            state=state_vector,
            action=action,
            proxy_reward=
                transition.proxy_reward,
            cost=
                transition.violation,
            next_state=
                next_state_vector,
            next_action_mask=
                next_mask,
        )

        agent.update_lagrange_multiplier(
            observed_cost=
                transition.violation
        )

        if (
            len(replay_buffer)
            >= max(
                dqn_config.replay_warmup,
                dqn_config.batch_size,
            )
        ):
            latest_loss = agent.optimize(
                replay_buffer
            )

        cumulative_proxy_reward += (
            transition.proxy_reward
        )

        cumulative_true_utility += (
            transition.true_utility
        )

        cumulative_cost += (
            transition.violation
        )

        if (
            step == 1
            or step % record_interval == 0
            or step == training_steps
        ):
            curve_records.append(
                {
                    "environment":
                        environment_name,
                    "agent":
                        mode,
                    "seed":
                        seed,
                    "corruption_probability":
                        corruption_probability,
                    "corruption_magnitude":
                        corruption_magnitude,
                    "constraint_error":
                        constraint_error,
                    "training_step":
                        step,
                    "training_proxy_return":
                        cumulative_proxy_reward
                        / step,
                    "training_true_utility":
                        cumulative_true_utility
                        / step,
                    "training_violation_rate":
                        cumulative_cost
                        / step,
                    "epsilon":
                        epsilon,
                    "loss":
                        latest_loss,
                    "lagrange_multiplier":
                        agent
                        .lagrange_multiplier,
                }
            )

        state = transition.next_state

    evaluation_summary = evaluate_agent(
        agent=agent,
        environment=
            evaluation_environment,
        mode=mode,
        evaluation_steps=
            evaluation_steps,
    )

    summary = {
        "environment":
            environment_name,
        "agent":
            mode,
        "seed":
            seed,
        "corruption_probability":
            corruption_probability,
        "corruption_magnitude":
            corruption_magnitude,
        "constraint_error":
            constraint_error,
        "training_steps":
            training_steps,
        "evaluation_steps":
            evaluation_steps,
        "device":
            str(device),
        "final_lagrange_multiplier":
            agent.lagrange_multiplier,
        **evaluation_summary,
    }

    return (
        summary,
        pd.DataFrame(
            curve_records
        ),
    )


def run_all_dqn(
    configuration: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = select_device(
        str(
            configuration.get(
                "device",
                "cpu",
            )
        )
    )

    print(
        f"Selected neural device: {device}"
    )

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
        in configuration["environments"]
        for agent
        in configuration["agents"]
        for probability
        in configuration[
            "corruption_probabilities"
        ]
        for magnitude
        in configuration[
            "corruption_magnitudes"
        ]
        for constraint_error
        in configuration[
            "constraint_errors"
        ]
        for seed
        in configuration["random_seeds"]
    ]

    summaries = []
    curves = []

    total_runs = len(
        combinations
    )

    reporting_interval = max(
        1,
        total_runs // 20,
    )

    for run_number, values in enumerate(
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
        ) = values

        summary, learning_curve = (
            run_single_dqn(
                environment_name=
                    environment,
                mode=
                    agent,
                corruption_probability=
                    probability,
                corruption_magnitude=
                    magnitude,
                constraint_error=
                    constraint_error,
                seed=
                    seed,
                training_steps=
                    int(
                        configuration[
                            "training_steps"
                        ]
                    ),
                evaluation_steps=
                    int(
                        configuration[
                            "evaluation_steps"
                        ]
                    ),
                configuration=
                    configuration,
                device=
                    device,
            )
        )

        summaries.append(
            summary
        )

        curves.append(
            learning_curve
        )

        if (
            run_number
            % reporting_interval
            == 0
            or run_number == total_runs
        ):
            print(
                (
                    f"Completed neural run "
                    f"{run_number}/{total_runs}"
                ),
                flush=True,
            )

    return (
        pd.DataFrame(summaries),
        pd.concat(
            curves,
            ignore_index=True,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    arguments = parser.parse_args()

    config_path = Path(
        arguments.config
    )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        configuration = yaml.safe_load(
            config_file
        )

    output_directory = Path(
        configuration["output_dir"]
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary, curves = run_all_dqn(
        configuration
    )

    summary.to_csv(
        output_directory
        / "dqn_run_summary.csv",
        index=False,
    )

    curves.to_csv(
        output_directory
        / "dqn_learning_curves.csv",
        index=False,
    )

    manifest = {
        "configuration":
            configuration,
        "python":
            platform.python_version(),
        "numpy":
            np.__version__,
        "pandas":
            pd.__version__,
        "torch":
            torch.__version__,
    }

    (
        output_directory
        / "dqn_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved neural results to {output_directory}"
    )


if __name__ == "__main__":
    main()