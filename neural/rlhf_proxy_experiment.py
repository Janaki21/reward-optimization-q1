"""Controlled RLHF-style reward-model perturbation experiment.

A neural reward model is trained from pairwise preferences. Preference scores
are progressively perturbed toward agreement, sycophancy, and unmonitored
deception. A neural response-selection policy then optimizes that learned
reward model.

This is a controlled mechanistic analogue of RLHF response selection, not a
claim about a production-scale language model.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml


METHODS = (
    "reward_only",
    "fixed_penalty",
    "lagrangian",
    "hard_veto",
)

CONTEXT_DIMENSION = 4
RESPONSE_DIMENSION = 6
NUMBER_OF_RESPONSES = 4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_contexts() -> torch.Tensor:
    """Generate all binary combinations of four prompt properties."""

    contexts = []

    for high_stakes in [0.0, 1.0]:
        for false_user_belief in [0.0, 1.0]:
            for monitored in [0.0, 1.0]:
                for adversarial in [0.0, 1.0]:
                    contexts.append(
                        [
                            high_stakes,
                            false_user_belief,
                            monitored,
                            adversarial,
                        ]
                    )

    return torch.tensor(
        contexts,
        dtype=torch.float32,
    )


def response_features(
    contexts: torch.Tensor,
) -> torch.Tensor:
    """Create response properties for every context and candidate response.

    Response features:
        truthfulness
        helpfulness
        agreement
        unsafe content
        deception
        refusal
    """

    rows = []

    for context in contexts:
        high_stakes = float(
            context[0].item()
        )
        false_belief = float(
            context[1].item()
        )
        adversarial = float(
            context[3].item()
        )

        truthful_agreement = (
            0.0
            if false_belief
            else 1.0
        )

        # Candidate 0: truthful and directly helpful.
        truthful = [
            1.0,
            1.0,
            truthful_agreement,
            0.0,
            0.0,
            0.0,
        ]

        # Candidate 1: agrees with the user, even when the user is wrong.
        sycophantic_truth = (
            0.0
            if false_belief
            else 1.0
        )

        sycophantic = [
            sycophantic_truth,
            1.0,
            1.0,
            high_stakes
            * (
                1.0
                - sycophantic_truth
            ),
            1.0
            - sycophantic_truth,
            0.0,
        ]

        # Candidate 2: safe refusal, but less useful.
        refusal = [
            1.0,
            0.2,
            0.0,
            0.0,
            0.0,
            1.0,
        ]

        # Candidate 3: deceptive reward-hacking response.
        exploit = [
            0.0,
            1.0,
            1.0,
            max(
                high_stakes,
                adversarial,
            ),
            1.0,
            0.0,
        ]

        rows.append(
            [
                truthful,
                sycophantic,
                refusal,
                exploit,
            ]
        )

    return torch.tensor(
        rows,
        dtype=torch.float32,
    )


def true_utility(
    features: torch.Tensor,
) -> torch.Tensor:
    truthfulness = features[..., 0]
    helpfulness = features[..., 1]
    refusal = features[..., 5]

    utility = (
        0.7 * truthfulness
        + 0.3 * helpfulness
        - 0.1 * refusal
    )

    return torch.clamp(
        utility,
        0.0,
        1.0,
    )


def violation_indicator(
    features: torch.Tensor,
) -> torch.Tensor:
    false_response = (
        features[..., 0]
        < 0.5
    )

    unsafe = (
        features[..., 3]
        > 0.0
    )

    deceptive = (
        features[..., 4]
        > 0.0
    )

    return (
        false_response
        | unsafe
        | deceptive
    ).to(
        dtype=torch.float32
    )


def perturbed_preference_score(
    contexts: torch.Tensor,
    features: torch.Tensor,
    perturbation: float,
) -> torch.Tensor:
    """Latent preference score used to generate comparison labels."""

    if not 0.0 <= perturbation <= 1.0:
        raise ValueError(
            "perturbation must be in [0, 1]."
        )

    base = (
        2.5 * features[..., 0]
        + 1.0 * features[..., 1]
        - 3.0 * features[..., 3]
        - 4.0 * features[..., 4]
        - 0.2 * features[..., 5]
    )

    agreement_bonus = (
        4.0
        * features[..., 2]
    )

    monitored = (
        contexts[:, 2]
        .unsqueeze(1)
    )

    unmonitored_deception_bonus = (
        5.0
        * (
            1.0 - monitored
        )
        * features[..., 4]
    )

    adversarial = (
        contexts[:, 3]
        .unsqueeze(1)
    )

    adversarial_exploit_bonus = (
        3.0
        * adversarial
        * features[..., 4]
    )

    corrupted_component = (
        agreement_bonus
        + unmonitored_deception_bonus
        + adversarial_exploit_bonus
    )

    return (
        (
            1.0 - perturbation
        )
        * base
        + perturbation
        * corrupted_component
    )


class RewardModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                CONTEXT_DIMENSION
                + RESPONSE_DIMENSION,
                32,
            ),
            nn.Tanh(),
            nn.Linear(
                32,
                32,
            ),
            nn.Tanh(),
            nn.Linear(
                32,
                1,
            ),
        )

    def forward(
        self,
        contexts: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat(
            [
                contexts,
                features,
            ],
            dim=-1,
        )

        return (
            self.network(
                inputs
            )
            .squeeze(-1)
        )


class ResponsePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                CONTEXT_DIMENSION,
                32,
            ),
            nn.Tanh(),
            nn.Linear(
                32,
                32,
            ),
            nn.Tanh(),
            nn.Linear(
                32,
                NUMBER_OF_RESPONSES,
            ),
        )

    def forward(
        self,
        contexts: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(
            contexts
        )


def pairwise_training_data(
    contexts: torch.Tensor,
    features: torch.Tensor,
    preference_scores: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    context_rows = []
    preferred_rows = []
    rejected_rows = []
    labels = []

    for context_index in range(
        contexts.shape[0]
    ):
        for first in range(
            NUMBER_OF_RESPONSES
        ):
            for second in range(
                first + 1,
                NUMBER_OF_RESPONSES,
            ):
                first_score = float(
                    preference_scores[
                        context_index,
                        first,
                    ].item()
                )

                second_score = float(
                    preference_scores[
                        context_index,
                        second,
                    ].item()
                )

                context_rows.append(
                    contexts[
                        context_index
                    ]
                )

                preferred_rows.append(
                    features[
                        context_index,
                        first,
                    ]
                )

                rejected_rows.append(
                    features[
                        context_index,
                        second,
                    ]
                )

                labels.append(
                    float(
                        first_score
                        > second_score
                    )
                )

    return (
        torch.stack(
            context_rows
        ),
        torch.stack(
            preferred_rows
        ),
        torch.stack(
            rejected_rows
        ),
        torch.tensor(
            labels,
            dtype=torch.float32,
        ),
        preference_scores,
    )


def train_reward_model(
    contexts: torch.Tensor,
    features: torch.Tensor,
    perturbation: float,
    seed: int,
    epochs: int,
    learning_rate: float,
) -> RewardModel:
    set_seed(seed)

    model = RewardModel()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    latent_scores = (
        perturbed_preference_score(
            contexts,
            features,
            perturbation,
        )
    )

    (
        pair_contexts,
        first_features,
        second_features,
        labels,
        _,
    ) = pairwise_training_data(
        contexts,
        features,
        latent_scores,
    )

    loss_function = (
        nn.BCEWithLogitsLoss()
    )

    model.train()

    for _ in range(epochs):
        first_scores = model(
            pair_contexts,
            first_features,
        )

        second_scores = model(
            pair_contexts,
            second_features,
        )

        logits = (
            first_scores
            - second_scores
        )

        loss = loss_function(
            logits,
            labels,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()

    return model


def reward_model_scores(
    model: RewardModel,
    contexts: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    number_of_contexts = (
        contexts.shape[0]
    )

    expanded_contexts = (
        contexts.unsqueeze(1)
        .expand(
            -1,
            NUMBER_OF_RESPONSES,
            -1,
        )
    )

    with torch.no_grad():
        scores = model(
            expanded_contexts.reshape(
                -1,
                CONTEXT_DIMENSION,
            ),
            features.reshape(
                -1,
                RESPONSE_DIMENSION,
            ),
        )

    return scores.reshape(
        number_of_contexts,
        NUMBER_OF_RESPONSES,
    )


def train_policy(
    contexts: torch.Tensor,
    features: torch.Tensor,
    learned_rewards: torch.Tensor,
    method: str,
    seed: int,
    epochs: int,
    learning_rate: float,
    fixed_penalty: float,
    cost_limit: float,
    lambda_learning_rate: float,
) -> tuple[
    ResponsePolicy,
    float,
]:
    if method not in METHODS:
        raise ValueError(
            f"Unknown method: {method}"
        )

    set_seed(
        seed + 100_003
    )

    policy = ResponsePolicy()

    optimizer = optim.Adam(
        policy.parameters(),
        lr=learning_rate,
    )

    costs = violation_indicator(
        features
    )

    lagrange_multiplier = 0.0

    for _ in range(epochs):
        logits = policy(
            contexts
        )

        if method == "hard_veto":
            logits = logits.masked_fill(
                costs > 0.5,
                -1e9,
            )

        probabilities = (
            torch.softmax(
                logits,
                dim=-1,
            )
        )

        if method == "fixed_penalty":
            optimized_scores = (
                learned_rewards
                - fixed_penalty
                * costs
            )

        elif method == "lagrangian":
            optimized_scores = (
                learned_rewards
                - lagrange_multiplier
                * costs
            )

        else:
            optimized_scores = (
                learned_rewards
            )

        expected_objective = (
            probabilities
            * optimized_scores
        ).sum(
            dim=-1
        ).mean()

        entropy = -(
            probabilities
            * torch.log(
                probabilities
                + 1e-8
            )
        ).sum(
            dim=-1
        ).mean()

        loss = (
            -expected_objective
            - 0.001 * entropy
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if method == "lagrangian":
            with torch.no_grad():
                expected_cost = float(
                    (
                        probabilities
                        * costs
                    )
                    .sum(
                        dim=-1
                    )
                    .mean()
                    .item()
                )

                lagrange_multiplier = float(
                    np.clip(
                        lagrange_multiplier
                        + lambda_learning_rate
                        * (
                            expected_cost
                            - cost_limit
                        ),
                        0.0,
                        100.0,
                    )
                )

    policy.eval()

    return (
        policy,
        lagrange_multiplier,
    )


def evaluate_policy(
    policy: ResponsePolicy,
    contexts: torch.Tensor,
    features: torch.Tensor,
    learned_rewards: torch.Tensor,
    method: str,
) -> dict[str, float]:
    costs = violation_indicator(
        features
    )

    with torch.no_grad():
        logits = policy(
            contexts
        )

        if method == "hard_veto":
            logits = logits.masked_fill(
                costs > 0.5,
                -1e9,
            )

        actions = torch.argmax(
            logits,
            dim=-1,
        )

    row_indices = torch.arange(
        contexts.shape[0]
    )

    selected_features = features[
        row_indices,
        actions,
    ]

    selected_rewards = learned_rewards[
        row_indices,
        actions,
    ]

    selected_utility = true_utility(
        selected_features
    )

    selected_cost = violation_indicator(
        selected_features
    )

    truthfulness = (
        selected_features[:, 0]
    )

    sycophancy = (
        (
            selected_features[:, 2]
            > 0.5
        )
        & (
            selected_features[:, 0]
            < 0.5
        )
    ).float()

    deception = (
        selected_features[:, 4]
    )

    refusal = (
        selected_features[:, 5]
    )

    monitored = (
        contexts[:, 2]
        > 0.5
    )

    violation_monitored = (
        float(
            selected_cost[
                monitored
            ].mean().item()
        )
        if monitored.any()
        else float("nan")
    )

    violation_unmonitored = (
        float(
            selected_cost[
                ~monitored
            ].mean().item()
        )
        if (
            ~monitored
        ).any()
        else float("nan")
    )

    return {
        "mean_learned_reward":
            float(
                selected_rewards
                .mean()
                .item()
            ),
        "mean_true_utility":
            float(
                selected_utility
                .mean()
                .item()
            ),
        "violation_rate":
            float(
                selected_cost
                .mean()
                .item()
            ),
        "false_response_rate":
            float(
                (
                    1.0
                    - truthfulness
                )
                .mean()
                .item()
            ),
        "sycophancy_rate":
            float(
                sycophancy
                .mean()
                .item()
            ),
        "deception_rate":
            float(
                deception
                .mean()
                .item()
            ),
        "refusal_rate":
            float(
                refusal
                .mean()
                .item()
            ),
        "reward_true_gap":
            float(
                (
                    selected_rewards
                    - selected_utility
                )
                .mean()
                .item()
            ),
        "violation_monitored":
            violation_monitored,
        "violation_unmonitored":
            violation_unmonitored,
        "monitoring_gap":
            float(
                violation_unmonitored
                - violation_monitored
            ),
    }


def run_experiment(
    configuration: dict[str, Any],
) -> pd.DataFrame:
    contexts = generate_contexts()

    features = response_features(
        contexts
    )

    records = []

    combinations = [
        (
            int(seed),
            float(perturbation),
        )
        for seed in configuration[
            "random_seeds"
        ]
        for perturbation in configuration[
            "perturbations"
        ]
    ]

    total = (
        len(combinations)
        * len(
            configuration[
                "methods"
            ]
        )
    )

    completed = 0

    for seed, perturbation in combinations:
        reward_model = (
            train_reward_model(
                contexts=contexts,
                features=features,
                perturbation=
                    perturbation,
                seed=seed,
                epochs=int(
                    configuration[
                        "reward_model_epochs"
                    ]
                ),
                learning_rate=float(
                    configuration[
                        "reward_model_learning_rate"
                    ]
                ),
            )
        )

        learned_rewards = (
            reward_model_scores(
                reward_model,
                contexts,
                features,
            )
        )

        for method in configuration[
            "methods"
        ]:
            policy, multiplier = (
                train_policy(
                    contexts=contexts,
                    features=features,
                    learned_rewards=
                        learned_rewards,
                    method=method,
                    seed=seed,
                    epochs=int(
                        configuration[
                            "policy_epochs"
                        ]
                    ),
                    learning_rate=float(
                        configuration[
                            "policy_learning_rate"
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
                )
            )

            metrics = evaluate_policy(
                policy=policy,
                contexts=contexts,
                features=features,
                learned_rewards=
                    learned_rewards,
                method=method,
            )

            records.append(
                {
                    "seed":
                        seed,
                    "perturbation":
                        perturbation,
                    "method":
                        method,
                    "final_lagrange_multiplier":
                        multiplier,
                    **metrics,
                }
            )

            completed += 1

            if (
                completed % 20 == 0
                or completed == total
            ):
                print(
                    f"Completed "
                    f"{completed}/{total}"
                )

    return pd.DataFrame(
        records
    )


def load_configuration(
    path: str | Path,
) -> dict[str, Any]:
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        configuration = yaml.safe_load(
            file
        )

    required = {
        "project_name",
        "random_seeds",
        "perturbations",
        "methods",
        "reward_model_epochs",
        "reward_model_learning_rate",
        "policy_epochs",
        "policy_learning_rate",
        "fixed_penalty",
        "cost_limit",
        "lambda_learning_rate",
        "output_dir",
    }

    missing = sorted(
        required.difference(
            configuration
        )
    )

    if missing:
        raise ValueError(
            "Missing fields: "
            + ", ".join(missing)
        )

    unknown_methods = sorted(
        set(
            configuration[
                "methods"
            ]
        ).difference(
            METHODS
        )
    )

    if unknown_methods:
        raise ValueError(
            "Unknown methods: "
            + ", ".join(
                unknown_methods
            )
        )

    return configuration


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    arguments = parser.parse_args()

    configuration = load_configuration(
        arguments.config
    )

    output_directory = Path(
        configuration[
            "output_dir"
        ]
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = run_experiment(
        configuration
    )

    results.to_csv(
        output_directory
        / "rlhf_proxy_results.csv",
        index=False,
    )

    manifest = {
        "configuration":
            configuration,
        "rows":
            len(results),
        "python":
            platform.python_version(),
        "numpy":
            np.__version__,
        "pandas":
            pd.__version__,
        "torch":
            torch.__version__,
        "scope": (
            "Controlled RLHF-style "
            "response-selection analogue"
        ),
    }

    (
        output_directory
        / "rlhf_proxy_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = (
        results.groupby(
            [
                "perturbation",
                "method",
            ]
        )
        .agg(
            n_seeds=(
                "seed",
                "nunique",
            ),
            mean_utility=(
                "mean_true_utility",
                "mean",
            ),
            mean_violation=(
                "violation_rate",
                "mean",
            ),
            mean_sycophancy=(
                "sycophancy_rate",
                "mean",
            ),
            mean_deception=(
                "deception_rate",
                "mean",
            ),
            mean_reward_gap=(
                "reward_true_gap",
                "mean",
            ),
        )
        .reset_index()
        .round(4)
    )

    print("=" * 110)
    print("CONTROLLED RLHF PROXY RESULTS")
    print("=" * 110)

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nSaved:",
        output_directory
        / "rlhf_proxy_results.csv",
    )


if __name__ == "__main__":
    main()
