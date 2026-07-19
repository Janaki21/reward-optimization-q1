"""Verify final experimental outputs and generate SHA-256 checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import torch


EXPECTED_SEEDS = [
    11,
    17,
    23,
    31,
    47,
    59,
    71,
    83,
    97,
    109,
    127,
    139,
    151,
    163,
    179,
    191,
    211,
    223,
    239,
    251,
    269,
    281,
    293,
    307,
    331,
    347,
    359,
    373,
    389,
    401,
]


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        while True:
            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def require_file(
    path: Path,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file is missing: {path}"
        )

    if path.stat().st_size == 0:
        raise RuntimeError(
            f"Required file is empty: {path}"
        )


def validate_seeds(
    frame: pd.DataFrame,
    label: str,
) -> None:
    actual = sorted(
        int(seed)
        for seed in frame[
            "seed"
        ].unique()
    )

    if actual != EXPECTED_SEEDS:
        raise RuntimeError(
            f"{label} seed mismatch.\n"
            f"Expected: {EXPECTED_SEEDS}\n"
            f"Actual: {actual}"
        )


def validate_status(
    frame: pd.DataFrame,
    label: str,
) -> None:
    if "status" not in frame.columns:
        return

    failures = frame[
        frame["status"]
        != "completed"
    ]

    if not failures.empty:
        raise RuntimeError(
            f"{label} contains "
            f"{len(failures)} failed rows."
        )


def validate_final_results(
    repository: Path,
) -> dict[str, Any]:
    paths = {
        "omnisafe": (
            repository
            / "outputs/final/"
            "omnisafe_30seed/"
            "omnisafe_pilot_summary.csv"
        ),
        "lagrange": (
            repository
            / "outputs/final/"
            "lagrange_30seed/"
            "lagrange_sensitivity_summary.csv"
        ),
        "veto": (
            repository
            / "outputs/final/"
            "veto_30seed/"
            "veto_run_summary.csv"
        ),
        "rlhf": (
            repository
            / "outputs/final/"
            "rlhf_calibrated_30seed/"
            "rlhf_calibrated_results.csv"
        ),
        "primary_summary": (
            repository
            / "outputs/final/"
            "analysis/tables/"
            "primary_summary.csv"
        ),
        "paired_tests": (
            repository
            / "outputs/final/"
            "analysis/tables/"
            "paired_tests_holm.csv"
        ),
        "veto_auc": (
            repository
            / "outputs/final/"
            "analysis/tables/"
            "veto_auc_summary.csv"
        ),
        "rlhf_summary": (
            repository
            / "outputs/final/"
            "rlhf_analysis/tables/"
            "rlhf_summary.csv"
        ),
        "rlhf_tests": (
            repository
            / "outputs/final/"
            "rlhf_analysis/tables/"
            "rlhf_paired_tests_holm.csv"
        ),
        "rlhf_auc": (
            repository
            / "outputs/final/"
            "rlhf_analysis/tables/"
            "rlhf_auc_summary.csv"
        ),
    }

    for path in paths.values():
        require_file(
            path
        )

    omnisafe = pd.read_csv(
        paths["omnisafe"]
    )

    lagrange = pd.read_csv(
        paths["lagrange"]
    )

    veto = pd.read_csv(
        paths["veto"]
    )

    rlhf = pd.read_csv(
        paths["rlhf"]
    )

    expected_rows = {
        "omnisafe": 180,
        "lagrange": 180,
        "veto": 450,
        "rlhf": 600,
    }

    actual_rows = {
        "omnisafe":
            len(omnisafe),
        "lagrange":
            len(lagrange),
        "veto":
            len(veto),
        "rlhf":
            len(rlhf),
    }

    for label, expected in (
        expected_rows.items()
    ):
        actual = (
            actual_rows[label]
        )

        if actual != expected:
            raise RuntimeError(
                f"{label}: expected "
                f"{expected} rows, "
                f"found {actual}."
            )

    validate_status(
        omnisafe,
        "OmniSafe",
    )

    validate_status(
        lagrange,
        "Lagrangian",
    )

    validate_seeds(
        omnisafe,
        "OmniSafe",
    )

    validate_seeds(
        lagrange,
        "Lagrangian",
    )

    validate_seeds(
        veto,
        "Veto",
    )

    validate_seeds(
        rlhf,
        "RLHF",
    )

    perfect_veto = veto[
        (
            veto["method"]
            == "ppo_hard_veto"
        )
        & (
            veto[
                "constraint_error"
            ]
            == 0.0
        )
    ]

    if len(perfect_veto) != 90:
        raise RuntimeError(
            "Expected 90 perfect-veto rows, "
            f"found {len(perfect_veto)}."
        )

    maximum_perfect_violation = float(
        perfect_veto[
            "policy_violation_rate"
        ].max()
    )

    if (
        maximum_perfect_violation
        != 0.0
    ):
        raise RuntimeError(
            "Perfect veto has a non-zero "
            "violation rate."
        )

    rlhf_veto = rlhf[
        rlhf["method"]
        == "hard_veto"
    ]

    if len(rlhf_veto) != 150:
        raise RuntimeError(
            "Expected 150 RLHF veto rows, "
            f"found {len(rlhf_veto)}."
        )

    maximum_rlhf_veto_violation = (
        float(
            rlhf_veto[
                "violation_rate"
            ].max()
        )
    )

    if (
        maximum_rlhf_veto_violation
        != 0.0
    ):
        raise RuntimeError(
            "RLHF hard veto has a non-zero "
            "violation rate."
        )

    for environment in sorted(
        veto[
            "environment"
        ].unique()
    ):
        environment_rows = veto[
            (
                veto[
                    "environment"
                ]
                == environment
            )
            & (
                veto["method"]
                == "ppo_hard_veto"
            )
        ]

        relationship = (
            environment_rows.groupby(
                "constraint_error"
            )[
                "policy_violation_rate"
            ]
            .mean()
            .sort_index()
            .to_numpy(
                dtype=float
            )
        )

        if np.any(
            np.diff(
                relationship
            )
            < -1e-12
        ):
            raise RuntimeError(
                "Veto violations are not "
                "non-decreasing for "
                f"{environment}."
            )

    neural_figures = list(
        (
            repository
            / "outputs/final/"
            "analysis/figures"
        ).glob("*")
    )

    rlhf_figures = list(
        (
            repository
            / "outputs/final/"
            "rlhf_analysis/figures"
        ).glob("*")
    )

    if len(neural_figures) != 12:
        raise RuntimeError(
            "Expected 12 primary neural "
            f"figure files, found "
            f"{len(neural_figures)}."
        )

    if len(rlhf_figures) != 12:
        raise RuntimeError(
            "Expected 12 RLHF figure files, "
            f"found {len(rlhf_figures)}."
        )

    checksum_paths = [
        *paths.values(),
        *neural_figures,
        *rlhf_figures,
    ]

    checksums = {
        str(
            path.relative_to(
                repository
            )
        ): sha256_file(path)
        for path in sorted(
            checksum_paths
        )
    }

    return {
        "row_counts":
            actual_rows,
        "seed_count":
            len(EXPECTED_SEEDS),
        "maximum_perfect_veto_violation":
            maximum_perfect_violation,
        "maximum_rlhf_veto_violation":
            maximum_rlhf_veto_violation,
        "primary_figure_files":
            len(
                neural_figures
            ),
        "rlhf_figure_files":
            len(
                rlhf_figures
            ),
        "checksums":
            checksums,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repository",
        default=".",
    )

    parser.add_argument(
        "--output",
        default=(
            "outputs/final/"
            "reproducibility_manifest.json"
        ),
    )

    arguments = parser.parse_args()

    repository = Path(
        arguments.repository
    ).resolve()

    validation = (
        validate_final_results(
            repository
        )
    )

    manifest = {
        "validation":
            "passed",
        **validation,
        "software": {
            "python":
                platform.python_version(),
            "numpy":
                np.__version__,
            "pandas":
                pd.__version__,
            "scipy":
                scipy.__version__,
            "torch":
                torch.__version__,
        },
    }

    output_path = (
        repository
        / arguments.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 90)
    print(
        "FINAL REPRODUCIBILITY VALIDATION"
    )
    print("=" * 90)

    print(
        "Row counts:",
        validation["row_counts"],
    )

    print(
        "Seeds:",
        validation["seed_count"],
    )

    print(
        "Maximum perfect-veto violation:",
        validation[
            "maximum_perfect_veto_violation"
        ],
    )

    print(
        "Maximum RLHF-veto violation:",
        validation[
            "maximum_rlhf_veto_violation"
        ],
    )

    print(
        "Primary figure files:",
        validation[
            "primary_figure_files"
        ],
    )

    print(
        "RLHF figure files:",
        validation[
            "rlhf_figure_files"
        ],
    )

    print(
        "Checksummed files:",
        len(
            validation["checksums"]
        ),
    )

    print(
        "Manifest:",
        output_path,
    )

    print(
        "\nReproducibility validation: PASS"
    )


if __name__ == "__main__":
    main()