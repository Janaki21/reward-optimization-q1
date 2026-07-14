from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from src.analysis import run_full_analysis
from src.config import load_config
from src.plots import generate_all_plots
from src.simulate import run_all_simulations
from src.utils import ensure_directories


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/base.yaml",
    )

    arguments = parser.parse_args()

    config = load_config(arguments.config)

    ensure_directories(
        [
            config["output_dir"],
            config["figures_dir"],
            config["tables_dir"],
            config["logs_dir"],
        ]
    )

    print("Starting reward-misspecification experiments")
    print(f"Configuration: {arguments.config}")

    run_summary, learning_curves = (
        run_all_simulations(config)
    )

    run_summary_path = (
        Path(config["logs_dir"])
        / "run_summary.csv"
    )

    learning_curves_path = (
        Path(config["logs_dir"])
        / "learning_curves.csv"
    )

    run_summary.to_csv(
        run_summary_path,
        index=False,
    )

    learning_curves.to_csv(
        learning_curves_path,
        index=False,
    )

    analysis_outputs = run_full_analysis(
        run_summary
    )

    for table_name, table in (
        analysis_outputs.items()
    ):
        csv_path = (
            Path(config["tables_dir"])
            / f"{table_name}.csv"
        )

        excel_path = (
            Path(config["tables_dir"])
            / f"{table_name}.xlsx"
        )

        table.to_csv(
            csv_path,
            index=False,
        )

        table.to_excel(
            excel_path,
            index=False,
        )

    generate_all_plots(
        aggregated=analysis_outputs[
            "aggregated_statistics"
        ],
        learning_curves=learning_curves,
        figures_directory=config[
            "figures_dir"
        ],
    )

    manifest = {
        "configuration_file": arguments.config,
        "configuration": config,
        "git_commit": get_git_commit(),
        "python_version":
            platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
    }

    manifest_path = (
        Path(config["output_dir"])
        / "run_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(
        f"Completed {len(run_summary)} independent runs."
    )

    print(
        f"Outputs saved in {config['output_dir']}"
    )


if __name__ == "__main__":
    main()