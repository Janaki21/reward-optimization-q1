from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis import run_full_analysis
from src.config import load_config
from src.plots import generate_all_plots
from src.simulate import run_all_simulations
from src.utils import ensure_directories


def save_tables(analysis_outputs: dict[str, pd.DataFrame], tables_dir: str) -> None:
    Path(tables_dir).mkdir(parents=True, exist_ok=True)

    for name, df in analysis_outputs.items():
        csv_path = Path(tables_dir) / f"{name}.csv"
        xlsx_path = Path(tables_dir) / f"{name}.xlsx"
        df.to_csv(csv_path, index=False)
        df.to_excel(xlsx_path, index=False)


def main() -> None:
    config = load_config("configs/base.yaml")

    ensure_directories(
        [
            config["output_dir"],
            config["figures_dir"],
            config["tables_dir"],
            config["logs_dir"],
        ]
    )

    print("=" * 80)
    print("RUNNING Q1-STYLE REWARD INVARIANCE EXPERIMENTS")
    print("=" * 80)

    episode_df, curve_df, run_summary_df = run_all_simulations(config)

    episode_df.to_csv(Path(config["logs_dir"]) / "episode_level_logs.csv", index=False)
    curve_df.to_csv(Path(config["logs_dir"]) / "learning_curves.csv", index=False)
    run_summary_df.to_csv(Path(config["logs_dir"]) / "run_summary.csv", index=False)

    print("Simulation complete.")
    print(f"Episode logs: {Path(config['logs_dir']) / 'episode_level_logs.csv'}")
    print(f"Learning curves: {Path(config['logs_dir']) / 'learning_curves.csv'}")
    print(f"Run summaries: {Path(config['logs_dir']) / 'run_summary.csv'}")

    analysis_outputs = run_full_analysis(run_summary_df)
    save_tables(analysis_outputs, config["tables_dir"])

    agg_df = analysis_outputs["aggregated_statistics"]
    generate_all_plots(agg_df, curve_df, config["figures_dir"])

    print("Analysis complete.")
    print(f"Tables saved in: {config['tables_dir']}")
    print(f"Figures saved in: {config['figures_dir']}")

    print("\nGenerated tables:")
    for name in analysis_outputs.keys():
        print(f" - {name}.csv")
        print(f" - {name}.xlsx")

    print("\nGenerated figures:")
    figure_names = [
        "adversarial_primary_metric.png",
        "truth_reward_primary_metric.png",
        "monitoring_primary_metric.png",
        "adversarial_utility_tradeoff.png",
        "truth_reward_utility_tradeoff.png",
        "monitoring_utility_tradeoff.png",
        "adversarial_learning_curve_avg_reward_so_far.png",
        "adversarial_learning_curve_cvr_so_far.png",
        "truth_reward_learning_curve_avg_reward_so_far.png",
        "truth_reward_learning_curve_far_so_far.png",
        "monitoring_learning_curve_avg_reward_so_far.png",
        "monitoring_learning_curve_cvr_so_far.png",
        "monitoring_learning_curve_deception_index_so_far.png",
        "adversarial_ablation_bar.png",
        "truth_reward_ablation_bar.png",
        "monitoring_ablation_bar.png",
        "adversarial_scaling_analysis.png",
        "truth_reward_scaling_analysis.png",
        "monitoring_scaling_analysis.png",
        "integrated_summary_figure.png",
    ]
    for fig in figure_names:
        print(f" - {fig}")

    print("\nDone.")


if __name__ == "__main__":
    main()