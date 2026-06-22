import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_hyperparameter_study import FACTOR_RUNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze controlled tuning study.")
    parser.add_argument(
        "--results-dir", type=Path,
        default=PROJECT_ROOT / "results" / "hyperparameter_study",
    )
    return parser.parse_args()


def load_stage(stage_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(stage_dir / "candidates.csv")
    per_seed = pd.read_csv(stage_dir / "training" / "per_seed_summary.csv")
    evaluation_paths = sorted(
        (stage_dir / "training" / "evaluations").glob("*_evaluations.csv")
    )
    if not evaluation_paths:
        raise FileNotFoundError(f"No periodic evaluations in {stage_dir}")
    evaluations = pd.concat(
        [pd.read_csv(path) for path in evaluation_paths], ignore_index=True
    )
    return candidates, per_seed, evaluations


def quality_table(candidates: pd.DataFrame, per_seed: pd.DataFrame) -> pd.DataFrame:
    quality = per_seed.groupby("run_id", as_index=False).agg(
        seeds=("seed", "nunique"),
        eval_mean_final_energy=("eval_mean_final_energy", "mean"),
        between_seed_final_energy_std=("eval_mean_final_energy", "std"),
        eval_survival_rate=("eval_survival_rate", "mean"),
        between_seed_survival_std=("eval_survival_rate", "std"),
    )
    return candidates.merge(quality, on="run_id", validate="one_to_one").sort_values(
        ["eval_mean_final_energy", "between_seed_final_energy_std"],
        ascending=[False, True],
    ).reset_index(drop=True)


def controlled_effects(
    candidates: pd.DataFrame, per_seed: pd.DataFrame
) -> pd.DataFrame:
    merged = per_seed.merge(candidates, on="run_id", validate="many_to_one")
    frames = []
    for parameter, run_ids in FACTOR_RUNS.items():
        data = merged[merged["run_id"].isin(run_ids)].copy()
        grouped = data.groupby(["run_id", parameter], as_index=False).agg(
            seeds=("seed", "nunique"),
            mean_final_energy=("eval_mean_final_energy", "mean"),
            final_energy_std=("eval_mean_final_energy", "std"),
            mean_survival=("eval_survival_rate", "mean"),
        )
        grouped.insert(0, "parameter", parameter)
        grouped = grouped.rename(columns={parameter: "parameter_value"})
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True)


def _finish_figure(fig, output_path: Path | None):
    if output_path is not None:
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
    return fig


def plot_controlled_effects(
    effects: pd.DataFrame, output_path: Path | None = None
):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, parameter in zip(axes.flat, FACTOR_RUNS):
        data = effects[effects["parameter"] == parameter].sort_values(
            "parameter_value"
        )
        ax.errorbar(
            data["parameter_value"], data["mean_final_energy"],
            yerr=data["final_energy_std"], marker="o", capsize=4,
        )
        for _, row in data.iterrows():
            ax.annotate(row["run_id"], (row["parameter_value"], row["mean_final_energy"]),
                        xytext=(4, 5), textcoords="offset points", fontsize=8)
        if parameter == "lr":
            ax.set_xscale("log")
        ax.set_title(parameter.replace("_", " ").title())
        ax.set_xlabel("Value")
        ax.set_ylabel("Greedy final energy (mean +/- seed SD)")
        ax.grid(alpha=0.25)
    fig.suptitle("Controlled One-Factor Hyperparameter Sensitivity")
    fig.tight_layout()
    return _finish_figure(fig, output_path)


def _plot_periodic_factor_curves(
    evaluations: pd.DataFrame, output_path: Path | None = None
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    for ax, (parameter, run_ids) in zip(axes.flat, FACTOR_RUNS.items()):
        for run_id in run_ids:
            data = evaluations[evaluations["run_id"] == run_id]
            curve = data.groupby("episode")["mean_final_energy"].agg(["mean", "std"])
            std = curve["std"].fillna(0)
            ax.plot(curve.index, curve["mean"], label=run_id)
            ax.fill_between(curve.index, curve["mean"] - std,
                            curve["mean"] + std, alpha=0.12)
        ax.set_title(parameter.replace("_", " ").title())
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Periodic greedy final energy")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("Greedy Learning Curves During Hyperparameter Screening")
    fig.tight_layout()
    return _finish_figure(fig, output_path)


def plot_periodic_factor_curves(
    evaluations: pd.DataFrame, output_path: Path | None = None
):
    return _plot_periodic_factor_curves(evaluations, output_path)


def plot_validation(
    per_seed: pd.DataFrame,
    evaluations: pd.DataFrame,
    output_path: Path | None = None,
):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric, ylabel in (
        (axes[0], "eval_mean_final_energy", "Final energy"),
        (axes[1], "eval_survival_rate", "Survival rate"),
    ):
        grouped = per_seed.groupby("run_id")[metric]
        means, errors = grouped.mean(), grouped.std()
        ax.bar(means.index, means, yerr=errors, capsize=4, alpha=0.75)
        for index, run_id in enumerate(means.index):
            values = per_seed.loc[per_seed["run_id"] == run_id, metric]
            ax.scatter(np.full(len(values), index), values, color="black", zorder=3)
        ax.set_xlabel("Validation candidate")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    for run_id, data in evaluations.groupby("run_id"):
        curve = data.groupby("episode")["mean_final_energy"].agg(["mean", "std"])
        std = curve["std"].fillna(0)
        axes[2].plot(curve.index, curve["mean"], label=run_id)
        axes[2].fill_between(curve.index, curve["mean"] - std,
                             curve["mean"] + std, alpha=0.15)
    axes[2].set_xlabel("Training episode")
    axes[2].set_ylabel("Periodic greedy final energy")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    fig.suptitle("Independent-Seed Validation: Baseline vs Combined Candidate")
    fig.tight_layout()
    return _finish_figure(fig, output_path)


def run_analysis(results_dir: Path) -> dict[str, pd.DataFrame]:
    screening = results_dir / "screening"
    validation = results_dir / "validation"
    screen_candidates, screen_seed, screen_evaluations = load_stage(screening)
    validation_candidates, validation_seed, validation_evaluations = load_stage(validation)
    tables = {
        "screening_quality": quality_table(screen_candidates, screen_seed),
        "controlled_effects": controlled_effects(screen_candidates, screen_seed),
        "validation_quality": quality_table(validation_candidates, validation_seed),
    }
    output_dir = results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    plot_controlled_effects(
        tables["controlled_effects"], output_dir / "controlled_sensitivity.png"
    )
    _plot_periodic_factor_curves(
        screen_evaluations, output_dir / "screening_greedy_curves.png"
    )
    plot_validation(
        validation_seed, validation_evaluations,
        output_dir / "validation_comparison.png",
    )
    selected = json.loads((results_dir / "selected_config.json").read_text())
    print(f"Selected {selected['run_id']}: {selected}")
    return tables


def main() -> None:
    args = parse_args()
    tables = run_analysis(args.results_dir)
    for name, table in tables.items():
        print(f"\n{name.replace('_', ' ').title()}\n{table.to_string(index=False)}")


if __name__ == "__main__":
    main()
