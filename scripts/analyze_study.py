import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game_env.primal_hunt_env import (
    BIG_ANIM,
    EMPTY,
    HOME,
    OBSTACLE,
    SMALL_ANIM,
    VEG_FRUIT,
    PrimalHuntEnv,
)


RUN_LABELS = {
    "A": "Full DQN",
    "B": "No target network",
    "C": "No replay",
    "D": "Online neural Q-learning",
}
CELL_LABELS = {
    HOME: "Home",
    VEG_FRUIT: "Vegetation",
    SMALL_ANIM: "Small animal",
    BIG_ANIM: "Big animal",
    OBSTACLE: "Obstacle",
    EMPTY: "Empty",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the DQN ablation study.")
    parser.add_argument(
        "--results-dir", type=Path, default=PROJECT_ROOT / "results"
    )
    parser.add_argument(
        "--rolling-window", type=int, default=200
    )
    parser.add_argument(
        "--energy-samples", type=int, default=20_000
    )
    return parser.parse_args()


def load_study_data(
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    episode_files = sorted(
        (results_dir / "training" / "episodes").glob("*_episodes.csv")
    )
    if not episode_files:
        raise FileNotFoundError("No training episode files were found.")
    episodes = pd.concat(
        [pd.read_csv(path) for path in episode_files], ignore_index=True
    )
    per_seed_path = results_dir / "training" / "per_seed_summary.csv"
    if not per_seed_path.exists():
        raise FileNotFoundError(f"Missing {per_seed_path}")
    per_seed = pd.read_csv(per_seed_path)
    evaluation_files = sorted(
        (results_dir / "training" / "evaluations").glob("*_evaluations.csv")
    )
    if not evaluation_files:
        raise FileNotFoundError("No periodic evaluation files were found.")
    evaluations = pd.concat(
        [pd.read_csv(path) for path in evaluation_files], ignore_index=True
    )
    return episodes, per_seed, evaluations


def validate_study_data(
    episodes: pd.DataFrame, per_seed: pd.DataFrame, evaluations: pd.DataFrame
) -> pd.DataFrame:
    required_episode_columns = {
        "run_id", "seed", "episode", "objective_return", "final_energy",
        "survived", "mean_td_loss", "epsilon", "mean_gradient_norm",
        "max_gradient_norm", "weight_norm", "mean_abs_q", "max_abs_q",
        "mean_abs_td_error", "target_distance",
    }
    required_summary_columns = {
        "run_id", "seed", "eval_mean_objective_return",
        "eval_survival_rate", "eval_mean_final_energy", "eval_final_energy_std",
    }
    if missing := required_episode_columns - set(episodes.columns):
        raise ValueError(f"Missing episode columns: {sorted(missing)}")
    if missing := required_summary_columns - set(per_seed.columns):
        raise ValueError(f"Missing summary columns: {sorted(missing)}")
    if set(episodes["run_id"].unique()) != set(RUN_LABELS):
        raise ValueError("Expected exactly configurations A, B, C, and D.")
    if episodes.duplicated(["run_id", "seed", "episode"]).any():
        raise ValueError("Duplicate run/seed/episode rows found.")

    counts = episodes.groupby(["run_id", "seed"])["episode"].count()
    expected_episodes = int(counts.iloc[0])
    if not counts.eq(expected_episodes).all():
        raise ValueError("Runs contain different numbers of episodes.")
    seeds_per_run = per_seed.groupby("run_id")["seed"].nunique()
    if seeds_per_run.nunique() != 1:
        raise ValueError("Configurations contain different seed counts.")
    finite_loss = episodes["mean_td_loss"].dropna()
    if not np.isfinite(finite_loss).all():
        raise ValueError("Non-finite TD losses found.")
    evaluation_counts = evaluations.groupby(["run_id", "seed"])["episode"].count()
    if evaluation_counts.nunique() != 1:
        raise ValueError("Runs contain different periodic evaluation counts.")

    return pd.DataFrame(
        {
            "check": [
                "Configurations",
                "Seeds per configuration",
                "Episodes per seed",
                "Duplicate episode rows",
                "Non-finite recorded losses",
                "Final epsilon (mean)",
                "Periodic evaluations per seed",
            ],
            "value": [
                ", ".join(sorted(RUN_LABELS)),
                int(seeds_per_run.iloc[0]),
                expected_episodes,
                0,
                0,
                episodes.groupby(["run_id", "seed"]).tail(1)["epsilon"].mean(),
                int(evaluation_counts.iloc[0]),
            ],
        }
    )


def sample_cell_energy_changes(samples: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parameters = {
        "Vegetation": (0.95, 5, 3, 0.05, 3),
        "Small animal": (0.70, 9, 5, 0.10, 6),
        "Big animal": (0.45, 60, 15, 0.30, 20),
        "Obstacle": (0.0, 0, 5, 0.25, 20),
    }

    def truncated(mean: float, size: int) -> np.ndarray:
        lower = -mean
        return stats.truncnorm.rvs(
            lower, np.inf, loc=mean, scale=1, size=size, random_state=rng
        )

    frames = []
    for label, (food_probability, food_mean, effort_mean,
                injury_probability, injury_mean) in parameters.items():
        food = np.zeros(samples)
        food_success = rng.random(samples) < food_probability
        if food_success.any():
            food[food_success] = truncated(food_mean, int(food_success.sum()))
        effort = truncated(effort_mean, samples)
        injury = np.zeros(samples)
        injury_event = rng.random(samples) < injury_probability
        if injury_event.any():
            injury[injury_event] = truncated(injury_mean, int(injury_event.sum()))
        frames.append(
            pd.DataFrame(
                {
                    "cell_type": label,
                    "food": food,
                    "effort": effort,
                    "injury": injury,
                    "energy_delta": food - effort - injury,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def energy_summary(energy_changes: pd.DataFrame) -> pd.DataFrame:
    return energy_changes.groupby("cell_type", as_index=False).agg(
        mean_energy_delta=("energy_delta", "mean"),
        energy_delta_std=("energy_delta", "std"),
        probability_negative=("energy_delta", lambda values: (values < 0).mean()),
        mean_food=("food", "mean"),
        mean_effort=("effort", "mean"),
        mean_injury=("injury", "mean"),
    )


def _finish_figure(fig, output_path: Path | None):
    if output_path is not None:
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
    return fig


def plot_map(output_path: Path | None = None):
    env = PrimalHuntEnv()
    colors = ["#f4f4f4", "#75b66b", "#d7a95b", "#b34b4b", "#6f7782", "#ffffff"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(env._grid, cmap=plt.matplotlib.colors.ListedColormap(colors), vmin=0, vmax=5)
    for row in range(env.cfg.grid_size):
        for col in range(env.cfg.grid_size):
            ax.text(
                col, row, CELL_LABELS[int(env._grid[row, col])],
                ha="center", va="center", fontsize=8,
            )
    ax.set_xticks(range(env.cfg.grid_size))
    ax.set_yticks(range(env.cfg.grid_size))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("Primal Hunt Map")
    ax.set_xticks(np.arange(-0.5, env.cfg.grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.cfg.grid_size, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=1)
    fig.tight_layout()
    return _finish_figure(fig, output_path)


def plot_energy_change_distributions(
    energy_changes: pd.DataFrame, output_path: Path | None = None
):
    order = ["Vegetation", "Small animal", "Big animal", "Obstacle"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        [energy_changes.loc[energy_changes["cell_type"] == label, "energy_delta"] for label in order],
        tick_labels=order, showfliers=False,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Net energy change")
    ax.set_title("One-Step Energy-Change Distributions")
    fig.tight_layout()
    return _finish_figure(fig, output_path)


def plot_training_metric(
    episodes: pd.DataFrame,
    metric: str,
    ylabel: str,
    rolling_window: int,
    output_path: Path | None = None,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    for run_id, run_data in episodes.groupby("run_id"):
        curves = []
        for _, seed_data in run_data.groupby("seed"):
            seed_data = seed_data.sort_values("episode")
            curve = seed_data[metric].rolling(
                rolling_window, min_periods=max(1, rolling_window // 10)
            ).mean()
            curves.append(pd.Series(curve.to_numpy(), index=seed_data["episode"]))
        frame = pd.concat(curves, axis=1)
        mean = frame.mean(axis=1)
        std = frame.std(axis=1).fillna(0)
        ax.plot(mean.index, mean, label=f"{run_id}: {RUN_LABELS[run_id]}")
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.15)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Training {ylabel}: {rolling_window}-Episode Rolling Mean")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _finish_figure(fig, output_path)


def policy_quality(per_seed: pd.DataFrame) -> pd.DataFrame:
    return per_seed.groupby("run_id", as_index=False).agg(
        seeds=("seed", "count"),
        eval_mean_final_energy=("eval_mean_final_energy", "mean"),
        between_seed_final_energy_std=("eval_mean_final_energy", "std"),
        mean_within_policy_final_energy_std=("eval_final_energy_std", "mean"),
        eval_survival_rate=("eval_survival_rate", "mean"),
        between_seed_survival_std=("eval_survival_rate", "std"),
    )


def plot_policy_quality(per_seed: pd.DataFrame, output_path: Path | None = None):
    quality = policy_quality(per_seed).set_index("run_id")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, error, ylabel in (
        (axes[0], "eval_mean_final_energy", "between_seed_final_energy_std", "Final energy"),
        (axes[1], "eval_survival_rate", "between_seed_survival_std", "Survival rate"),
    ):
        ax.bar(quality.index, quality[metric], yerr=quality[error], capsize=5, alpha=0.75)
        for index, run_id in enumerate(quality.index):
            values = per_seed.loc[per_seed["run_id"] == run_id, metric]
            ax.scatter(np.full(len(values), index), values, color="black", s=20, zorder=3)
        ax.set_xlabel("Configuration")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[1].set_ylim(0, 1)
    fig.suptitle("Held-Out Greedy Policy Quality")
    fig.tight_layout()
    return _finish_figure(fig, output_path)


def plot_energy_survival_tradeoff(
    per_seed: pd.DataFrame, output_path: Path | None = None
):
    fig, ax = plt.subplots(figsize=(8, 6))
    for run_id, data in per_seed.groupby("run_id"):
        ax.scatter(
            data["eval_mean_final_energy"], data["eval_survival_rate"],
            label=f"{run_id}: {RUN_LABELS[run_id]}", s=55,
        )
    ax.set_xlabel("Greedy evaluation final energy")
    ax.set_ylabel("Greedy evaluation survival rate")
    ax.set_title("Energy-Survival Tradeoff Across Seeds")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _finish_figure(fig, output_path)


def paired_effects(per_seed: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("Target effect with replay", "A", "B"),
        ("Target effect without replay", "C", "D"),
        ("Replay effect with target", "A", "C"),
        ("Replay effect without target", "B", "D"),
    ]
    rows = []
    for metric in ("eval_mean_final_energy", "eval_survival_rate"):
        pivot = per_seed.pivot(index="seed", columns="run_id", values=metric)
        for label, positive, negative in comparisons:
            difference = (pivot[positive] - pivot[negative]).dropna()
            count = len(difference)
            sem = stats.sem(difference) if count > 1 else np.nan
            margin = stats.t.ppf(0.975, count - 1) * sem if count > 1 else np.nan
            test = stats.ttest_1samp(difference, 0.0) if count > 1 else None
            rows.append(
                {
                    "metric": metric,
                    "effect": label,
                    "contrast": f"{positive} - {negative}",
                    "seeds": count,
                    "mean_paired_difference": difference.mean(),
                    "ci_95_low": difference.mean() - margin,
                    "ci_95_high": difference.mean() + margin,
                    "p_value": test.pvalue if test else np.nan,
                }
            )
    return pd.DataFrame(rows)


def stability_summary(episodes: pd.DataFrame, per_seed: pd.DataFrame) -> pd.DataFrame:
    final_window = (
        episodes.sort_values("episode")
        .groupby(["run_id", "seed"], group_keys=False)
        .tail(500)
    )
    per_run_seed = final_window.groupby(["run_id", "seed"], as_index=False).agg(
        late_objective_return=("objective_return", "mean"),
        late_final_energy=("final_energy", "mean"),
        late_final_energy_std=("final_energy", "std"),
        late_survival=("survived", "mean"),
        late_td_loss=("mean_td_loss", "mean"),
        late_td_loss_std=("mean_td_loss", "std"),
        late_gradient_norm=("mean_gradient_norm", "mean"),
        late_gradient_norm_max=("max_gradient_norm", "max"),
        late_weight_norm=("weight_norm", "mean"),
        late_mean_abs_q=("mean_abs_q", "mean"),
        late_max_abs_q=("max_abs_q", "max"),
        late_target_distance=("target_distance", "mean"),
    )
    stability = per_run_seed.groupby("run_id", as_index=False).agg(
        late_objective_return=("late_objective_return", "mean"),
        late_final_energy=("late_final_energy", "mean"),
        late_final_energy_variability=("late_final_energy_std", "mean"),
        late_train_survival=("late_survival", "mean"),
        late_td_loss=("late_td_loss", "mean"),
        late_td_loss_variability=("late_td_loss_std", "mean"),
        late_gradient_norm=("late_gradient_norm", "mean"),
        late_max_gradient_norm=("late_gradient_norm_max", "mean"),
        late_weight_norm=("late_weight_norm", "mean"),
        late_mean_abs_q=("late_mean_abs_q", "mean"),
        late_max_abs_q=("late_max_abs_q", "mean"),
        late_target_distance=("late_target_distance", "mean"),
    )
    evaluation_variance = per_seed.groupby("run_id")["eval_final_energy_std"].mean()
    stability["evaluation_final_energy_variability"] = stability["run_id"].map(
        evaluation_variance
    )
    return stability


def periodic_evaluation_summary(evaluations: pd.DataFrame) -> pd.DataFrame:
    final = evaluations.sort_values("episode").groupby(
        ["run_id", "seed"], as_index=False
    ).tail(1)
    return final.groupby("run_id", as_index=False).agg(
        checkpoint_episode=("episode", "max"),
        mean_final_energy=("mean_final_energy", "mean"),
        between_seed_energy_std=("mean_final_energy", "std"),
        mean_survival=("survival_rate", "mean"),
        between_seed_survival_std=("survival_rate", "std"),
    )


def plot_periodic_evaluation(
    evaluations: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path | None = None,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    for run_id, data in evaluations.groupby("run_id"):
        curve = data.groupby("episode")[metric].agg(["mean", "std"])
        std = curve["std"].fillna(0)
        ax.plot(curve.index, curve["mean"], label=f"{run_id}: {RUN_LABELS[run_id]}")
        ax.fill_between(curve.index, curve["mean"] - std,
                        curve["mean"] + std, alpha=0.15)
    ax.set_xlabel("Training episode")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Periodic Greedy Evaluation: {ylabel}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _finish_figure(fig, output_path)


def plot_stability_diagnostics(
    episodes: pd.DataFrame,
    rolling_window: int,
    output_path: Path | None = None,
):
    metrics = (
        ("mean_td_loss", "TD loss"),
        ("mean_gradient_norm", "Gradient norm before clipping"),
        ("weight_norm", "Policy weight norm"),
        ("max_abs_q", "Maximum absolute Q"),
        ("mean_abs_td_error", "Mean absolute TD error"),
        ("target_distance", "Policy-target distance"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (metric, label) in zip(axes.flat, metrics):
        for run_id, run_data in episodes.groupby("run_id"):
            if not run_data[metric].notna().any():
                continue
            curves = []
            for _, seed_data in run_data.groupby("seed"):
                seed_data = seed_data.sort_values("episode")
                curve = seed_data[metric].rolling(
                    rolling_window,
                    min_periods=max(1, rolling_window // 10),
                ).mean()
                curves.append(pd.Series(curve.to_numpy(), index=seed_data["episode"]))
            frame = pd.concat(curves, axis=1)
            ax.plot(frame.index, frame.mean(axis=1), label=run_id)
        ax.set_title(label)
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(title="Configuration")
    fig.suptitle("Empirical Neural Q-Learning Stability Diagnostics")
    fig.tight_layout()
    return _finish_figure(fig, output_path)


def run_analysis(
    results_dir: Path,
    rolling_window: int = 200,
    energy_samples: int = 20_000,
) -> dict[str, pd.DataFrame]:
    output_dir = results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes, per_seed, evaluations = load_study_data(results_dir)
    validation = validate_study_data(episodes, per_seed, evaluations)
    energy_changes = sample_cell_energy_changes(energy_samples)
    tables = {
        "validation": validation,
        "energy_summary": energy_summary(energy_changes),
        "policy_quality": policy_quality(per_seed),
        "paired_effects": paired_effects(per_seed),
        "stability": stability_summary(episodes, per_seed),
        "periodic_evaluation": periodic_evaluation_summary(evaluations),
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    plot_map(output_dir / "environment_map.png")
    plot_energy_change_distributions(
        energy_changes, output_dir / "cell_energy_change_distributions.png"
    )
    for metric, ylabel, filename in (
        ("final_energy", "Final energy", "training_energy.png"),
        ("survived", "Survival rate", "training_survival.png"),
    ):
        plot_training_metric(
            episodes, metric, ylabel, rolling_window, output_dir / filename
        )
    plot_policy_quality(per_seed, output_dir / "final_policy_quality.png")
    plot_energy_survival_tradeoff(per_seed, output_dir / "energy_survival_tradeoff.png")
    plot_periodic_evaluation(
        evaluations, "mean_final_energy", "Final energy",
        output_dir / "greedy_evaluation_energy.png",
    )
    plot_periodic_evaluation(
        evaluations, "survival_rate", "Survival rate",
        output_dir / "greedy_evaluation_survival.png",
    )
    plot_stability_diagnostics(
        episodes, rolling_window, output_dir / "stability_diagnostics.png"
    )

    metadata = {
        "rolling_window": rolling_window,
        "energy_samples_per_cell": energy_samples,
        "runs": sorted(episodes["run_id"].unique().tolist()),
        "seeds": sorted(per_seed["seed"].unique().tolist()),
        "episodes_per_seed": int(episodes.groupby(["run_id", "seed"]).size().iloc[0]),
    }
    (output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2))
    return tables


def main() -> None:
    args = parse_args()
    tables = run_analysis(args.results_dir, args.rolling_window, args.energy_samples)
    for name, table in tables.items():
        print(f"\n{name.replace('_', ' ').title()}\n{table.to_string(index=False)}")
    print(f"\nAnalysis written to {args.results_dir / 'analysis'}")


if __name__ == "__main__":
    main()
