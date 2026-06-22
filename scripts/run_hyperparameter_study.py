import argparse
import json
import sys
from argparse import Namespace
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import ExperimentConfig
from game_env.primal_hunt_env import PrimalHuntEnv
from scripts.train import (
    build_summary,
    evaluate_agent,
    make_agent_config,
    save_aggregate_summary,
    save_outputs,
    seed_everything,
    train_loop,
)


DEVELOPMENT_SEEDS = (11, 22, 33)
VALIDATION_SEEDS = (44, 55, 66)


@dataclass(frozen=True)
class Candidate:
    run_id: str
    lr: float = 1e-3
    batch_size: int = 128
    target_update: int = 50
    epsilon_decay: int = 30_000
    warmup_steps: int = 1_000
    hidden_dim: int = 64
    varying_parameter: str = "baseline"
    value: float = 0.0


BASELINE = Candidate("H00", value=0.0)
SCREENING_CANDIDATES = (
    BASELINE,
    replace(BASELINE, run_id="H_LR_LOW", lr=3e-4,
            varying_parameter="lr", value=3e-4),
    replace(BASELINE, run_id="H_LR_HIGH", lr=3e-3,
            varying_parameter="lr", value=3e-3),
    replace(BASELINE, run_id="H_BATCH_LOW", batch_size=64,
            varying_parameter="batch_size", value=64),
    replace(BASELINE, run_id="H_BATCH_HIGH", batch_size=256,
            varying_parameter="batch_size", value=256),
    replace(BASELINE, run_id="H_TARGET_LOW", target_update=25,
            varying_parameter="target_update", value=25),
    replace(BASELINE, run_id="H_TARGET_HIGH", target_update=100,
            varying_parameter="target_update", value=100),
    replace(BASELINE, run_id="H_EPSILON_LOW", epsilon_decay=20_000,
            varying_parameter="epsilon_decay", value=20_000),
    replace(BASELINE, run_id="H_EPSILON_HIGH", epsilon_decay=40_000,
            varying_parameter="epsilon_decay", value=40_000),
)

FACTOR_RUNS = {
    "lr": ("H_LR_LOW", "H00", "H_LR_HIGH"),
    "batch_size": ("H_BATCH_LOW", "H00", "H_BATCH_HIGH"),
    "target_update": ("H_TARGET_LOW", "H00", "H_TARGET_HIGH"),
    "epsilon_decay": ("H_EPSILON_LOW", "H00", "H_EPSILON_HIGH"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled DQN hyperparameter screening and validation."
    )
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=100)
    parser.add_argument("--development-seeds", type=int, nargs="+",
                        default=list(DEVELOPMENT_SEEDS))
    parser.add_argument("--validation-seeds", type=int, nargs="+",
                        default=list(VALIDATION_SEEDS))
    parser.add_argument(
        "--results-dir", type=Path,
        default=PROJECT_ROOT / "results" / "hyperparameter_study",
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def training_args(candidate: Candidate) -> Namespace:
    return Namespace(
        hidden_dim=candidate.hidden_dim,
        gamma=1.0,
        lr=candidate.lr,
        batch_size=candidate.batch_size,
        buffer_size=100_000,
        warmup_steps=candidate.warmup_steps,
        epsilon_decay=candidate.epsilon_decay,
        target_update=candidate.target_update,
        train_freq=1,
        device=None,
    )


def run_candidate(
    candidate: Candidate, seed: int, args: argparse.Namespace, results_dir: Path
) -> None:
    run_cfg = ExperimentConfig(
        run_id=candidate.run_id, use_replay=True, use_target=True
    )
    seed_everything(seed)
    env = PrimalHuntEnv(seed=seed)
    agent_cfg = make_agent_config(env, training_args(candidate), run_cfg)
    episodes, evaluations, agent = train_loop(
        env,
        agent_cfg,
        args.episodes,
        run_cfg,
        seed,
        eval_every=args.eval_every,
        eval_episodes=args.checkpoint_eval_episodes,
        eval_seed=seed + 2_000_000,
    )
    final_evaluation = evaluate_agent(
        env.cfg, agent, episodes=args.eval_episodes, seed=seed + 1_000_000
    )
    summary = build_summary(
        episodes, agent_cfg, run_cfg, seed, final_evaluation, env.cfg
    )
    save_outputs(episodes, summary, results_dir, evaluations, agent=None)


def run_candidates(
    candidates: tuple[Candidate, ...],
    seeds: list[int],
    args: argparse.Namespace,
    results_dir: Path,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(candidate) for candidate in candidates]).to_csv(
        results_dir / "candidates.csv", index=False
    )
    for candidate in candidates:
        for seed in seeds:
            summary_path = (
                results_dir / "training" / "summaries"
                / f"{candidate.run_id}_seed{seed}_summary.json"
            )
            if summary_path.exists() and not args.force:
                print(f"Skipping completed run {candidate.run_id}, seed={seed}")
                continue
            print(f"=== {candidate.run_id}, seed={seed} ===")
            run_candidate(candidate, seed, args, results_dir)
    save_aggregate_summary(results_dir)


def build_combined_candidate(screening_dir: Path) -> tuple[Candidate, dict]:
    per_seed = pd.read_csv(screening_dir / "training" / "per_seed_summary.csv")
    means = per_seed.groupby("run_id")["eval_mean_final_energy"].mean()
    by_id = {candidate.run_id: candidate for candidate in SCREENING_CANDIDATES}
    selected_levels = {}
    values = asdict(BASELINE)
    for parameter, run_ids in FACTOR_RUNS.items():
        best_run = max(run_ids, key=lambda run_id: means.loc[run_id])
        selected_levels[parameter] = {
            "run_id": best_run,
            "mean_final_energy": float(means.loc[best_run]),
            "value": getattr(by_id[best_run], parameter),
        }
        values[parameter] = getattr(by_id[best_run], parameter)
    values.update(
        run_id="V1", varying_parameter="combined", value=0.0
    )
    return Candidate(**values), selected_levels


def select_validated_config(validation_dir: Path) -> dict:
    candidates = pd.read_csv(validation_dir / "candidates.csv")
    per_seed = pd.read_csv(validation_dir / "training" / "per_seed_summary.csv")
    quality = per_seed.groupby("run_id", as_index=False).agg(
        mean_final_energy=("eval_mean_final_energy", "mean"),
        final_energy_std=("eval_mean_final_energy", "std"),
    )
    winner = quality.sort_values(
        ["mean_final_energy", "final_energy_std", "run_id"],
        ascending=[False, True, True],
    ).iloc[0]
    selected = candidates[candidates["run_id"] == winner["run_id"]].iloc[0]
    return {
        "run_id": str(selected["run_id"]),
        "lr": float(selected["lr"]),
        "batch_size": int(selected["batch_size"]),
        "target_update": int(selected["target_update"]),
        "warmup_steps": int(selected["warmup_steps"]),
        "epsilon_decay": int(selected["epsilon_decay"]),
        "hidden_dim": int(selected["hidden_dim"]),
        "gamma": 1.0,
        "buffer_size": 100_000,
        "train_freq": 1,
        "validation_mean_final_energy": float(winner["mean_final_energy"]),
        "validation_final_energy_std": float(winner["final_energy_std"]),
    }


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    screening_dir = args.results_dir / "screening"
    validation_dir = args.results_dir / "validation"
    run_candidates(
        SCREENING_CANDIDATES, args.development_seeds, args, screening_dir
    )
    combined, selected_levels = build_combined_candidate(screening_dir)
    validation_candidates = (
        replace(BASELINE, run_id="V0", varying_parameter="validation_baseline"),
        combined,
    )
    run_candidates(
        validation_candidates, args.validation_seeds, args, validation_dir
    )
    selected = select_validated_config(validation_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "selected_config.json").write_text(
        json.dumps(selected, indent=2) + "\n"
    )
    metadata = {
        "episodes": args.episodes,
        "final_evaluation_episodes": args.eval_episodes,
        "periodic_evaluation_every": args.eval_every,
        "periodic_evaluation_episodes": args.checkpoint_eval_episodes,
        "development_seeds": args.development_seeds,
        "validation_seeds": args.validation_seeds,
        "selected_factor_levels": selected_levels,
        "selection_metric": "mean greedy-evaluation final energy",
    }
    (args.results_dir / "study_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Selected configuration: {json.dumps(selected, indent=2)}")


if __name__ == "__main__":
    main()
