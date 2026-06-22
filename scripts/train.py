import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from tqdm import trange

from agent import DQNAgent, DQNConfig, ExperimentConfig
from game_env.primal_hunt_env import PrimalHuntEnv


ABLATION_SETUPS = {
    "A": ExperimentConfig(run_id="A", use_replay=True, use_target=True),
    "B": ExperimentConfig(run_id="B", use_replay=True, use_target=False),
    "C": ExperimentConfig(run_id="C", use_replay=False, use_target=True),
    "D": ExperimentConfig(run_id="D", use_replay=False, use_target=False),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DQN agents on Primal Hunt with replay/target ablations."
    )
    parser.add_argument("--episodes", type=int, default=5_000,
                        help="Episodes per experiment run.")
    parser.add_argument("--experiment", choices=ABLATION_SETUPS.keys(),
                        default="A", help="Which ablation to run.")
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all four ablations sequentially (A-D).",
    )
    parser.add_argument(
        "--double-dqn",
        action="store_true",
        help="Include an additional Double DQN run (baseline config with double Q-learning).",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Single seed used when --seeds is not provided.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Independent seeds to run for every selected configuration.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=500,
        help="Greedy evaluation episodes after each training run.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=250,
        help="Episodes between periodic greedy evaluations; zero disables them.",
    )
    parser.add_argument(
        "--checkpoint-eval-episodes",
        type=int,
        default=100,
        help="Episodes in each periodic greedy evaluation.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Results root; training files are organized in subdirectories.",
    )

    # DQN hyper-parameters
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="Hidden size for the Q-network.")
    parser.add_argument("--gamma", type=float,
                        default=1.0, help="Discount factor.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate.")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size when replay buffer is enabled.")
    parser.add_argument("--buffer-size", type=int,
                        default=100_000, help="Replay buffer capacity.")
    parser.add_argument("--warmup-steps", type=int, default=1_000,
                        help="Steps to collect before training when replay is enabled.")
    parser.add_argument(
        "--epsilon-decay",
        type=int,
        default=30_000,
        help="Number of environment steps for epsilon to decay from start to end.",
    )
    parser.add_argument(
        "--target-update",
        type=int,
        default=50,
        help="How often (steps) to sync the target network.",
    )
    parser.add_argument(
        "--train-freq",
        type=int,
        default=1,
        help="How often (steps) to update the network.",
    )
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device to use (cpu, cuda, etc.).")
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="CPU threads used by Torch; one is efficient for this small network.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    runs = build_run_list(args)
    seeds = args.seeds if args.seeds is not None else [args.seed]
    for run_cfg in runs:
        for seed in seeds:
            seed_everything(seed)
            print(
                f"=== Running experiment {run_cfg.run_id}, seed={seed} "
                f"(replay={run_cfg.use_replay}, target={run_cfg.use_target}, "
                f"double={run_cfg.use_double_q}) ==="
            )
            env = PrimalHuntEnv(seed=seed)
            agent_cfg = make_agent_config(env, args, run_cfg)
            df, periodic_evaluations, agent = train_loop(
                env,
                agent_cfg,
                args.episodes,
                run_cfg,
                seed,
                eval_every=args.eval_every,
                eval_episodes=args.checkpoint_eval_episodes,
                eval_seed=seed + 2_000_000,
            )
            evaluation = evaluate_agent(
                env.cfg,
                agent,
                episodes=args.eval_episodes,
                seed=seed + 1_000_000,
            )
            summary = build_summary(
                df, agent_cfg, run_cfg, seed, evaluation, env.cfg
            )
            save_outputs(
                df, summary, args.results_dir, periodic_evaluations, agent
            )

    save_aggregate_summary(args.results_dir)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_run_list(args: argparse.Namespace) -> List[ExperimentConfig]:
    runs: List[ExperimentConfig] = []
    if args.run_all:
        runs.extend(ABLATION_SETUPS.values())
    else:
        runs.append(ABLATION_SETUPS[args.experiment])

    if args.double_dqn:
        runs.append(ExperimentConfig(run_id="A_DDQN",
                    use_replay=True, use_target=True, use_double_q=True))
    return runs


def make_agent_config(env: PrimalHuntEnv, args: argparse.Namespace, run_cfg: ExperimentConfig) -> DQNConfig:
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    batch_size = args.batch_size if run_cfg.use_replay else 1
    config = DQNConfig(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=batch_size,
        buffer_size=args.buffer_size,
        min_buffer_size=(
            max(args.warmup_steps, batch_size)
            if run_cfg.use_replay
            else batch_size
        ),
        epsilon_decay=args.epsilon_decay,
        target_update_freq=args.target_update,
        train_freq=args.train_freq,
        use_replay=run_cfg.use_replay,
        use_target=run_cfg.use_target,
        use_double_q=run_cfg.use_double_q,
        device=args.device,
    )
    return config


def train_loop(
    env: PrimalHuntEnv,
    config: DQNConfig,
    episodes: int,
    run_cfg: ExperimentConfig,
    seed: int,
    eval_every: int = 250,
    eval_episodes: int = 100,
    eval_seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, DQNAgent]:
    agent = DQNAgent(config)
    episode_records = []
    evaluation_records = []
    if eval_every > 0:
        evaluation_records.append(
            {
                "run_id": run_cfg.run_id,
                "seed": seed,
                "episode": 0,
                **evaluate_agent(
                    env.cfg,
                    agent,
                    episodes=eval_episodes,
                    seed=eval_seed if eval_seed is not None else seed + 2_000_000,
                ),
            }
        )
    step_iterator = trange(
        1, episodes + 1, desc=f"Run {run_cfg.run_id}", leave=False)

    for episode in step_iterator:
        obs, info = env.reset()
        done = False
        objective_return = 0.0
        ep_losses: List[float] = []
        diagnostics: dict[str, list[float]] = {}
        step_count = 0

        while not done:
            valid_actions = info["valid_actions"]
            action = agent.act(obs, valid_actions)
            next_obs, reward, terminated, truncated, next_info = env.step(
                action)
            done = terminated or truncated
            agent.remember(obs, action, reward, next_obs,
                           done, next_info["valid_actions"])
            loss = agent.learn()
            if loss is not None:
                ep_losses.append(loss)
                for name, value in agent.last_diagnostics.items():
                    if np.isfinite(value):
                        diagnostics.setdefault(name, []).append(value)

            obs = next_obs
            info = next_info
            objective_return += reward
            step_count += 1

        final_energy = float(info["cumulative_energy"])
        survived = int(bool(info["survived"]))
        mean_loss = float(np.mean(ep_losses)) if ep_losses else np.nan
        episode_records.append(
            {
                "run_id": run_cfg.run_id,
                "seed": seed,
                "episode": episode,
                "objective_return": objective_return,
                "final_energy": final_energy,
                "survived": survived,
                "steps": step_count,
                "mean_td_loss": mean_loss,
                "mean_gradient_norm": _mean_or_nan(
                    diagnostics.get("gradient_norm", [])
                ),
                "max_gradient_norm": _max_or_nan(
                    diagnostics.get("gradient_norm", [])
                ),
                "weight_norm": _last_or_nan(diagnostics.get("weight_norm", [])),
                "mean_abs_q": _mean_or_nan(diagnostics.get("mean_abs_q", [])),
                "max_abs_q": _max_or_nan(diagnostics.get("max_abs_q", [])),
                "mean_abs_td_error": _mean_or_nan(
                    diagnostics.get("mean_abs_td_error", [])
                ),
                "target_distance": _last_or_nan(
                    diagnostics.get("target_distance", [])
                ),
                "epsilon": agent.epsilon,
            }
        )
        if episode % 100 == 0 or episode == episodes:
            step_iterator.set_postfix(
                {
                    "survived": survived,
                    "energy": f"{final_energy:.1f}",
                    "loss": f"{mean_loss:.4f}" if ep_losses else "nan",
                }
            )
        if eval_every > 0 and episode % eval_every == 0:
            evaluation_records.append(
                {
                    "run_id": run_cfg.run_id,
                    "seed": seed,
                    "episode": episode,
                    **evaluate_agent(
                        env.cfg,
                        agent,
                        episodes=eval_episodes,
                        seed=eval_seed if eval_seed is not None else seed + 2_000_000,
                    ),
                }
            )

    df = pd.DataFrame(episode_records)
    evaluations = pd.DataFrame(evaluation_records)
    return df, evaluations, agent


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else np.nan


def _max_or_nan(values: list[float]) -> float:
    return float(np.max(values)) if values else np.nan


def _last_or_nan(values: list[float]) -> float:
    return float(values[-1]) if values else np.nan


def evaluate_agent(
    env_config,
    agent: DQNAgent,
    episodes: int,
    seed: int,
) -> dict:
    env = PrimalHuntEnv(config=env_config, seed=seed)
    objective_returns = []
    final_energies = []
    survived = []

    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        done = False
        objective_return = 0.0
        while not done:
            action = agent.act(
                obs, info["valid_actions"], eval_mode=True
            )
            obs, reward, terminated, truncated, info = env.step(action)
            objective_return += reward
            done = terminated or truncated
        final_energy = float(info["cumulative_energy"])
        objective_returns.append(objective_return)
        final_energies.append(final_energy)
        survived.append(float(info["survived"]))

    survivor_energies = [
        energy for energy, did_survive in zip(final_energies, survived)
        if did_survive
    ]
    failure_energies = [
        energy for energy, did_survive in zip(final_energies, survived)
        if not did_survive
    ]

    def optional_mean(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    return {
        "episodes": episodes,
        "mean_objective_return": float(np.mean(objective_returns)),
        "objective_return_std": float(np.std(objective_returns)),
        "survival_rate": float(np.mean(survived)),
        "mean_final_energy": float(np.mean(final_energies)),
        "final_energy_std": float(np.std(final_energies)),
        "mean_survivor_energy": optional_mean(survivor_energies),
        "mean_failure_energy": optional_mean(failure_energies),
    }


def build_summary(
    df: pd.DataFrame,
    config: DQNConfig,
    run_cfg: ExperimentConfig,
    seed: int,
    evaluation: dict,
    environment_config,
) -> dict:
    td_mean = float(df["mean_td_loss"].dropna().mean()
                    ) if df["mean_td_loss"].notna().any() else None
    summary = {
        "run_id": run_cfg.run_id,
        "seed": seed,
        "episodes": int(df["episode"].max()),
        "mean_objective_return": float(df["objective_return"].mean()),
        "mean_final_energy": float(df["final_energy"].mean()),
        "survival_rate": float(df["survived"].mean()),
        "td_loss_mean": td_mean,
        "evaluation": evaluation,
        "config": asdict(config),
        "environment": asdict(environment_config),
        "experiment": {
            "use_replay": run_cfg.use_replay,
            "use_target": run_cfg.use_target,
            "use_double_q": run_cfg.use_double_q,
        },
    }
    return summary


def save_outputs(
    df: pd.DataFrame,
    summary: dict,
    results_dir: Path,
    periodic_evaluations: pd.DataFrame | None = None,
    agent: DQNAgent | None = None,
) -> None:
    run_id = summary["run_id"]
    seed = summary["seed"]
    episodes_dir = results_dir / "training" / "episodes"
    summaries_dir = results_dir / "training" / "summaries"
    evaluations_dir = results_dir / "training" / "evaluations"
    checkpoints_dir = results_dir / "training" / "checkpoints"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    evaluations_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    csv_path = episodes_dir / f"{run_id}_seed{seed}_episodes.csv"
    df.to_csv(csv_path, index=False)

    summary_path = summaries_dir / f"{run_id}_seed{seed}_summary.json"
    with summary_path.open("w") as fp:
        json.dump(summary, fp, indent=2, default=_json_default)
    if periodic_evaluations is not None and not periodic_evaluations.empty:
        periodic_evaluations.to_csv(
            evaluations_dir / f"{run_id}_seed{seed}_evaluations.csv", index=False
        )
    if agent is not None:
        torch.save(
            {
                "policy_state_dict": agent.policy_net.state_dict(),
                "agent_config": asdict(agent.config),
                "run_id": run_id,
                "seed": seed,
            },
            checkpoints_dir / f"{run_id}_seed{seed}.pt",
        )
    print(f"Wrote {csv_path} and {summary_path}")


def save_aggregate_summary(results_dir: Path) -> None:
    summaries = []
    summaries_dir = results_dir / "training" / "summaries"
    for summary_path in sorted(summaries_dir.glob("*_summary.json")):
        with summary_path.open() as fp:
            summary = json.load(fp)
        if "mean_objective_return" in summary and "evaluation" in summary:
            summaries.append(summary)

    rows = []
    for summary in summaries:
        evaluation = summary["evaluation"]
        rows.append(
            {
                "run_id": summary["run_id"],
                "seed": summary["seed"],
                "train_mean_objective_return": summary["mean_objective_return"],
                "train_survival_rate": summary["survival_rate"],
                "train_mean_final_energy": summary["mean_final_energy"],
                "eval_mean_objective_return": evaluation["mean_objective_return"],
                "eval_survival_rate": evaluation["survival_rate"],
                "eval_mean_final_energy": evaluation["mean_final_energy"],
                "eval_final_energy_std": evaluation["final_energy_std"],
                "eval_mean_survivor_energy": evaluation["mean_survivor_energy"],
                "eval_mean_failure_energy": evaluation["mean_failure_energy"],
            }
        )

    per_seed = pd.DataFrame(rows)
    aggregate = per_seed.groupby("run_id", as_index=False).agg(
        seeds=("seed", "count"),
        eval_mean_survival_rate=("eval_survival_rate", "mean"),
        eval_survival_between_seed_std=("eval_survival_rate", "std"),
        eval_mean_final_energy=("eval_mean_final_energy", "mean"),
        eval_final_energy_between_seed_std=("eval_mean_final_energy", "std"),
    )
    output_dir = results_dir / "training"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_dir / "per_seed_summary.csv", index=False)
    aggregate.to_csv(output_dir / "ablation_summary.csv", index=False)


def _json_default(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable")


if __name__ == "__main__":
    main()
