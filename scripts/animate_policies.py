import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import DQNAgent, DQNConfig
from game_env.primal_hunt_env import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    PrimalHuntEnv,
)


ACTION_NAMES = {
    ACTION_LEFT: "left",
    ACTION_UP: "up",
    ACTION_RIGHT: "right",
    ACTION_DOWN: "down",
}
CELL_TEXT = {0: "H", 1: "V", 2: "S", 3: "B", 4: "O", 5: ""}
CELL_COLORS = ["#f2d16b", "#79b96b", "#a7d6f2", "#d6845f", "#777777", "#eeeeee"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate final greedy A-D policies.")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "results" / "animations")
    parser.add_argument("--training-seed", type=int, default=303)
    parser.add_argument("--evaluation-seed", type=int, default=3_030_303)
    parser.add_argument("--fps", type=int, default=1)
    return parser.parse_args()


def load_agent(checkpoint_path: Path) -> DQNAgent:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_values = checkpoint["agent_config"]
    config_values["device"] = "cpu"
    agent = DQNAgent(DQNConfig(**config_values))
    agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
    agent.policy_net.eval()
    return agent


def collect_episode(agent: DQNAgent, evaluation_seed: int) -> tuple[PrimalHuntEnv, list[dict]]:
    env = PrimalHuntEnv(seed=evaluation_seed)
    obs, info = env.reset(seed=evaluation_seed)
    records = [
        {
            "step": 0,
            "position": env._pos,
            "action": None,
            "reward": 0.0,
            "energy": 0.0,
            "survived": None,
        }
    ]
    done = False
    while not done:
        action = agent.act(obs, info["valid_actions"], eval_mode=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        records.append(
            {
                "step": len(records),
                "position": env._pos,
                "action": action,
                "reward": float(reward),
                "energy": float(info["cumulative_energy"]),
                "survived": info["survived"],
            }
        )
    return env, records


def create_animation(
    env: PrimalHuntEnv,
    records: list[dict],
    run_id: str,
    training_seed: int,
    evaluation_seed: int,
    fps: int,
):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    grid = env._grid

    def draw(frame_index: int) -> None:
        ax.clear()
        ax.imshow(grid, cmap=ListedColormap(CELL_COLORS), vmin=0, vmax=5)
        ax.set_xticks(np.arange(-0.5, env.cfg.grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, env.cfg.grid_size, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="both", bottom=False, left=False,
                       labelbottom=False, labelleft=False)
        for row in range(env.cfg.grid_size):
            for col in range(env.cfg.grid_size):
                ax.text(col, row, CELL_TEXT[int(grid[row, col])],
                        ha="center", va="center", fontweight="bold")
        path = [record["position"] for record in records[: frame_index + 1]]
        rows = [position[0] for position in path]
        cols = [position[1] for position in path]
        ax.plot(cols, rows, "o-", color="#6a1b9a", linewidth=3, markersize=8)
        record = records[frame_index]
        action = "start" if record["action"] is None else ACTION_NAMES[record["action"]]
        status = ""
        if record["survived"] is not None:
            status = " | survived" if record["survived"] else " | did not survive"
        ax.set_title(
            f"{run_id}, training seed {training_seed}\n"
            f"step {record['step']}: {action} | reward {record['reward']:+.1f} | "
            f"energy {record['energy']:+.1f}{status}\n"
            f"evaluation seed {evaluation_seed}",
            fontsize=10,
        )
        ax.set_xlim(-0.5, env.cfg.grid_size - 0.5)
        ax.set_ylim(env.cfg.grid_size - 0.5, -0.5)

    movie = animation.FuncAnimation(
        fig, draw, frames=len(records), interval=1000 / max(1, fps)
    )
    return movie


def build_policy_animation(
    checkpoint_path: Path,
    run_id: str,
    training_seed: int,
    evaluation_seed: int,
    fps: int = 1,
):
    agent = load_agent(checkpoint_path)
    env, records = collect_episode(agent, evaluation_seed)
    return create_animation(
        env, records, run_id, training_seed, evaluation_seed, fps
    )


def save_animation(
    env: PrimalHuntEnv,
    records: list[dict],
    run_id: str,
    training_seed: int,
    evaluation_seed: int,
    output_path: Path,
    fps: int,
) -> None:
    movie = create_animation(
        env, records, run_id, training_seed, evaluation_seed, fps
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    movie.save(output_path, writer=animation.PillowWriter(fps=fps))
    plt.close(movie._fig)


def main() -> None:
    args = parse_args()
    for run_id in ("A", "B", "C", "D"):
        checkpoint = (
            args.results_dir / "training" / "checkpoints"
            / f"{run_id}_seed{args.training_seed}.pt"
        )
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint {checkpoint}")
        agent = load_agent(checkpoint)
        env, records = collect_episode(agent, args.evaluation_seed)
        output_path = args.output_dir / f"policy_{run_id}_seed{args.training_seed}.gif"
        save_animation(
            env, records, run_id, args.training_seed, args.evaluation_seed,
            output_path, args.fps,
        )
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
