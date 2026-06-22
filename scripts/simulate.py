import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game_env.primal_hunt_env import PrimalHuntEnv


STEP_FIELDS = [
    "episode",
    "step",
    "action",
    "valid_action",
    "objective_reward",
    "food",
    "effort",
    "injury",
    "energy_delta",
    "row",
    "col",
    "steps_left",
    "cell_type",
    "cumulative_energy",
    "terminated",
    "truncated",
]

EPISODE_FIELDS = [
    "episode",
    "objective_return",
    "final_energy",
    "survived",
    "steps",
    "final_row",
    "final_col",
]


def decode_position(obs: np.ndarray, grid_size: int) -> tuple[int, int]:
    """Return (row, col) from one-hot encoded position in the observation."""
    idx = int(np.argmax(obs[: grid_size * grid_size]))
    return divmod(idx, grid_size)


def select_action(valid_actions: Iterable[bool], rng: np.random.Generator) -> int:
    """Randomly select one of the valid moves."""
    mask = np.array(valid_actions, dtype=bool)
    valid_indices = np.flatnonzero(mask)
    if valid_indices.size == 0:
        return int(rng.integers(0, 4))
    return int(rng.choice(valid_indices))


def run_simulation(num_episodes: int, steps_csv: Path, episodes_csv: Path, seed: int | None = None) -> None:
    steps_csv.parent.mkdir(parents=True, exist_ok=True)
    episodes_csv.parent.mkdir(parents=True, exist_ok=True)
    env = PrimalHuntEnv(seed=seed)
    rng = np.random.default_rng(seed)
    grid_size = env.cfg.grid_size

    with steps_csv.open("w", newline="") as steps_file, episodes_csv.open("w", newline="") as episodes_file:
        step_writer = csv.DictWriter(steps_file, fieldnames=STEP_FIELDS)
        episodes_writer = csv.DictWriter(
            episodes_file, fieldnames=EPISODE_FIELDS)
        step_writer.writeheader()
        episodes_writer.writeheader()

        for episode in range(1, num_episodes + 1):
            obs, info = env.reset()
            objective_return = 0.0
            steps_taken = 0

            while True:
                valid_actions = info["valid_actions"]
                action = select_action(valid_actions, rng)
                next_obs, reward, terminated, truncated, info = env.step(
                    action)
                steps_taken += 1
                objective_return += reward

                row, col = decode_position(next_obs, grid_size)
                steps_left = int(info["steps_left"])
                cumulative_energy = float(info["cumulative_energy"])
                cell_type = int(info["cell_type"])

                step_writer.writerow(
                    {
                        "episode": episode,
                        "step": steps_taken,
                        "action": action,
                        "valid_action": bool(valid_actions[action]),
                        "objective_reward": reward,
                        "food": info["food"],
                        "effort": info["effort"],
                        "injury": info["injury"],
                        "energy_delta": info["energy_delta"],
                        "row": row,
                        "col": col,
                        "steps_left": steps_left,
                        "cell_type": cell_type,
                        "cumulative_energy": cumulative_energy,
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                )

                obs = next_obs
                if terminated or truncated:
                    break

            final_row, final_col = decode_position(obs, grid_size)
            episodes_writer.writerow(
                {
                    "episode": episode,
                    "objective_return": objective_return,
                    "final_energy": float(info["cumulative_energy"]),
                    "survived": bool(info["survived"]),
                    "steps": steps_taken,
                    "final_row": final_row,
                    "final_col": final_col,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play PrimalHuntEnv episodes and log steps and episodes."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10_000,
        help="Number of episodes to simulate.",
    )
    parser.add_argument(
        "--steps-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "simulation" / "steps.csv",
        help="Output CSV path for per-step logs.",
    )
    parser.add_argument(
        "--episodes-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "simulation" / "episodes.csv",
        help="Output CSV path for per-episode summaries.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for numpy RNG and environment.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_simulation(
        num_episodes=args.episodes,
        steps_csv=args.steps_csv,
        episodes_csv=args.episodes_csv,
        seed=args.seed,
    )
