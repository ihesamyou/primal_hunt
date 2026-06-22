import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Tuple, Optional


# ----------------------------
# Cell type tags
# ----------------------------
HOME = 0
VEG_FRUIT = 1
SMALL_ANIM = 2
BIG_ANIM = 3
OBSTACLE = 4
EMPTY = 5

# Actions: Left, Up, Right, Down
ACTION_LEFT, ACTION_UP, ACTION_RIGHT, ACTION_DOWN = 0, 1, 2, 3
ACTIONS = [ACTION_LEFT, ACTION_UP, ACTION_RIGHT, ACTION_DOWN]


@dataclass(frozen=True)
class Config:
    grid_size: int = 5
    episode_len: int = 6
    survival_threshold: float = 0.0
    observation_energy_scale: float = 100.0
    # Fixed coordinates (row, col), 0-based
    home: Tuple[int, int] = (2, 2)
    veg_fruit: Tuple[Tuple[int, int], ...] = ((0, 4), (1, 1), (2, 0), (3, 3))
    small_animals: Tuple[Tuple[int, int], ...] = ((0, 1), (1, 3), (4, 0))
    big_animals: Tuple[Tuple[int, int], ...] = ((4, 1), (4, 4))
    obstacles: Tuple[Tuple[int, int], ...] = ((1, 0), (2, 4), (3, 1), (3, 2))


class PrimalHuntEnv(gym.Env):
    """
    5x5 grid, 6-step episodes.
    Observation = [one-hot position (25), visited mask (25), normalized
    steps_left, scaled cumulative_energy] -> shape (52,).
    Reward and energy change are Food - Effort - Injury at every step. After six
    steps, the episode ends without an additional terminal reward or penalty.
    The agent cannot revisit cells. The objective is expected final energy;
    survival (final energy above the threshold) is a secondary metric.
    """

    def __init__(self, config: Optional[Config] = None, seed: Optional[int] = None):
        super().__init__()
        self.cfg = config or Config()
        self.rng: np.random.Generator = np.random.default_rng(seed)

        # Action/observation spaces
        self.action_space = spaces.Discrete(4)  # L, U, R, D
        observation_dim = 2 * self.grid_cells + 2
        low = np.concatenate(
            [
                np.zeros(observation_dim - 1, dtype=np.float32),
                np.array([-np.inf], dtype=np.float32),
            ]
        )
        high = np.concatenate(
            [
                np.ones(observation_dim - 1, dtype=np.float32),
                np.array([np.inf], dtype=np.float32),
            ]
        )
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

        # Build the grid of cell types
        self._grid = np.full(
            (self.cfg.grid_size, self.cfg.grid_size), EMPTY, dtype=np.int32)
        self._place_cells()

        # Episode state
        self._pos: Tuple[int, int] = self.cfg.home
        self._steps_left: int = self.cfg.episode_len
        self._cum_energy: float = 0.0

    @property
    def grid_cells(self) -> int:
        return self.cfg.grid_size * self.cfg.grid_size

    @property
    def visited_slice(self) -> slice:
        return slice(self.grid_cells, 2 * self.grid_cells)

    @property
    def steps_index(self) -> int:
        return 2 * self.grid_cells

    @property
    def energy_index(self) -> int:
        return self.steps_index + 1

    def seed(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        self._pos = self.cfg.home
        self._steps_left = self.cfg.episode_len
        self._cum_energy = 0.0
        # Track visited cells as (row, col) tuples
        self.visited = set()
        self.visited.add(self._pos)
        obs = self._build_obs()
        info = self._build_info(
            energy_delta=0.0,
            survived=None,
        )
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action index."
        valid_actions = self.filter_valid_actions(self._pos)
        if not valid_actions[action]:
            next_pos = self._pos
        else:
            next_pos = self._move(self._pos, action)

        # Track visited cells
        self.visited.add(next_pos)

        cell_type = self._grid[next_pos]
        F, C, I = self._sample_FCI(cell_type)
        energy_delta = float(F - (C + I))
        self._cum_energy += energy_delta

        self._pos = next_pos
        self._steps_left -= 1

        terminated = (self._steps_left <= 0)
        truncated = False
        survived = self._cum_energy > self.cfg.survival_threshold
        reward = energy_delta

        obs = self._build_obs()
        info = self._build_info(
            energy_delta=energy_delta,
            survived=survived if terminated else None,
            food=F,
            effort=C,
            injury=I,
        )
        return obs, reward, terminated, truncated, info

    def _build_info(
        self,
        energy_delta: float,
        survived: Optional[bool],
        food: float = 0.0,
        effort: float = 0.0,
        injury: float = 0.0,
    ) -> dict:
        return {
            "valid_actions": self.filter_valid_actions(self._pos),
            "food": float(food),
            "effort": float(effort),
            "injury": float(injury),
            "energy_delta": float(energy_delta),
            "cumulative_energy": float(self._cum_energy),
            "steps_left": int(self._steps_left),
            "cell_type": int(self._grid[self._pos]),
            "survived": survived,
        }

    def _place_cells(self):
        self._grid[:, :] = EMPTY
        self._grid[self.cfg.home] = HOME
        for r, c in self.cfg.veg_fruit:
            self._grid[r, c] = VEG_FRUIT
        for r, c in self.cfg.small_animals:
            self._grid[r, c] = SMALL_ANIM
        for r, c in self.cfg.big_animals:
            self._grid[r, c] = BIG_ANIM
        for r, c in self.cfg.obstacles:
            self._grid[r, c] = OBSTACLE

    def _build_obs(self) -> np.ndarray:
        one_hot = np.zeros(self.grid_cells, dtype=np.float32)
        idx = self._pos[0] * self.cfg.grid_size + self._pos[1]
        one_hot[idx] = 1.0
        visited = np.zeros(self.grid_cells, dtype=np.float32)
        for row, col in self.visited:
            visited[row * self.cfg.grid_size + col] = 1.0
        steps_feat = np.array(
            [self._steps_left / self.cfg.episode_len], dtype=np.float32
        )
        energy_feat = np.array(
            [self._cum_energy / self.cfg.observation_energy_scale],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                one_hot,
                visited,
                steps_feat,
                energy_feat,
            ],
            axis=0,
        )

    def _move(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        r, c = pos
        if action == ACTION_LEFT:
            c -= 1
        if action == ACTION_UP:
            r -= 1
        if action == ACTION_RIGHT:
            c += 1
        if action == ACTION_DOWN:
            r += 1

        return (r, c)

    def filter_valid_actions(self, pos: Tuple[int, int]) -> np.ndarray:
        """Filter valid actions [L, U, R, D]; True = valid and leads to unvisited cell."""
        r, c = pos
        n = self.cfg.grid_size
        candidates = [
            (r, c - 1),  # Left
            (r - 1, c),  # Up
            (r, c + 1),  # Right
            (r + 1, c),  # Down
        ]
        valid = []
        for cand in candidates:
            rr, cc = cand
            # Check bounds and if not visited
            if 0 <= rr < n and 0 <= cc < n and (cand not in self.visited):
                valid.append(True)
            else:
                valid.append(False)
        return np.array(valid, dtype=bool)

    # Sampling F, C, I per cell type
    def _sample_FCI(self, cell_type: int) -> Tuple[float, float, float]:
        """
        Returns Food F, Effort C, Injury I sampled per cell type.
        Uses truncated normal distributions for all rewards/costs.

        Distribution summary:
          - VEG_FRUIT: 95% chance to get food (~5±1); effort always costs ~3±1; injuries happen 5% of the time (~3±1).
          - SMALL_ANIM: 70% chance to get food (~9±1); effort always costs ~5±1; injuries 10% of the time (~6±1).
          - BIG_ANIM: 45% chance to get food (~60±1); effort always costs ~15±1; injuries 30% of the time (~20±1).
          - OBSTACLE: never gives food; effort always costs ~5±1; injuries 25% of the time (~20±1).
          - EMPTY/HOME: food, effort, and injury are always zero.
        """
        from scipy.stats import truncnorm

        def truncated_normal(mean, std):
            a, b = (0 - mean) / std, np.inf
            return truncnorm(a, b, loc=mean, scale=std).rvs(random_state=self.rng)

        uniform = self.rng.uniform
        F, C, I = 0.0, 0.0, 0.0

        if cell_type == VEG_FRUIT:
            if uniform() < 0.95:
                F = float(truncated_normal(5, 1))
            C = float(truncated_normal(3, 1))
            if uniform() < 0.05:
                I = float(truncated_normal(3, 1))

        if cell_type == SMALL_ANIM:
            if uniform() < 0.70:
                F = float(truncated_normal(9, 1))
            C = float(truncated_normal(5, 1))
            if uniform() < 0.10:
                I = float(truncated_normal(6, 1))

        if cell_type == BIG_ANIM:
            if uniform() < 0.45:
                F = float(truncated_normal(60, 1))
            C = float(truncated_normal(15, 1))
            if uniform() < 0.30:
                I = float(truncated_normal(20, 1))

        if cell_type == OBSTACLE:
            F = 0.0
            C = float(truncated_normal(5, 1))
            if uniform() < 0.25:
                I = float(truncated_normal(20, 1))

        if cell_type == EMPTY or cell_type == HOME:
            F = 0.0
            C = 0.0
            I = 0.0

        # print(f"Cell type: {cell_type}, F: {F:.2f}, C: {C:.2f}, I: {I:.2f}")
        return F, C, I


if __name__ == "__main__":
    env = PrimalHuntEnv()
    obs, info = env.reset(seed=42)

    rng = np.random.default_rng(42)

    objective_return = 0.0
    for t in range(6):
        # sample only valid actions
        valid_actions = np.flatnonzero(info["valid_actions"])
        action = np.random.choice(valid_actions)
        obs, reward, terminated, trunctuated, info = env.step(action)
        objective_return += reward
        print(
            f"t={t} a={action} reward={reward:.2f} "
            f"energy_delta={info['energy_delta']:.2f} done={terminated}"
        )
    print("Objective return:", objective_return)
    print("Final energy:", info["cumulative_energy"])
