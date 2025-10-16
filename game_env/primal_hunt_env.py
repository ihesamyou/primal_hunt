import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional


# ----------------------------
# Cell type tags (for clarity)
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
    episode_len: int = 10
    # Fixed coordinates (row, col), 0-based
    home: Tuple[int, int] = (2, 2)
    veg_fruit: Tuple[Tuple[int, int], ...] = ((0, 4), (1, 1), (2, 0), (3, 3))
    small_animals: Tuple[Tuple[int, int], ...] = ((0, 1), (1, 3), (4, 0))
    big_animals: Tuple[Tuple[int, int], ...] = ((4, 1), (4, 4))
    obstacles: Tuple[Tuple[int, int], ...] = ((1, 0), (2, 4), (3, 1), (3, 2))
    # Reward scaling used for observation normalization only
    energy_norm: float = 20.0


class PrimalHuntEnv(gym.Env):
    """
    5x5 grid, 10-step episodes.
    Observation = [one-hot position (25), steps_left/10, cumulative_energy/20] -> shape (27,)
    Reward per step: ΔE = Food - (Effort + Injury), sampled per cell type.
    No early termination, no terminal bonus. Action masking provided in info['action_mask'].
    """
    metadata = {"render_modes": []}

    def __init__(self, config: Optional[Config] = None, seed: Optional[int] = None):
        super().__init__()
        self.cfg = config or Config()
        self.rng: np.random.Generator = np.random.default_rng(seed)

        # Action/observation spaces
        self.action_space = spaces.Discrete(4)  # L, U, R, D
        # 25 one-hot + 2 scalars
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32)

        # Build the grid of cell types
        self._grid = np.full(
            (self.cfg.grid_size, self.cfg.grid_size), EMPTY, dtype=np.int32)
        self._place_cells()

        # Episode state
        self._pos: Tuple[int, int] = self.cfg.home
        self._steps_left: int = self.cfg.episode_len
        self._cum_energy: float = 0.0

    # ------------- Gym API -------------

    def seed(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed(seed)
        self._pos = self.cfg.home
        self._steps_left = self.cfg.episode_len
        self._cum_energy = 0.0
        obs = self._build_obs()
        info = {"action_mask": self._action_mask(self._pos)}
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action index."
        # Mask invalid moves (off-grid). If invalid is chosen, we simply stay in place and count the step.
        valid_mask = self._action_mask(self._pos)
        if not valid_mask[action]:
            next_pos = self._pos  # stays
        else:
            next_pos = self._move(self._pos, action)

        # Sample per-cell stochastic quantities on ENTERING the cell
        cell_type = self._grid[next_pos]
        F, C, I = self._sample_FCI(cell_type)
        reward = float(F - (C + I))
        self._cum_energy += reward

        self._pos = next_pos
        self._steps_left -= 1

        terminated = (self._steps_left == 0)
        truncated = False

        obs = self._build_obs()
        info = {"action_mask": self._action_mask(self._pos)}
        return obs, reward, terminated, truncated, info

    # ------------- Helpers -------------

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
        one_hot = np.zeros(self.cfg.grid_size *
                           self.cfg.grid_size, dtype=np.float32)
        idx = self._pos[0] * self.cfg.grid_size + self._pos[1]
        one_hot[idx] = 1.0
        steps_feat = np.array(
            [self._steps_left / self.cfg.episode_len], dtype=np.float32)
        energy_feat = np.array(
            [self._cum_energy / self.cfg.energy_norm], dtype=np.float32)
        return np.concatenate([one_hot, steps_feat, energy_feat], axis=0)

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

        # Clamp inside grid (we only call _move if valid; this is a safeguard)
        r = int(np.clip(r, 0, self.cfg.grid_size - 1))
        c = int(np.clip(c, 0, self.cfg.grid_size - 1))
        return (r, c)

    def _action_mask(self, pos: Tuple[int, int]) -> np.ndarray:
        """Boolean mask over [U, D, L, R]; True = valid from current cell."""
        r, c = pos
        n = self.cfg.grid_size
        valid = np.array([
            c > 0,           # Left
            r > 0,           # Up
            c < n - 1,       # Right
            r < n - 1,       # Down
        ], dtype=bool)
        return valid

    # ---------- Stochastic rewards ----------

    def _sample_FCI(self, cell_type: int) -> Tuple[float, float, float]:
        """
        Returns Food F, Effort C, Injury I (I applied even if 0-probability).
        Distributions per our agreed spec (unbounded, positive).
        """
        # Convenience
        beta = self.rng.beta
        gamma = self.rng.gamma
        lognormal = self.rng.lognormal
        uniform = self.rng.uniform

        if cell_type == VEG_FRUIT:
            F = float(lognormal(mean=np.log(2.0), sigma=0.35))
            C = float(gamma(shape=3.0, scale=0.3))
            I = float(gamma(shape=2.0, scale=1.0)) if (
                uniform() < 0.03) else 0.0
            return F, C, I

        if cell_type == SMALL_ANIM:
            success = (uniform() < 0.65)
            F = float(lognormal(mean=np.log(5.0), sigma=0.4)
                      ) if success else 0.0
            C = float(gamma(shape=3.0, scale=0.7))
            I = float(gamma(shape=2.0, scale=2.0)) if (
                uniform() < 0.12) else 0.0
            return F, C, I

        if cell_type == BIG_ANIM:
            success = (uniform() < 0.35)
            F = float(lognormal(mean=np.log(12.0), sigma=0.45)
                      ) if success else 0.0
            C = float(gamma(shape=4.0, scale=1.2))
            I = float(gamma(shape=3.0, scale=3.0)) if (
                uniform() < 0.30) else 0.0
            return F, C, I

        if cell_type == OBSTACLE:
            F = 0.0
            C = float(gamma(shape=4.0, scale=1.5))
            I = float(gamma(shape=2.0, scale=3.0)) if (
                uniform() < 0.18) else 0.0
            return F, C, I

        if cell_type == EMPTY:
            F = 0.0
            C = float(gamma(shape=2.0, scale=0.25))
            I = float(gamma(shape=2.0, scale=1.0)) if (
                uniform() < 0.02) else 0.0
            return F, C, I

        if cell_type == HOME:
            # Home has no food/cost by default (you can change later)
            return 0.0, 0.0, 0.0

        # Fallback (should never hit)
        return 0.0, 0.0, 0.0


if __name__ == "__main__":
    env = PrimalHuntEnv()
    obs, info = env.reset(seed=42)
    print("obs shape:", obs.shape)
    print("mask at start:", info["action_mask"])

    ep_return = 0.0
    for t in range(10):
        # sample only valid actions
        valid = np.flatnonzero(info["action_mask"])
        a = np.random.choice(valid)
        obs, r, term, trunc, info = env.step(a)
        ep_return += r
        print(f"t={t:02d} a={a} r={r:.2f} done={term}")

    print("episode return:", ep_return)
