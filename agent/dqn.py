from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, List, Sequence
from collections import deque, namedtuple

import numpy as np
import torch
from torch import nn, optim


Transition = namedtuple(
    "Transition", ["state", "action", "reward",
                   "next_state", "done", "next_valid_actions"]
)


@dataclass
class DQNConfig:
    obs_dim: int
    n_actions: int
    hidden_dim: int = 64
    gamma: float = 1.0
    lr: float = 1e-3
    batch_size: int = 128
    buffer_size: int = 100_000
    min_buffer_size: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 30_000  # environment steps to reach epsilon_end
    target_update_freq: int = 50  # environment steps between target syncs
    train_freq: int = 1  # environment steps between optimizer updates
    gradient_clip: float = 5.0
    use_replay: bool = True
    use_target: bool = True
    use_double_q: bool = False
    device: str | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    """Helper dataclass used in train.py to define ablations."""

    run_id: str
    use_replay: bool
    use_target: bool
    use_double_q: bool = False


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.storage: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.storage)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_valid_actions: np.ndarray,
    ) -> None:
        valid = np.array(next_valid_actions, dtype=bool)
        transition = Transition(state, action, reward, next_state, done, valid)
        self.storage.append(transition)

    def sample(self, batch_size: int, *, recent: bool = False) -> List[Transition]:
        if recent:
            return list(self.storage)[-batch_size:]

        indices = np.random.choice(
            len(self.storage), size=batch_size, replace=False)
        return [self.storage[idx] for idx in indices]


class DQNAgent:
    def __init__(self, config: DQNConfig):
        self.config = config
        device_str = config.device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)

        self.policy_net = QNetwork(
            config.obs_dim, config.n_actions, config.hidden_dim).to(self.device)
        self.target_net = QNetwork(
            config.obs_dim, config.n_actions, config.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr)
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        self.global_step = 0
        self.last_diagnostics: dict[str, float] = {}

        # When replay is disabled we want to train online as quickly as possible.
        if not self.config.use_replay:
            self.config.min_buffer_size = config.batch_size

    def act(self, obs: np.ndarray, valid_actions: Sequence[bool], *, eval_mode: bool = False) -> int:
        epsilon = 0.0 if eval_mode else self._current_epsilon()

        if not eval_mode and np.random.rand() < epsilon:
            valid_indices = np.flatnonzero(valid_actions)
            if valid_indices.size == 0:
                return int(np.random.randint(0, self.config.n_actions))
            return int(np.random.choice(valid_indices))

        obs_tensor = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(obs_tensor).squeeze(0).cpu().numpy()
        q_values = self._mask_invalid_actions(q_values, valid_actions)
        return int(np.argmax(q_values))

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_valid_actions: np.ndarray,
    ) -> None:
        self.replay_buffer.push(state, action, reward,
                                next_state, done, next_valid_actions)
        self.global_step += 1

    def learn(self) -> float | None:
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return None

        if (self.global_step % self.config.train_freq) != 0:
            return None

        recent = not self.config.use_replay
        batch = self.replay_buffer.sample(
            self.config.batch_size, recent=recent)

        batch_states = torch.tensor(
            np.stack([t.state for t in batch]), dtype=torch.float32, device=self.device)
        batch_actions = torch.tensor(
            [t.action for t in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        batch_rewards = torch.tensor(
            [t.reward for t in batch], dtype=torch.float32, device=self.device)
        batch_next_states = torch.tensor(np.stack(
            [t.next_state for t in batch]), dtype=torch.float32, device=self.device)
        batch_dones = torch.tensor(
            [t.done for t in batch], dtype=torch.float32, device=self.device)
        batch_next_valid = torch.tensor(
            np.stack([t.next_valid_actions for t in batch]),
            dtype=torch.bool,
            device=self.device,
        )

        current_q = self.policy_net(batch_states).gather(
            1, batch_actions).squeeze(1)

        with torch.no_grad():
            if self.config.use_double_q:
                online_next_q = self.policy_net(batch_next_states)
                online_next_q = self._mask_invalid_tensor(
                    online_next_q, batch_next_valid)
                best_actions = online_next_q.argmax(dim=1, keepdim=True)
                target_next_q = self.get_target_q_values(
                    batch_next_states).gather(1, best_actions).squeeze(1)
            else:
                target_next_q = self.get_target_q_values(batch_next_states)
                target_next_q = self._mask_invalid_tensor(
                    target_next_q, batch_next_valid)
                target_next_q = target_next_q.max(dim=1).values

            target_q = batch_rewards + self.config.gamma * \
                (1.0 - batch_dones) * target_next_q

        loss = nn.functional.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.config.gradient_clip
        )
        self.optimizer.step()

        if self.config.use_target and (self.global_step % self.config.target_update_freq == 0):
            self.target_net.load_state_dict(self.policy_net.state_dict())

        with torch.no_grad():
            weight_norm = torch.sqrt(
                sum(parameter.square().sum() for parameter in self.policy_net.parameters())
            )
            if self.config.use_target:
                target_distance = torch.sqrt(
                    sum(
                        (policy - target).square().sum()
                        for policy, target in zip(
                            self.policy_net.parameters(), self.target_net.parameters()
                        )
                    )
                )
                target_distance_value = float(target_distance.item())
            else:
                target_distance_value = float("nan")
            td_error = target_q - current_q
            self.last_diagnostics = {
                "gradient_norm": float(gradient_norm.item()),
                "weight_norm": float(weight_norm.item()),
                "mean_abs_q": float(current_q.abs().mean().item()),
                "max_abs_q": float(current_q.abs().max().item()),
                "mean_abs_td_error": float(td_error.abs().mean().item()),
                "target_distance": target_distance_value,
            }

        return float(loss.item())

    def get_target_q_values(self, states: torch.Tensor) -> torch.Tensor:
        if self.config.use_target:
            return self.target_net(states)
        return self.policy_net(states)

    def _mask_invalid_actions(self, q_values: np.ndarray, valid_actions: Sequence[bool]) -> np.ndarray:
        mask = np.array(valid_actions, dtype=bool)
        if mask.any():
            q_values = q_values.copy()
            q_values[~mask] = -1e9
            return q_values
        return q_values

    def _mask_invalid_tensor(self, q_values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Sets invalid action values to a large negative number before taking max."""
        large_neg = torch.finfo(q_values.dtype).min
        masked = q_values.masked_fill(~valid_mask, large_neg)
        no_valid = ~valid_mask.any(dim=1, keepdim=True)
        return torch.where(no_valid, torch.zeros_like(masked), masked)

    def _current_epsilon(self) -> float:
        fraction = min(1.0, self.global_step /
                       max(1, self.config.epsilon_decay))
        return self.config.epsilon_start + fraction * (self.config.epsilon_end - self.config.epsilon_start)

    @property
    def epsilon(self) -> float:
        return self._current_epsilon()
