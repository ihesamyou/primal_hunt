import math
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Simple MLP Q-network (27->4)
# -----------------------------
class QNet(nn.Module):
    def __init__(self, state_dim: int = 27, n_actions: int = 4, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_actions)

        # Kaiming init (nice default)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


# -----------------------------
# Target network helpers
# -----------------------------
def hard_update(target: nn.Module, source: nn.Module):
    target.load_state_dict(source.state_dict())


def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.005):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1 - tau).add_(sp.data, alpha=tau)


# -----------------------------
# Uniform replay buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = int(capacity)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity,), dtype=np.int64)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.bool_)
        self.idx = 0
        self.full = False

    def push(self, s, a, r, s2, d):
        i = self.idx
        self.state[i] = s
        self.action[i] = a
        self.reward[i] = r
        self.next_state[i] = s2
        self.done[i] = d
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample(self, batch_size: int):
        n = len(self)
        idxs = np.random.randint(0, n, size=batch_size)
        batch = (
            torch.from_numpy(self.state[idxs]),
            torch.from_numpy(self.action[idxs]),
            torch.from_numpy(self.reward[idxs]),
            torch.from_numpy(self.next_state[idxs]),
            torch.from_numpy(self.done[idxs].astype(np.float32)),
        )
        return batch


# -----------------------------
# Epsilon schedule (linear)
# -----------------------------
@dataclass
class EpsSchedule:
    eps_start: float = 1.0
    eps_end: float = 0.05
    decay_steps: int = 200_000

    def value(self, t: int) -> float:
        # linear anneal
        if t >= self.decay_steps:
            return self.eps_end
        ratio = 1.0 - (t / float(self.decay_steps))
        return self.eps_end + (self.eps_start - self.eps_end) * ratio


# -----------------------------
# Action masking utilities
# -----------------------------
def masked_argmax(q: torch.Tensor, mask: np.ndarray) -> int:
    """
    q: shape [4]; mask: bool[4] where True means valid.
    Returns index of argmax over valid entries.
    """
    qm = q.clone()
    invalid = torch.tensor(~mask, dtype=torch.bool, device=qm.device)
    qm[invalid] = -1e9
    return int(torch.argmax(qm).item())


def masked_max(q_next: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    """
    q_next: shape [B,4]; mask: bool[4] OR array of shape [B,4]
    Returns tensor [B] with max over valid actions.
    """
    if q_next.dim() == 1:
        q_next = q_next.unsqueeze(0)
    if isinstance(mask, np.ndarray) and mask.ndim == 1:
        mask = np.tile(mask[None, :], (q_next.shape[0], 1))
    mask_t = torch.from_numpy(mask.astype(np.bool_)).to(q_next.device)
    q_masked = q_next.clone()
    q_masked[~mask_t] = -1e9
    return q_masked.max(dim=1).values
