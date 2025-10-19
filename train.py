import argparse
import os
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from game_env.primal_hunt_env import PrimalHuntEnv
from agents.dqn_agent import (
    QNet, ReplayBuffer, EpsSchedule,
    hard_update, soft_update,
    masked_argmax, masked_max
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--replay", action="store_true",
                    help="enable replay buffer")
    ap.add_argument("--target", action="store_true",
                    help="enable target network")
    ap.add_argument("--soft_tau", type=float, default=0.0,
                    help=">0 to use soft target updates")
    ap.add_argument("--target_update_every", type=int, default=500)
    ap.add_argument("--buffer_size", type=int, default=50_000)
    ap.add_argument("--warmup", type=int, default=1_000)
    ap.add_argument("--eval_every", type=int, default=10_000)
    ap.add_argument("--log_every", type=int, default=1_000)
    ap.add_argument("--logdir", type=str, default="runs/primal_hunt")
    return ap.parse_args()


def make_env(seed):
    env = PrimalHuntEnv()
    env.seed(seed)
    return env


def evaluate(env, qnet, episodes=10, device="cpu"):
    qnet.eval()
    returns = []
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            with torch.no_grad():
                q = qnet(torch.from_numpy(obs).to(device))
                a = masked_argmax(q, info["action_mask"])
            obs, r, done, trunc, info = env.step(a)
            ep_ret += r
        returns.append(ep_ret)
    qnet.train()
    return float(np.mean(returns)), float(np.std(returns))


def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = make_env(args.seed)
    obs, info = env.reset()

    state_dim = len(obs)
    n_actions = 4

    q = QNet(state_dim, n_actions).to(device)
    q_target = QNet(state_dim, n_actions).to(device)
    hard_update(q_target, q)

    opt = optim.Adam(q.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()

    eps_sched = EpsSchedule()
    rb = ReplayBuffer(args.buffer_size, state_dim) if args.replay else None

    step = 0
    episode = 0
    ep_return = 0.0
    returns_history = deque(maxlen=100)

    # for logging
    log_path = os.path.join(args.logdir, "train_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("step,episode,ep_return,loss,epsilon,eval_mean,eval_std\n")

    while step < args.steps:
        # --- episode start ---
        obs, info = env.reset()
        done = False
        ep_return = 0.0

        while not done and step < args.steps:
            epsilon = eps_sched.value(step)
            # ε-greedy over valid actions
            if np.random.rand() < epsilon:
                valid_idxs = np.flatnonzero(info["action_mask"])
                action = int(np.random.choice(valid_idxs))
            else:
                with torch.no_grad():
                    q_vals = q(torch.from_numpy(obs).to(device))
                action = masked_argmax(q_vals, info["action_mask"])

            next_obs, reward, done, trunc, next_info = env.step(action)

            # store transition
            if rb is not None:
                rb.push(obs, action, reward, next_obs, float(done))

            ep_return += reward
            step += 1

            # learn
            loss_val = 0.0
            can_learn = (rb is not None and len(rb) >= max(
                args.batch_size, args.warmup)) or (rb is None)
            if can_learn:
                if rb is not None:
                    batch = rb.sample(args.batch_size)
                    s_b, a_b, r_b, s2_b, d_b = [t.to(device) for t in batch]
                else:
                    # online update using the most recent transition (tiny batch of 1)
                    s_b = torch.from_numpy(np.asarray(
                        [obs], dtype=np.float32)).to(device)
                    a_b = torch.tensor(
                        [action], dtype=torch.int64, device=device)
                    r_b = torch.tensor(
                        [reward], dtype=torch.float32, device=device)
                    s2_b = torch.from_numpy(np.asarray(
                        [next_obs], dtype=np.float32)).to(device)
                    d_b = torch.tensor(
                        [float(done)], dtype=torch.float32, device=device)

                # Q(s,a)
                q_sa = q(s_b).gather(1, a_b.view(-1, 1)).squeeze(1)

                # target r + gamma * max_a' Q_target(s',a') over valid actions
                with torch.no_grad():
                    if args.target:
                        q_next_all = q_target(s2_b)
                    else:
                        q_next_all = q(s2_b)

                    # Build masks for each next state (derive from obs one-hot position)
                    # obs layout: [25 one-hot][steps_left][cum_energy]
                    def obs_to_pos(o):
                        idx = int(np.argmax(o[:25]))
                        return idx // 5, idx % 5

                    masks = []
                    s2_np = s2_b.cpu().numpy()
                    for o in s2_np:
                        r2, c2 = obs_to_pos(o)
                        mask = np.array(
                            [r2 > 0, r2 < 4, c2 > 0, c2 < 4], dtype=bool)
                        masks.append(mask)
                    masks = np.stack(masks, axis=0)

                    max_next = masked_max(q_next_all, masks)
                    target = r_b + (1.0 - d_b) * args.gamma * max_next

                loss = criterion(q_sa, target)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()
                loss_val = float(loss.item())

                # target update
                if args.target:
                    if args.soft_tau > 0.0:
                        soft_update(q_target, q, tau=args.soft_tau)
                    elif step % args.target_update_every == 0:
                        hard_update(q_target, q)

            # advance
            obs, info = next_obs, next_info

            # logging
            if step % args.log_every == 0:
                eval_mean = eval_std = ""
                if args.eval_every > 0 and step % args.eval_every == 0:
                    eval_mean, eval_std = evaluate(
                        env, q, episodes=10, device=device)

                with open(log_path, "a") as f:
                    f.write(
                        f"{step},{episode},{ep_return:.4f},{loss_val},{epsilon:.4f},{eval_mean},{eval_std}\n")

        episode += 1

    print(f"Training finished. Logs saved to: {log_path}")


if __name__ == "__main__":
    main()
