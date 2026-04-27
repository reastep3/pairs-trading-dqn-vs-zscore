"""Deep Q-Network agent for pairs trading.

Theoretical grounding — Bellman optimality (Lecture #3, slides 14–18)
---------------------------------------------------------------------
Lecture 3 derives the fundamental pricing equation p = E[mx] by solving
an investor's infinite-horizon dynamic-programming problem:

    V(W_t) = max_ξ { u(c_t) + β E[V(W_{t+1})] }        (Bellman equation)

Our DQN solves the *same class* of problem for the pairs-trading agent.
The Bellman optimality equation for the action-value function Q is:

    Q*(s, a) = E[ r(s,a) + γ · max_a' Q*(s', a') ]

Mapping to the course:
    β  (subjective discount, Lecture 3)   ↔  γ = 0.99  (DQN_GAMMA)
    u(c_t)  (consumption utility)         ↔  reward    (daily P&L)
    ξ  (investment decision)              ↔  action ∈ {hold, long, short, exit}
    p = E[mx]  (pricing equation)         ↔  Q(s,a) = E[r + γ max Q(s',a')]
    Investor's optimal ξ*                 ↔  argmax_a Q(s,a)  (greedy policy)

The loss function (Huber loss on Bellman error) measures how far the
network's Q-estimates are from satisfying the Bellman equation — i.e.,
how far the agent is from the optimal trading policy.

Architecture
------------
  Input  : state vector (DQN_STATE_DIM features)
  Hidden : [128, 64] with LayerNorm + ReLU + Dropout(0.1)
  Output : Q-values for each of the 4 actions

Training
--------
  - Experience replay (circular buffer)
  - Separate target network, synced every DQN_TARGET_UPDATE steps
  - Epsilon-greedy exploration with exponential decay
  - Huber loss on Bellman error + gradient clipping
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import tqdm

from config import (
    DQN_STATE_DIM, DQN_ACTION_DIM, DQN_HIDDEN_DIMS,
    DQN_LR, DQN_GAMMA,
    DQN_EPSILON_START, DQN_EPSILON_MIN, DQN_EPSILON_DECAY,
    DQN_BUFFER_SIZE, DQN_BATCH_SIZE, DQN_TARGET_UPDATE,
    DQN_TRAIN_EPISODES,
)
from trading_env import PairsTradingEnv


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(np.array(s),  dtype=torch.float32),
            torch.tensor(a,            dtype=torch.long),
            torch.tensor(r,            dtype=torch.float32),
            torch.tensor(np.array(s2), dtype=torch.float32),
            torch.tensor(d,            dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class DQNNetwork(nn.Module):
    def __init__(self,
                 state_dim: int   = DQN_STATE_DIM,
                 action_dim: int  = DQN_ACTION_DIM,
                 hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = DQN_HIDDEN_DIMS

        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(0.10),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    def __init__(self, device=None):
        self.device  = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy  = DQNNetwork().to(self.device)
        self.target  = DQNNetwork().to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.policy.parameters(), lr=DQN_LR)
        self.loss_fn   = nn.SmoothL1Loss()          # Huber loss
        self.buffer    = ReplayBuffer(DQN_BUFFER_SIZE)

        self.epsilon   = DQN_EPSILON_START
        self.steps     = 0

    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, epsilon=None) -> int:
        eps = epsilon if epsilon is not None else self.epsilon
        if random.random() < eps:
            return random.randrange(DQN_ACTION_DIM)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            return int(self.policy(s).argmax(dim=1).item())

    def push(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Return Q-values for all actions given a state (no gradient)."""
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            return self.policy(s).squeeze(0).cpu().numpy()

    def update(self):
        """Minimize the Bellman error (Lecture 3, slides 14-18).

        The Bellman optimality equation:  Q*(s,a) = E[r + γ max_a' Q*(s',a')]
        The loss = |Q(s,a) - [r + γ max_a' Q_target(s',a')]|²  (Huber)
        drives Q toward the fixed-point of the Bellman operator.
        """
        if len(self.buffer) < DQN_BATCH_SIZE:
            return None

        s, a, r, s2, d = [x.to(self.device) for x in self.buffer.sample(DQN_BATCH_SIZE)]

        # LHS of Bellman equation: Q(s, a)
        q_vals = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # RHS of Bellman equation: r + γ · max_a' Q(s', a')
        # Double DQN: policy selects a*, target evaluates Q(s', a*)
        with torch.no_grad():
            best_actions = self.policy(s2).argmax(dim=1, keepdim=True)
            q_next       = self.target(s2).gather(1, best_actions).squeeze(1)
            q_target     = r + DQN_GAMMA * q_next * (1.0 - d)

        # Bellman error — Huber loss for robustness to outlier rewards
        loss = self.loss_fn(q_vals, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % DQN_TARGET_UPDATE == 0:
            self.target.load_state_dict(self.policy.state_dict())

        return float(loss.item())

    # ------------------------------------------------------------------

    def train(self, env: PairsTradingEnv,
              n_episodes: int = DQN_TRAIN_EPISODES,
              verbose: bool = True):
        """Train on a trading environment for n_episodes passes through the data."""
        self.policy.train()    # re-enable dropout after a prior evaluate()
        episode_returns = []
        pbar = tqdm(range(n_episodes), desc="DQN training", disable=not verbose)

        for ep in pbar:
            state = env.reset()
            done  = False
            ep_reward = 0.0

            while not done:
                action              = self.select_action(state)
                next_state, r, done, _ = env.step(action)
                self.push(state, action, r, next_state, float(done))
                self.update()
                state      = next_state
                ep_reward += r

            self.epsilon = max(DQN_EPSILON_MIN,
                               self.epsilon * DQN_EPSILON_DECAY)
            episode_returns.append(ep_reward)

            if verbose and (ep + 1) % 50 == 0:
                avg = np.mean(episode_returns[-50:])
                pbar.set_postfix({"eps": f"{self.epsilon:.3f}",
                                  "avg50_ret": f"{avg:.4f}"})

        return episode_returns

    # ------------------------------------------------------------------

    def evaluate(self, env: PairsTradingEnv) -> dict:
        """Run one greedy episode (epsilon=0); return equity curve, trades, and step-level logs."""
        self.policy.eval()     # deterministic Q-values: disable Dropout
        env.log_steps = True
        state = env.reset()
        done  = False
        q_log = []
        while not done:
            q_vals = self.get_q_values(state)
            action = int(np.argmax(q_vals))
            state, _, done, _ = env.step(action)
            q_log.append({
                "dqn_q_hold":        round(float(q_vals[0]), 6),
                "dqn_q_enter_long":  round(float(q_vals[1]), 6),
                "dqn_q_enter_short": round(float(q_vals[2]), 6),
                "dqn_q_exit":        round(float(q_vals[3]), 6),
            })
        return {
            "equity_curve": env.get_equity_curve(),
            "trades":       env.trades,
            "step_log":     env.step_log,
            "q_log":        q_log,
        }
