"""Custom pairs-trading environment for the DQN agent.

Theoretical connection — Lecture #3 (Pricing), slides 5–8
----------------------------------------------------------
The investor's problem in the course:

    max_ξ  u(c_t) + β E[u(c_{t+1})]
    s.t.   c_t   = e_t - p_t · ξ          (give up consumption to invest)
           c_{t+1} = e_{t+1} + x_{t+1} · ξ  (receive payoff tomorrow)

Here ξ maps to the agent's position (+1, -1, or 0), p_t · ξ is the
transaction cost of entering, and x_{t+1} · ξ is the spread P&L.
The reward at each step is the analog of u(c_{t+1}) - u(c_t): the
marginal change in portfolio value from holding the position.

Convention (earn-then-act):
  At timestep t the agent:
    1. Earns today's mark-to-market P&L from the *previous* position.
    2. Decides an action (hold / enter-long / enter-short / exit).
    3. Pays transaction costs if the position changes.
  This mirrors how real trading works: you held overnight and then decide.

Actions:
    0  Hold          – keep current position (or stay flat)
    1  Enter Long    – buy the spread (long asset1, short asset2); valid only when flat
    2  Enter Short   – sell the spread (short asset1, long asset2); valid only when flat
    3  Exit          – close open position; valid only when in a position
"""

import numpy as np
import pandas as pd
from config import TRANSACTION_COST, DQN_STATE_DIM, DQN_STOP_LOSS, POSITION_SIZE


class PairsTradingEnv:
    HOLD        = 0
    ENTER_LONG  = 1
    ENTER_SHORT = 2
    EXIT        = 3
    ACTION_NAMES = {0: "hold", 1: "enter_long", 2: "enter_short", 3: "exit"}

    def __init__(self,
                 data: pd.DataFrame,
                 initial_capital: float = 100_000.0,
                 position_size: float = POSITION_SIZE,
                 stop_loss_z: float = DQN_STOP_LOSS,
                 log_steps: bool = False):
        self.data          = data.reset_index(drop=True)
        self.n             = len(self.data)
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.stop_loss_z   = stop_loss_z
        self.state_size    = DQN_STATE_DIM
        self.action_size   = 4
        self.log_steps     = log_steps

        # Pre-compute arrays for speed
        self._z   = self.data["zscore"].values.astype(np.float32)
        self._sr  = self.data["spread_return"].values.astype(np.float32)

        sr_series     = pd.Series(self._sr)
        vol20         = sr_series.rolling(20).std()
        vol20         = vol20.fillna(vol20.median())
        self._vol20   = vol20.values.astype(np.float32)

        self._reset_state()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self._reset_state()
        return self._get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        t  = self.t
        z  = float(self._z[t])
        sr = float(self._sr[t])

        # ---- Appendix: capture pre-step state ----
        pos_before = self.position
        action_requested = action
        state_seen = self._get_state()

        # ---- 1. Earn today's P&L from the *current* position ----
        reward = self.position * sr * self.position_size
        self.cumulative_return += sr if self.position != 0 else 0.0

        # ---- 2. Resolve action validity + stop-loss ----
        # Stop-loss override
        if self.position != 0 and abs(z) > self.stop_loss_z:
            action = self.EXIT
        # Mask invalid actions → silently convert to HOLD
        if action == self.ENTER_LONG  and self.position != 0:
            action = self.HOLD
        if action == self.ENTER_SHORT and self.position != 0:
            action = self.HOLD
        if action == self.EXIT        and self.position == 0:
            action = self.HOLD

        action_resolved = action

        # ---- 3. Execute action ----
        if action == self.ENTER_LONG:
            self.position          = 1
            self.days_held         = 0
            self.cumulative_return = 0.0
            reward -= TRANSACTION_COST * 2          # both legs, entry half

        elif action == self.ENTER_SHORT:
            self.position          = -1
            self.days_held         = 0
            self.cumulative_return = 0.0
            reward -= TRANSACTION_COST * 2

        elif action == self.EXIT:
            reward -= TRANSACTION_COST * 2          # exit half
            self.trades.append({
                "t_exit":    t,
                "position":  self.position,
                "days_held": self.days_held,
                "z_exit":    z,
                "cum_ret":   self.cumulative_return * self.position,  # signed P&L
            })
            self.position          = 0
            self.days_held         = 0
            self.cumulative_return = 0.0

        # ---- 4. Advance time ----
        if self.position != 0:
            self.days_held += 1
        self.portfolio_value *= (1.0 + reward)
        self.equity_curve.append(self.portfolio_value)
        self.t += 1

        done = (self.t >= self.n - 1)
        # Force close on the final step
        if done and self.position != 0:
            # Apply one last exit cost and close
            self.portfolio_value *= (1.0 - TRANSACTION_COST * 2)
            self.equity_curve[-1] = self.portfolio_value
            self.position = 0

        # ---- Appendix: log step details ----
        if self.log_steps:
            self.step_log.append({
                "t": t,
                "dqn_pos_before": pos_before,
                "dqn_action_requested": self.ACTION_NAMES[action_requested],
                "dqn_action_resolved": self.ACTION_NAMES[action_resolved],
                "dqn_pos_after": self.position,
                "dqn_reward": round(float(reward), 8),
                "dqn_equity": round(self.portfolio_value, 2),
                "dqn_days_held": self.days_held,
                "dqn_state_z_norm": float(state_seen[0]),
                "dqn_state_z_5d": float(state_seen[1]),
                "dqn_state_pos_enc": float(state_seen[2]),
                "dqn_state_days_norm": float(state_seen[3]),
                "dqn_state_upnl": float(state_seen[4]),
                "dqn_state_vol_norm": float(state_seen[5]),
            })

        return self._get_state(), float(reward), done, {"portfolio_value": self.portfolio_value}

    def get_equity_curve(self) -> pd.Series:
        return pd.Series(self.equity_curve, name="portfolio_value")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reset_state(self):
        # Start at t=0 so the DQN and Z-score baseline trade the same test
        # window. prepare_pair_data.dropna() already removed the rolling-window
        # warm-up; _vol20 NaNs are median-filled; z_5d clips to max(0, t-5).
        self.t                 = 0
        self.position          = 0
        self.days_held         = 0
        self.cumulative_return = 0.0
        self.portfolio_value   = self.initial_capital
        self.equity_curve      = [self.initial_capital]
        self.trades            = []
        self.step_log          = []

    def _get_state(self) -> np.ndarray:
        t = min(self.t, self.n - 1)
        z = self._z[t]

        # 1. z-score (clipped, normalised to ~[-1,1])
        z_norm   = float(np.clip(z / 4.0, -1.0, 1.0))
        # 2. 5-day z-score change — is spread moving toward or away from zero?
        z_5d     = float(np.clip((z - self._z[max(0, t - 5)]) / 2.0, -1.0, 1.0))
        # 3. Position encoded as -1 / 0 / +1
        pos_enc  = float(self.position)
        # 4. Days held (normalised, caps at 60)
        days_norm = float(min(self.days_held / 60.0, 1.0))
        # 5. Unrealised P&L (cumulative spread return scaled to ~[-1,1])
        upnl     = float(np.clip(self.cumulative_return * self.position / 0.05, -1.0, 1.0))
        # 6. 20-day spread volatility — how risky is the current environment?
        vol_norm = float(np.clip(self._vol20[t] / 0.015, 0.0, 3.0) - 1.0)

        return np.array([z_norm, z_5d, pos_enc, days_norm, upnl, vol_norm],
                        dtype=np.float32)
