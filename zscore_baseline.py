"""Z-Score pairs-trading baseline.

Strategy logic
--------------
  Position = 0  (flat)
    → Enter LONG spread  if z < -entry_thr
    → Enter SHORT spread if z >  entry_thr

  Position = +1 (long spread, i.e. spread expected to rise)
    → Exit if z > -exit_thr  (z reverted above the exit threshold)
    → Stop-loss if z < -stop_thr

  Position = -1 (short spread)
    → Exit if z <  exit_thr
    → Stop-loss if z >  stop_thr

Grid search selects the combination that maximises Sharpe ratio on training data.
"""

import itertools
import numpy as np
import pandas as pd
from config import (
    TRANSACTION_COST, INITIAL_CAPITAL, POSITION_SIZE,
    ENTRY_THRESHOLDS, EXIT_THRESHOLDS, STOP_THRESHOLDS,
)


# ---------------------------------------------------------------------------
# Core backtest (vectorised for speed in grid search)
# ---------------------------------------------------------------------------

def run_backtest(data: pd.DataFrame,
                 entry_thr: float,
                 exit_thr: float,
                 stop_thr: float) -> dict:
    """
    Simulate the Z-score strategy on *data* (must have columns zscore, spread_return).
    Returns dict with equity_curve (pd.Series), trades (list[dict]), and step_log (list[dict]).
    """
    z   = data["zscore"].values
    sr  = data["spread_return"].values
    n   = len(z)

    position       = 0
    days_held      = 0
    cum_return     = 0.0
    portfolio      = INITIAL_CAPITAL
    equity         = [INITIAL_CAPITAL]
    trades         = []
    step_log       = []

    for t in range(n):
        zt  = z[t]
        srt = sr[t]

        pos_before   = position
        action_label = "hold"

        # --- Mark-to-market P&L from current position ---
        daily_pnl = position * srt * POSITION_SIZE

        # --- Accumulate spread return while in position (trade stats) ---
        if position != 0:
            cum_return += srt

        # --- Stop-loss check ---
        if position == 1  and zt < -stop_thr:
            action_label = "stop_loss"
            daily_pnl -= TRANSACTION_COST * 2
            trades.append({"type": "stop_loss", "position": position, "z": zt, "days": days_held, "cum_ret": cum_return * position})
            position, days_held, cum_return = 0, 0, 0.0

        elif position == -1 and zt > stop_thr:
            action_label = "stop_loss"
            daily_pnl -= TRANSACTION_COST * 2
            trades.append({"type": "stop_loss", "position": position, "z": zt, "days": days_held, "cum_ret": cum_return * position})
            position, days_held, cum_return = 0, 0, 0.0

        # --- Exit signal check (after stop-loss so stop takes priority) ---
        elif position == 1  and zt > -exit_thr:
            action_label = "exit"
            daily_pnl -= TRANSACTION_COST * 2
            trades.append({"type": "exit", "position": position, "z": zt, "days": days_held, "cum_ret": cum_return * position})
            position, days_held, cum_return = 0, 0, 0.0

        elif position == -1 and zt < exit_thr:
            action_label = "exit"
            daily_pnl -= TRANSACTION_COST * 2
            trades.append({"type": "exit", "position": position, "z": zt, "days": days_held, "cum_ret": cum_return * position})
            position, days_held, cum_return = 0, 0, 0.0

        # --- Entry signal (only when flat) ---
        elif position == 0:
            if zt < -entry_thr:
                action_label = "enter_long"
                position   = 1
                days_held  = 0
                cum_return = 0.0
                daily_pnl -= TRANSACTION_COST * 2
                trades.append({"type": "entry", "position": 1, "z": zt})
            elif zt > entry_thr:
                action_label = "enter_short"
                position   = -1
                days_held  = 0
                cum_return = 0.0
                daily_pnl -= TRANSACTION_COST * 2
                trades.append({"type": "entry", "position": -1, "z": zt})

        if position != 0:
            days_held += 1

        portfolio *= (1.0 + daily_pnl)
        equity.append(portfolio)

        step_log.append({
            "z_pos_before": pos_before,
            "z_action": action_label,
            "z_pos_after": position,
            "z_daily_pnl": round(daily_pnl, 8),
            "z_equity": round(portfolio, 2),
            "z_days_held": days_held,
        })

    # Force-close at end
    if position != 0:
        portfolio *= (1.0 - TRANSACTION_COST * 2)
        equity[-1] = portfolio

    return {
        "equity_curve": pd.Series(equity, name="zscore_portfolio"),
        "trades":        trades,
        "step_log":      step_log,
        "params":        {"entry": entry_thr, "exit": exit_thr, "stop": stop_thr},
    }


# ---------------------------------------------------------------------------
# Grid search (train set)
# ---------------------------------------------------------------------------

def sharpe_from_equity(equity: pd.Series, trading_days: int = 252) -> float:
    rets = equity.pct_change().dropna()
    if rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(trading_days))


def optimize_thresholds(train_data: pd.DataFrame,
                        entry_range: list = None,
                        exit_range:  list = None,
                        stop_range:  list = None,
                        verbose: bool = False) -> tuple[dict, pd.DataFrame]:
    """
    Grid-search over (entry, exit, stop) threshold combinations on train_data.
    Returns (best_params_dict, full_results_DataFrame).
    """
    entry_range = entry_range or ENTRY_THRESHOLDS
    exit_range  = exit_range  or EXIT_THRESHOLDS
    stop_range  = stop_range  or STOP_THRESHOLDS

    rows = []
    for entry, exit_t, stop in itertools.product(entry_range, exit_range, stop_range):
        if exit_t >= entry:          # exit must be tighter than entry
            continue
        result = run_backtest(train_data, entry, exit_t, stop)
        eq     = result["equity_curve"]
        shrp   = sharpe_from_equity(eq)
        total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
        rows.append({
            "entry": entry, "exit": exit_t, "stop": stop,
            "sharpe": round(shrp, 4),
            "total_return": round(total_ret, 4),
            "n_trades": sum(1 for tr in result["trades"] if tr["type"] == "entry"),
        })
        if verbose:
            print(f"  entry={entry} exit={exit_t} stop={stop}: Sharpe={shrp:.3f}")

    grid_df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    best = grid_df.iloc[0].to_dict()
    best_params = {"entry_thr": best["entry"], "exit_thr": best["exit"], "stop_thr": best["stop"]}
    return best_params, grid_df
