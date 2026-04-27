"""Performance metrics for equity curves and trade logs."""

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def compute_metrics(equity_curve: pd.Series, label: str = "") -> dict:
    """
    Compute standard performance metrics from an equity curve (daily portfolio values).
    """
    eq   = equity_curve.dropna()
    rets = eq.pct_change().dropna()

    total_return    = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    n_years         = len(rets) / TRADING_DAYS
    annual_return   = float((1 + total_return) ** (1 / max(n_years, 1e-6)) - 1)
    daily_std       = float(rets.std())
    sharpe          = float(rets.mean() / daily_std * np.sqrt(TRADING_DAYS)) if daily_std > 0 else 0.0

    downside        = rets[rets < 0]
    sortino_denom   = float(downside.std() * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else 1e-9
    sortino         = float(rets.mean() * TRADING_DAYS / sortino_denom)

    # Maximum drawdown
    cummax          = eq.cummax()
    drawdown        = (eq - cummax) / cummax
    max_drawdown    = float(drawdown.min())

    calmar          = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    return {
        "label":          label,
        "total_return":   round(total_return,  4),
        "annual_return":  round(annual_return, 4),
        "sharpe":         round(sharpe,        4),
        "sortino":        round(sortino,       4),
        "max_drawdown":   round(max_drawdown,  4),
        "calmar":         round(calmar,        4),
        "daily_vol":      round(daily_std,     6),
    }


def compute_trade_stats(trades: list, label: str = "") -> dict:
    """Summarise trade-level statistics."""
    # Z-score trades have a "type" key; DQN trades have "t_exit" but no "type"
    entries = [t for t in trades if t.get("type") == "entry"
               or "t_exit" in t]    # DQN exit records = one closed round-trip each
    exits   = [t for t in trades if t.get("type") in ("exit", "stop_loss")]

    cum_rets = [t.get("cum_ret", np.nan) for t in trades if "cum_ret" in t]
    wins     = [r for r in cum_rets if r > 0]
    losses   = [r for r in cum_rets if r <= 0]

    win_rate     = len(wins) / len(cum_rets) if cum_rets else np.nan
    avg_win      = float(np.mean(wins))  if wins   else 0.0
    avg_loss     = float(np.mean(losses)) if losses else 0.0
    profit_factor = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else np.inf

    return {
        "label":         label,
        "n_entries":     len(entries),
        "n_exits":       len(exits),
        "win_rate":      round(win_rate, 4) if not np.isnan(win_rate) else "n/a",
        "avg_win":       round(avg_win,  5),
        "avg_loss":      round(avg_loss, 5),
        "profit_factor": round(profit_factor, 3) if np.isfinite(profit_factor) else "∞",
    }


def print_comparison_table(results: list):
    """Pretty-print a side-by-side metrics table."""
    keys = [
        "label", "total_return", "annual_return", "sharpe",
        "sortino", "max_drawdown", "calmar",
    ]
    fmt = {
        "total_return":  ".2%",
        "annual_return": ".2%",
        "sharpe":        ".3f",
        "sortino":       ".3f",
        "max_drawdown":  ".2%",
        "calmar":        ".3f",
    }
    col_w = 18
    header = "".join(k.ljust(col_w) for k in keys)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        row = ""
        for k in keys:
            v = r.get(k, "—")
            if k in fmt and isinstance(v, float):
                row += format(v, fmt[k]).ljust(col_w)
            else:
                row += str(v).ljust(col_w)
        print(row)
    print("=" * len(header))
