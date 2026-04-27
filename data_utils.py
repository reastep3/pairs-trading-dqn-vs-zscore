"""Data fetching, cointegration testing, and spread preparation.

Hedge ratio estimation uses the OLS framework from slide 17/20 of the
Columbia lecture (coursework_math.ols_predict).  Log returns use the
slide-19 definition (coursework_math.log_return).
"""

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint, adfuller

from coursework_math import log_return, ols_predict


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_data(pairs: list, start: str, end: str) -> pd.DataFrame:
    """Download adjusted-close prices for every ticker that appears in pairs."""
    tickers = sorted({t for pair in pairs for t in pair})
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Single ticker edge-case
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    return prices.ffill().dropna(how="all")


# ---------------------------------------------------------------------------
# Spread primitives
# ---------------------------------------------------------------------------

def estimate_hedge_ratio(log_p1: pd.Series, log_p2: pd.Series) -> tuple[float, float]:
    """
    OLS: log_p1 = beta * log_p2 + alpha  →  (beta, alpha).

    Uses coursework_math.ols_predict (slides 17, 20): same OLS framework as
    the predictability regression R_{t→t+k} = a + b·x_t + ε, here applied
    to levels to extract the cointegrating vector.
    """
    result = ols_predict(log_p1, log_p2, newey_west_lags=0)
    return float(result["b"]), float(result["a"])


def compute_spread(log_p1: pd.Series, log_p2: pd.Series,
                   beta: float, alpha: float) -> pd.Series:
    return log_p1 - beta * log_p2 - alpha


def compute_rolling_zscore(spread: pd.Series, window: int) -> pd.Series:
    mean = spread.rolling(window).mean()
    std  = spread.rolling(window).std()
    return ((spread - mean) / std.replace(0.0, np.nan)).rename("zscore")


# ---------------------------------------------------------------------------
# Cointegration test
# ---------------------------------------------------------------------------

def test_cointegration(p1: pd.Series, p2: pd.Series) -> dict:
    """Engle-Granger test on log prices; also reports ADF on the residual spread."""
    log_p1, log_p2 = np.log(p1), np.log(p2)
    _, coint_pval, _ = coint(log_p1, log_p2)
    beta, alpha = estimate_hedge_ratio(log_p1, log_p2)
    spread = compute_spread(log_p1, log_p2, beta, alpha)
    adf_pval = adfuller(spread.dropna())[1]
    return {
        "coint_pvalue": round(float(coint_pval), 4),
        "adf_pvalue":   round(float(adf_pval),   4),
        "hedge_ratio":  round(float(beta),        4),
        "alpha":        round(float(alpha),        4),
        "cointegrated": bool(coint_pval < 0.05 and adf_pval < 0.05),
    }


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------

def select_pairs(prices: pd.DataFrame,
                 candidates: list,
                 verbose: bool = True) -> list:
    """
    Test all candidate pairs. Return those that are cointegrated (p < 0.05),
    sorted by Engle-Granger p-value ascending.
    """
    results = []
    for t1, t2 in candidates:
        if t1 not in prices.columns or t2 not in prices.columns:
            if verbose:
                print(f"  [skip] {t1}/{t2}: missing data")
            continue
        common = prices[[t1, t2]].dropna()
        if len(common) < 250:
            if verbose:
                print(f"  [skip] {t1}/{t2}: insufficient rows ({len(common)})")
            continue
        stats = test_cointegration(common[t1], common[t2])
        stats["pair"] = (t1, t2)
        results.append(stats)
        if verbose:
            mark = "✓" if stats["cointegrated"] else "✗"
            print(
                f"  {mark} {t1}/{t2}: "
                f"coint_p={stats['coint_pvalue']:.4f}  "
                f"adf_p={stats['adf_pvalue']:.4f}  "
                f"β={stats['hedge_ratio']:.3f}"
            )
    # Return ALL candidates sorted by coint p-value (best first),
    # so we can compare cointegrated vs non-cointegrated pairs.
    coint_count = sum(1 for r in results if r["cointegrated"])
    print(f"  [{coint_count}/{len(results)} pairs pass cointegration test]")
    results.sort(key=lambda x: x["coint_pvalue"])
    return results


# ---------------------------------------------------------------------------
# Feature DataFrame
# ---------------------------------------------------------------------------

def prepare_pair_data(prices: pd.DataFrame,
                      ticker1: str, ticker2: str,
                      beta: float, alpha: float,
                      zscore_window: int) -> pd.DataFrame:
    """
    Build a feature DataFrame using a *pre-estimated* hedge ratio so there is
    no look-ahead bias when the hedge ratio was estimated only on training data.

    Columns: p1, p2, log_p1, log_p2, spread, zscore, r1, r2, spread_return
    """
    common = prices[[ticker1, ticker2]].dropna()
    p1, p2 = common[ticker1], common[ticker2]
    log_p1, log_p2 = np.log(p1), np.log(p2)

    spread  = compute_spread(log_p1, log_p2, beta, alpha)
    zscore  = compute_rolling_zscore(spread, zscore_window)
    # Slide 19: r_t = log(p_t) - log(p_{t-1})
    r1      = log_return(log_p1)
    r2      = log_return(log_p2)
    spread_return = r1 - beta * r2   # log-return of the dollar-neutral spread

    df = pd.DataFrame({
        "p1": p1, "p2": p2,
        "log_p1": log_p1, "log_p2": log_p2,
        "spread": spread, "zscore": zscore,
        "r1": r1, "r2": r2,
        "spread_return": spread_return,
    })
    return df.dropna()
