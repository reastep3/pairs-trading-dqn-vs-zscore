"""
DQN vs Z-Score Pairs Trading — Research Orchestrator
=====================================================
Run:  python main.py

For each cointegrated pair (from CANDIDATE_PAIRS):
  1. Download prices, estimate hedge ratio on training data only.
  2. Z-Score strategy: grid-search thresholds on train, evaluate on test.
  3. DQN strategy:    train on train period, evaluate on test period.
  4. Print comparison table and save equity-curve plots + CSVs.

All evaluation is strictly out-of-sample (test period).
"""

import os
import sys
import random
import warnings
from datetime import datetime

# Windows CP1252 cannot encode Greek letters / math symbols in print()
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")            # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

from config import (
    CANDIDATE_PAIRS, PAIR_STRATA, DATA_START, DATA_END, TRAIN_RATIO,
    ZSCORE_WINDOW, INITIAL_CAPITAL, DQN_TRAIN_EPISODES,
)
from data_utils import (
    download_data, select_pairs,
    estimate_hedge_ratio, prepare_pair_data,
)
from zscore_baseline import optimize_thresholds, run_backtest
from dqn_agent import DQNAgent
from trading_env import PairsTradingEnv
from metrics import compute_metrics, compute_trade_stats, print_comparison_table
from coursework_math import (
    ar1_fit, predictability_table, variance_decomposition, persistence_buildup,
    lop_spread_test, market_neutrality, bellman_convergence,
)


BASE_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
# Descriptive tag for this run; timestamp appended so reruns never clobber.
RUN_TAG       = "run_30pairs_postbugfix"
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RESULTS_DIR   = os.path.join(BASE_RESULTS_DIR, f"{RUN_TAG}_{RUN_TIMESTAMP}")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ===========================================================================
# Plotting
# ===========================================================================

def _drawdown_pct(equity_values: np.ndarray) -> np.ndarray:
    cummax = np.maximum.accumulate(equity_values)
    return (equity_values - cummax) / cummax * 100.0


def plot_pair_results(pair_label: str,
                      test_data: pd.DataFrame,
                      zscore_eq: pd.Series,
                      dqn_eq: pd.Series,
                      dqn_episode_returns: list,
                      best_zscore_params: dict):
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f"Pairs Trading Research — {pair_label}",
                 fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    min_len = min(len(zscore_eq), len(dqn_eq))

    # ---- 1. Equity curves (test period) ----
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(zscore_eq.values[:min_len],
             color="#2196F3", lw=1.8, label="Z-Score (grid-optimised)")
    ax1.plot(dqn_eq.values[:min_len],
             color="#F44336", lw=1.8, label="DQN Agent")
    # Dollar-neutral buy-and-hold of the spread as a reference
    spread_ret = test_data["spread_return"].values[:min_len]
    bnh = INITIAL_CAPITAL * np.cumprod(1.0 + spread_ret * 0.5)
    ax1.plot(np.insert(bnh, 0, INITIAL_CAPITAL),
             color="#9C27B0", lw=1.2, linestyle="--", label="B&H spread (0.5x)")
    ax1.axhline(INITIAL_CAPITAL, color="grey", lw=0.8, linestyle=":")
    ax1.set_title("Out-of-Sample Equity Curves")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(fontsize=9)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # ---- 2. Z-Score over test period ----
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(test_data["zscore"].values, color="#455A64", lw=1.0)
    e = best_zscore_params["entry_thr"]
    x = best_zscore_params["exit_thr"]
    ax2.axhline( e, color="#F44336", lw=0.8, linestyle="--",
                label=f"±Entry ({e})")
    ax2.axhline(-e, color="#F44336", lw=0.8, linestyle="--")
    ax2.axhline( x, color="#4CAF50", lw=0.8, linestyle=":",
                label=f"±Exit ({x})")
    ax2.axhline(-x, color="#4CAF50", lw=0.8, linestyle=":")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_title("Z-Score (test period)")
    ax2.set_ylabel("Z-Score")
    ax2.legend(fontsize=8)

    # ---- 3. DQN training curve ----
    ax3 = fig.add_subplot(gs[1, 1])
    window = max(1, len(dqn_episode_returns) // 20)
    smoothed = pd.Series(dqn_episode_returns).rolling(window).mean()
    ax3.plot(dqn_episode_returns, alpha=0.25, color="#F44336", lw=0.8)
    ax3.plot(smoothed.values, color="#F44336", lw=1.5,
             label=f"Rolling {window}-ep avg")
    ax3.set_title("DQN Training Curve")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Cumulative Reward")
    ax3.legend(fontsize=8)

    # ---- 4. Drawdown comparison ----
    ax4 = fig.add_subplot(gs[2, 0])
    z_dd  = _drawdown_pct(zscore_eq.values[:min_len])
    dq_dd = _drawdown_pct(dqn_eq.values[:min_len])
    ax4.fill_between(range(min_len), z_dd,  0,
                     alpha=0.4, color="#2196F3", label="Z-Score")
    ax4.fill_between(range(min_len), dq_dd, 0,
                     alpha=0.4, color="#F44336", label="DQN")
    ax4.set_title("Drawdown")
    ax4.set_ylabel("Drawdown (%)")
    ax4.legend(fontsize=8)

    # ---- 5. Rolling 60-day Sharpe ----
    ax5 = fig.add_subplot(gs[2, 1])
    z_rets  = pd.Series(zscore_eq.values[:min_len]).pct_change().dropna()
    dq_rets = pd.Series(dqn_eq.values[:min_len]).pct_change().dropna()
    roll_w  = 60

    def rolling_sharpe(rets, w):
        return rets.rolling(w).apply(
            lambda r: r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0,
            raw=True)

    ax5.plot(rolling_sharpe(z_rets,  roll_w).values,
             color="#2196F3", lw=1.2, label="Z-Score")
    ax5.plot(rolling_sharpe(dq_rets, roll_w).values,
             color="#F44336", lw=1.2, label="DQN")
    ax5.axhline(0, color="black", lw=0.5)
    ax5.set_title(f"Rolling {roll_w}-Day Sharpe")
    ax5.set_ylabel("Sharpe")
    ax5.legend(fontsize=8)

    path = os.path.join(RESULTS_DIR, f"{pair_label}_results.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# ===========================================================================
# Aggregate presentation chart
# ===========================================================================

def plot_aggregate_summary(summary_df: pd.DataFrame):
    """Presentation-ready aggregate summary across all pairs."""
    from math import comb as _comb

    df = summary_df.copy()
    stratum_order = {"Strong": 0, "Moderate": 1, "Weak": 2}
    df["_sort"] = df["stratum"].map(stratum_order)
    df = df.sort_values(["_sort", "coint_pvalue"]).reset_index(drop=True)

    STRATA       = ["Strong", "Moderate", "Weak"]
    S_COLOR      = {"Strong": "#4CAF50", "Moderate": "#FF9800", "Weak": "#F44336"}
    S_BG         = {"Strong": "#C8E6C9", "Moderate": "#FFF9C4", "Weak": "#FFCDD2"}

    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(
        "DQN vs Z-Score Pairs Trading — Aggregate Results\n"
        f"{len(df)} Pairs  |  {DATA_START} -> {DATA_END}  |  "
        f"Train {TRAIN_RATIO:.0%} / Test {1-TRAIN_RATIO:.0%}",
        fontsize=15, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.5, 1],
                           hspace=0.40, wspace=0.35)

    # ================================================================
    # Panel 1 (top, full width): Sharpe Ratio by Pair
    # ================================================================
    ax1 = fig.add_subplot(gs[0, :])
    n_pairs = len(df)
    x = np.arange(n_pairs)
    w = 0.35
    ax1.bar(x - w/2, df["z_sharpe"], w, color="#2196F3",
            label="Z-Score", alpha=0.85, zorder=3)
    ax1.bar(x + w/2, df["dqn_sharpe"], w, color="#F44336",
            label="DQN", alpha=0.85, zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["pair"], rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Sharpe Ratio (out-of-sample)")
    ax1.set_title("Sharpe Ratio Comparison by Pair", fontsize=13)
    ax1.axhline(0, color="black", lw=0.6, zorder=2)
    ax1.legend(loc="upper right", fontsize=10)

    # Stratum background shading + labels
    prev_stratum, start = None, 0
    for i in range(n_pairs + 1):
        cur = df["stratum"].iloc[i] if i < n_pairs else None
        if cur != prev_stratum:
            if prev_stratum is not None:
                ax1.axvspan(start - 0.5, i - 0.5, alpha=0.12,
                            color=S_BG[prev_stratum], zorder=0)
                mid = (start + i - 1) / 2
                yhi = ax1.get_ylim()[1]
                ax1.text(mid, yhi * 0.93,
                         f"\u2190 {prev_stratum} cointegration \u2192",
                         ha="center", fontsize=9, fontstyle="italic",
                         color=S_COLOR[prev_stratum], fontweight="bold")
            start = i
            prev_stratum = cur

    # Winner markers
    for i, row in df.iterrows():
        winner_x = i - w/2 if row["winner"] == "Z-Score" else i + w/2
        winner_y = row["z_sharpe"] if row["winner"] == "Z-Score" else row["dqn_sharpe"]
        offset = 0.03 if winner_y >= 0 else -0.06
        ax1.text(winner_x, winner_y + offset, "\u2605",
                 ha="center", fontsize=8, color="gold", fontweight="bold")

    # ================================================================
    # Panel 2 (bottom-left): Win Count by Stratum
    # ================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    z_wins, d_wins, totals = [], [], []
    for s in STRATA:
        mask = df["stratum"] == s
        zw = (df.loc[mask, "winner"] == "Z-Score").sum()
        dw = (df.loc[mask, "winner"] == "DQN").sum()
        z_wins.append(zw); d_wins.append(dw); totals.append(zw + dw)

    x2 = np.arange(3)
    bw = 0.30
    ax2.bar(x2 - bw/2, z_wins, bw, color="#2196F3", label="Z-Score")
    ax2.bar(x2 + bw/2, d_wins, bw, color="#F44336", label="DQN")
    ax2.set_xticks(x2)
    ax2.set_xticklabels([f"{s}\n(n={t})" for s, t in zip(STRATA, totals)])
    ax2.set_ylabel("Pairs Won")
    ax2.set_title("Win Count by Cointegration Stratum", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, max(max(z_wins), max(d_wins)) + 1.8)
    for i in range(3):
        ax2.text(x2[i] - bw/2, z_wins[i] + 0.15, str(z_wins[i]),
                 ha="center", fontweight="bold", color="#1565C0", fontsize=12)
        ax2.text(x2[i] + bw/2, d_wins[i] + 0.15, str(d_wins[i]),
                 ha="center", fontweight="bold", color="#C62828", fontsize=12)

    # ================================================================
    # Panel 3 (bottom-mid): Cointegration Strength vs Z-Score Advantage
    # ================================================================
    ax3 = fig.add_subplot(gs[1, 1])
    sharpe_adv = df["z_sharpe"] - df["dqn_sharpe"]
    coint_str  = -np.log10(df["coint_pvalue"].clip(lower=1e-10))

    for s in STRATA:
        mask = df["stratum"] == s
        ax3.scatter(coint_str[mask], sharpe_adv[mask],
                    c=S_COLOR[s], s=70, label=s, zorder=5,
                    edgecolors="white", linewidths=0.5)

    # Trend line
    coeffs = np.polyfit(coint_str, sharpe_adv, 1)
    x_fit  = np.linspace(coint_str.min() - 0.3, coint_str.max() + 0.3, 100)
    ax3.plot(x_fit, np.polyval(coeffs, x_fit), color="grey",
             linestyle="--", alpha=0.6, lw=1.5,
             label=f"trend (\u03b2={coeffs[0]:.3f})")

    ax3.axhline(0, color="black", lw=0.5, linestyle=":")
    ax3.set_xlabel("\u2212log\u2081\u2080(Engle-Granger p-value)  \u2192  stronger cointegration")
    ax3.set_ylabel("Z Sharpe \u2212 DQN Sharpe\n(positive = Z-Score better)")
    ax3.set_title("Cointegration Strength vs Strategy Edge", fontsize=11)
    ax3.legend(fontsize=8, loc="best")

    # ================================================================
    # Panel 4 (bottom-right): Key Statistics
    # ================================================================
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")

    total_z = (df["winner"] == "Z-Score").sum()
    total_d = (df["winner"] == "DQN").sum()
    total_n = len(df)

    # One-sided binomial p-value: P(X >= max(z,d) | n, p=0.5)
    k = max(total_z, total_d)
    binom_p = sum(_comb(total_n, j) * 0.5**total_n for j in range(k, total_n + 1))

    # Wilcoxon signed-rank test (uses magnitude of Sharpe differences)
    from scipy.stats import wilcoxon as _wilcoxon
    _sharpe_diff = df["dqn_sharpe"] - df["z_sharpe"]
    try:
        _, wilcox_p = _wilcoxon(_sharpe_diff)
    except (ValueError, TypeError):
        wilcox_p = np.nan

    rows = [
        ("OVERALL RECORD",   f"Z-Score {total_z}  \u2014  DQN {total_d}",  True),
        ("Binomial p-value", f"{binom_p:.4f}" + ("  *" if binom_p < 0.05 else ""), False),
        ("Wilcoxon p-value", f"{wilcox_p:.4f}" + ("  *" if wilcox_p < 0.05 else ""), False),
        ("", "", False),
        ("MEAN SHARPE",
         f"Z: {df['z_sharpe'].mean():+.3f}    D: {df['dqn_sharpe'].mean():+.3f}", True),
        ("", "", False),
    ]
    for s in STRATA:
        mask = df["stratum"] == s
        zm = df.loc[mask, "z_sharpe"].mean()
        dm = df.loc[mask, "dqn_sharpe"].mean()
        zw = (df.loc[mask, "winner"] == "Z-Score").sum()
        dw = (df.loc[mask, "winner"] == "DQN").sum()
        rows.append((f"  {s} (n={mask.sum()})",
                      f"Z: {zm:+.3f}  D: {dm:+.3f}  ({zw}\u2013{dw})", False))
    rows += [
        ("", "", False),
        ("MEAN MAX DD",
         f"Z: {df['z_max_dd'].mean():.1%}    D: {df['dqn_max_dd'].mean():.1%}", True),
        ("MEAN TOTAL RET",
         f"Z: {df['z_total_ret'].mean():.1%}    D: {df['dqn_total_ret'].mean():.1%}", False),
    ]
    # Overfitting diagnostic (train-test Sharpe gap)
    if "z_train_sharpe" in df.columns:
        z_gap   = df["z_train_sharpe"].mean() - df["z_sharpe"].mean()
        dqn_gap = df["dqn_train_sharpe"].mean() - df["dqn_sharpe"].mean()
        rows += [
            ("", "", False),
            ("TRAIN\u2192TEST GAP",
             f"Z: {z_gap:+.3f}    D: {dqn_gap:+.3f}", True),
        ]

    y = 0.95
    for label, value, bold in rows:
        if not label and not value:
            y -= 0.03
            continue
        wt = "bold" if bold else "normal"
        ax4.text(0.02, y, label, fontsize=10, fontweight=wt,
                 transform=ax4.transAxes, va="top", family="monospace")
        ax4.text(0.98, y, value, fontsize=10, fontweight=wt,
                 transform=ax4.transAxes, va="top", ha="right", family="monospace")
        y -= 0.07
    ax4.set_title("Key Statistics", fontsize=11)

    path = os.path.join(RESULTS_DIR, "aggregate_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# ===========================================================================
# Per-pair pipeline
# ===========================================================================

def run_pair(prices: pd.DataFrame, t1: str, t2: str, stats: dict):
    label = f"{t1}-{t2}"
    print(f"\n{'='*60}")
    print(f"  Pair : {label}")
    print(f"  coint_p={stats['coint_pvalue']:.4f}  "
          f"adf_p={stats['adf_pvalue']:.4f}  β≈{stats['hedge_ratio']:.3f}")
    print(f"{'='*60}")

    # ---- Estimate hedge ratio on training dates only (no look-ahead) ----
    common  = prices[[t1, t2]].dropna()
    cut_idx = int(len(common) * TRAIN_RATIO)
    tr_px   = common.iloc[:cut_idx]
    beta, alpha = estimate_hedge_ratio(np.log(tr_px[t1]), np.log(tr_px[t2]))
    print(f"  Train hedge ratio: β={beta:.4f}  α={alpha:.4f}")

    # Build full feature DataFrame (spread / z-score use training β)
    full_data  = prepare_pair_data(prices, t1, t2, beta, alpha, ZSCORE_WINDOW)
    train_cut  = int(len(full_data) * TRAIN_RATIO)
    train_data = full_data.iloc[:train_cut].reset_index(drop=True)
    test_dates = full_data.iloc[train_cut:].index
    test_data  = full_data.iloc[train_cut:].reset_index(drop=True)
    print(f"  Train: {len(train_data)} days  |  Test: {len(test_data)} days")

    if len(test_data) < 60:
        print("  [skip] Insufficient test data.")
        return None

    # ================================================================
    # Coursework Math Analysis (training period only — no look-ahead)
    # ================================================================
    print("\n  [Coursework Math] Lecture 5 analysis on training data...")

    # Slides 18, 20 — AR(1) of the z-score predictor
    ar1 = ar1_fit(train_data["zscore"])
    print(f"    AR(1) z-score:  ρ={ar1['rho']}  "
          f"half_life={ar1['half_life_days']} days  "
          f"t={ar1['tstat']}")

    # Slide 17 (Table 20.1 analog) — multi-horizon predictability
    pred_tbl = predictability_table(
        spread=train_data["spread"],
        spread_returns=train_data["spread_return"],
        z_scores=train_data["zscore"],
    )
    print("    Predictability table (z-score -> forward spread return):")
    print("    " + pred_tbl.to_string(index=False))

    # Slides 20–21 — persistence build-up using training-period b and ρ
    b_h1 = pred_tbl.loc[pred_tbl["horizon_days"] == 1, "b"]
    if not b_h1.empty and not np.isnan(b_h1.values[0]):
        pbdf = persistence_buildup(b=b_h1.values[0], rho=ar1["rho"], max_horizon=20)
        limit_coef = pbdf["cumulative_coef"].iloc[-1]
        frac_at5   = pbdf.loc[pbdf["horizon"] == 5, "fraction_of_limit"]
        frac_str   = f"{frac_at5.values[0]:.1%}" if not frac_at5.empty else "n/a"
        print(f"    Persistence build-up (slide 20–21): "
              f"long-run coef≈{limit_coef:.4f}, "
              f"5-day fraction of limit={frac_str}")

    # Slides 27, 29 — present-value variance decomposition
    vd = variance_decomposition(
        z_scores=train_data["zscore"],
        spread_returns=train_data["spread_return"],
        rho=ar1["rho"],
    )
    print(f"    PV decomp (slides 27–29):  "
          f"Var[z]={vd['var_z']:.4f}  "
          f"return fraction={vd['return_fraction']:.2%}")
    if vd["return_fraction"] >= 0.5:
        interp = "spread variation is substantially return-driven -> mean reversion credible"
    else:
        interp = "low return fraction -> spread may not be stationary / fundamental risk"
    print(f"    Interpretation: {interp}")

    pred_tbl.to_csv(
        os.path.join(RESULTS_DIR, f"{label}_predictability.csv"), index=False)

    # Lectures 1–2 (Arbitrage): Law of One Price / no-arbitrage test
    lop = lop_spread_test(train_data["spread_return"])
    print(f"    LOP test (Lec 1-2, slides 11/14): "
          f"drift={lop['annual_drift']:+.4f}/yr  t={lop['t_stat']}  "
          f"{'LOP holds (p>.05)' if lop['lop_holds'] else 'drift detected'}")

    # Lecture 5 slide 33 + Lecture 4 slides 8-9: CAPM market neutrality
    if "SPY" in prices.columns:
        mkt_ret = np.log(prices["SPY"]).diff()
        mkt_aligned = mkt_ret.reindex(full_data.index)
        train_mkt = mkt_aligned.iloc[:train_cut].reset_index(drop=True)
        mn = market_neutrality(train_data["spread_return"], train_mkt)
        print(f"    Market neutrality (Lec 4-5, CAPM): "
              f"\u03b2={mn['beta']}  \u03b1={mn['alpha_annual']:+.4f}/yr  "
              f"R\u00b2={mn['R2']}")
    else:
        mn = {"beta": np.nan, "alpha_annual": np.nan, "R2": np.nan}

    # ================================================================
    # Z-Score strategy
    # ================================================================
    print("\n  [Z-Score] Grid-searching thresholds on training data...")
    best_params, grid_df = optimize_thresholds(train_data)
    print(f"  [Z-Score] Best -> entry={best_params['entry_thr']}  "
          f"exit={best_params['exit_thr']}  stop={best_params['stop_thr']}")
    train_z_sharpe = float(grid_df.iloc[0]["sharpe"])
    grid_df.to_csv(
        os.path.join(RESULTS_DIR, f"{label}_zscore_grid.csv"), index=False)

    zscore_result  = run_backtest(test_data, **best_params)
    zscore_eq      = zscore_result["equity_curve"]
    zscore_metrics = compute_metrics(zscore_eq, label="Z-Score")
    zscore_trades  = compute_trade_stats(zscore_result["trades"], label="Z-Score")

    # ================================================================
    # DQN strategy
    # ================================================================
    print("\n  [DQN] Training agent on training period ...")
    train_env  = PairsTradingEnv(train_data, initial_capital=INITIAL_CAPITAL)
    agent      = DQNAgent()
    ep_returns = agent.train(train_env, n_episodes=DQN_TRAIN_EPISODES, verbose=True)

    # DQN training-period performance (overfitting diagnostic)
    train_env_eval    = PairsTradingEnv(train_data, initial_capital=INITIAL_CAPITAL)
    train_dqn_result  = agent.evaluate(train_env_eval)
    train_dqn_sharpe  = compute_metrics(train_dqn_result["equity_curve"])["sharpe"]

    print("  [DQN] Evaluating (greedy) on test period ...")
    test_env   = PairsTradingEnv(test_data, initial_capital=INITIAL_CAPITAL)
    dqn_result = agent.evaluate(test_env)
    dqn_eq     = dqn_result["equity_curve"]
    dqn_metrics = compute_metrics(dqn_eq, label="DQN Agent")
    dqn_trades  = compute_trade_stats(dqn_result["trades"], label="DQN Agent")

    # Lecture 3 slides 14-18: Bellman equation convergence diagnostic
    from config import DQN_GAMMA
    dqn_rewards = [step["dqn_reward"] for step in dqn_result["step_log"]]
    bellman = bellman_convergence(
        dqn_result["q_log"], dqn_result["step_log"], dqn_rewards, DQN_GAMMA)
    print(f"\n  [Bellman diagnostic (Lec 3, slides 14-18)]")
    print(f"    Mean Bellman error: {bellman['mean_bellman_error']:.6f}  "
          f"MAE: {bellman['mean_abs_error']:.6f}  "
          f"({bellman['n_steps']} steps)")

    # ================================================================
    # Display
    # ================================================================
    print(f"\n  --- Performance Metrics (out-of-sample) ---")
    print_comparison_table([zscore_metrics, dqn_metrics])

    print(f"\n  --- Trade Stats ---")
    cols = ["label", "n_entries", "win_rate", "avg_win", "avg_loss", "profit_factor"]
    for ts in [zscore_trades, dqn_trades]:
        print("  " + "  |  ".join(f"{k}: {ts[k]}" for k in cols))

    # ================================================================
    # Persist
    # ================================================================
    pd.DataFrame([zscore_metrics, dqn_metrics]).to_csv(
        os.path.join(RESULTS_DIR, f"{label}_metrics.csv"), index=False)

    pd.Series(ep_returns, name="episode_return").to_csv(
        os.path.join(RESULTS_DIR, f"{label}_dqn_training.csv"), index=False)

    min_len = min(len(zscore_eq), len(dqn_eq))
    pd.DataFrame({
        "zscore_equity": zscore_eq.values[:min_len],
        "dqn_equity":    dqn_eq.values[:min_len],
    }).to_csv(os.path.join(RESULTS_DIR, f"{label}_equity_curves.csv"), index=False)

    # ================================================================
    # Appendix: step-by-step decision log
    # ================================================================
    appendix = pd.DataFrame({
        "date":          test_dates.values,
        "p1":            test_data["p1"].values,
        "p2":            test_data["p2"].values,
        "spread":        test_data["spread"].values,
        "zscore":        test_data["zscore"].values,
        "spread_return": test_data["spread_return"].values,
    })

    # Z-Score agent decisions (one row per test day)
    z_log_df = pd.DataFrame(zscore_result["step_log"])
    for col in z_log_df.columns:
        appendix[col] = z_log_df[col].values

    # DQN agent decisions + Q-values (starts at t=20 due to warm-up)
    dqn_step_df  = pd.DataFrame(dqn_result["step_log"])
    dqn_q_df     = pd.DataFrame(dqn_result["q_log"])
    dqn_combined = pd.concat([dqn_step_df, dqn_q_df], axis=1)
    dqn_aligned  = dqn_combined.set_index("t").reindex(range(len(test_data)))
    for col in dqn_aligned.columns:
        appendix[col] = dqn_aligned[col].values

    appendix.to_csv(
        os.path.join(RESULTS_DIR, f"{label}_appendix.csv"), index=False)
    print(f"  [saved] {label}_appendix.csv ({len(appendix)} rows × {len(appendix.columns)} cols)")

    plot_pair_results(label, test_data, zscore_eq, dqn_eq, ep_returns, best_params)

    return {
        "pair":            label,
        "z_sharpe":        zscore_metrics["sharpe"],
        "dqn_sharpe":      dqn_metrics["sharpe"],
        "z_total_ret":     zscore_metrics["total_return"],
        "dqn_total_ret":   dqn_metrics["total_return"],
        "z_max_dd":        zscore_metrics["max_drawdown"],
        "dqn_max_dd":      dqn_metrics["max_drawdown"],
        "winner":          "DQN" if dqn_metrics["sharpe"] > zscore_metrics["sharpe"]
                           else "Z-Score",
        # Overfitting diagnostic: train-period Sharpe
        "z_train_sharpe":  train_z_sharpe,
        "dqn_train_sharpe": train_dqn_sharpe,
        # Coursework math diagnostics
        "ar1_rho":         ar1["rho"],
        "half_life_days":  ar1["half_life_days"],
        "innovation_var":  ar1["innovation_var"],
        "pv_ret_fraction": vd["return_fraction"],
        "lop_annual_drift": lop["annual_drift"],
        "lop_holds":       lop["lop_holds"],
        "market_beta":     mn.get("beta", np.nan),
        "bellman_mae":     bellman["mean_abs_error"],
        # Cointegration test results
        "coint_pvalue":    stats["coint_pvalue"],
        "adf_pvalue":      stats["adf_pvalue"],
        "cointegrated":    stats["cointegrated"],
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    # ---- Reproducibility ----
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

    print("\n" + "=" * 60)
    print("  DQN vs Z-Score Pairs Trading Research")
    print("=" * 60)

    print(f"\n[1/4] Downloading price data ({DATA_START} -> {DATA_END})...")
    prices = download_data(CANDIDATE_PAIRS, DATA_START, DATA_END)
    print(f"  {len(prices)} trading days  |  tickers: {list(prices.columns)}")

    print("\n[2/4] Cointegration tests...")
    selected = select_pairs(prices, CANDIDATE_PAIRS, verbose=True)
    if not selected:
        print("  No cointegrated pairs — aborting.")
        return
    # Run all pairs — stratified across cointegration levels
    selected = selected[:30]
    print(f"\n  Running pipeline on: {[s['pair'] for s in selected]}")

    print("\n[3/4] Running Z-Score & DQN for each pair...")
    summary_rows = []
    for stats in selected:
        t1, t2 = stats["pair"]
        row = run_pair(prices, t1, t2, stats)
        if row:
            summary_rows.append(row)

    if not summary_rows:
        print("No results to report.")
        return

    print("\n[4/4] FINAL SUMMARY")
    print("=" * 70)
    summary_df = pd.DataFrame(summary_rows)
    summary_df["stratum"] = summary_df["pair"].map(PAIR_STRATA)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
    print(f"\n  Results saved to: {RESULTS_DIR}/")

    dqn_wins    = (summary_df["winner"] == "DQN").sum()
    zscore_wins = (summary_df["winner"] == "Z-Score").sum()
    avg_z   = summary_df["z_sharpe"].mean()
    avg_dqn = summary_df["dqn_sharpe"].mean()

    print(f"\n  DQN wins {dqn_wins}/{len(summary_df)} pairs | "
          f"Z-Score wins {zscore_wins}/{len(summary_df)} pairs")
    print(f"  Mean Sharpe — Z-Score: {avg_z:.3f}  |  DQN: {avg_dqn:.3f}")

    # ---- Per-stratum breakdown ----
    print(f"\n  Per-stratum analysis:")
    for s in ["Strong", "Moderate", "Weak"]:
        mask = summary_df["stratum"] == s
        if not mask.any():
            continue
        z_m  = summary_df.loc[mask, "z_sharpe"].mean()
        d_m  = summary_df.loc[mask, "dqn_sharpe"].mean()
        d_w  = (summary_df.loc[mask, "winner"] == "DQN").sum()
        tot  = mask.sum()
        print(f"    {s:8s}: Z={z_m:+.3f}  DQN={d_m:+.3f}  (DQN wins {d_w}/{tot})")

    # ---- Wilcoxon signed-rank test (uses magnitude, not just sign) ----
    from scipy.stats import wilcoxon as _wilcoxon
    sharpe_diff = summary_df["dqn_sharpe"] - summary_df["z_sharpe"]
    try:
        _, wilcox_p = _wilcoxon(sharpe_diff)
    except (ValueError, TypeError):
        wilcox_p = np.nan
    print(f"\n  Wilcoxon signed-rank p-value: {wilcox_p:.4f}"
          + ("  *" if wilcox_p < 0.05 else ""))

    # ---- Overfitting diagnostic ----
    if "z_train_sharpe" in summary_df.columns:
        z_gap   = summary_df["z_train_sharpe"].mean() - avg_z
        dqn_gap = summary_df["dqn_train_sharpe"].mean() - avg_dqn
        print(f"\n  Overfitting diagnostic (mean train Sharpe - mean test Sharpe):")
        print(f"    Z-Score: {z_gap:+.3f}  |  DQN: {dqn_gap:+.3f}")
        if dqn_gap > z_gap + 0.3:
            print("    [!] DQN shows notably larger train-test gap — possible overfitting.")
        else:
            print("    DQN generalisation gap is comparable to Z-Score baseline.")

    # ---- Verdict (significance-based) ----
    if not np.isnan(wilcox_p) and wilcox_p < 0.05 and avg_dqn > avg_z:
        verdict = ("DQN demonstrates statistically significant superior "
                   "risk-adjusted performance (Wilcoxon p < 0.05).")
    elif not np.isnan(wilcox_p) and wilcox_p < 0.05 and avg_z > avg_dqn:
        verdict = ("Z-Score baseline outperforms with statistical significance — "
                   "DQN may be overfitting or the market regimes favour fixed rules.")
    elif avg_dqn > avg_z:
        verdict = ("DQN shows a positive but not statistically significant edge — "
                   "a larger sample or longer test period may clarify.")
    elif avg_z > avg_dqn:
        verdict = ("Z-Score baseline has a non-significant edge — "
                   "both strategies perform comparably on these pairs.")
    else:
        verdict = ("Both strategies perform identically — "
                   "DQN offers no clear systematic advantage.")
    print(f"\n  VERDICT: {verdict}")

    # Aggregate presentation chart
    plot_aggregate_summary(summary_df)


if __name__ == "__main__":
    main()
