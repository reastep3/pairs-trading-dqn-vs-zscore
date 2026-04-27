# DQN vs Z-Score Pairs Trading

A research codebase that compares a **Deep Q-Network (DQN) reinforcement-learning agent** against a **classical Z-Score grid-search baseline** on 30 cointegrated equity pairs. Built around the Columbia *Financial Economics* curriculum — every quantitative diagnostic in the pipeline traces back to a specific lecture/slide, so the empirical results sit on top of textbook theory rather than alongside it.

> **Headline finding (30 pairs, out-of-sample 2018–2024, 70/30 train/test split):**
> DQN edges Z-Score on raw win count (16–14) and mean Sharpe (−0.22 vs −0.33), but the Wilcoxon signed-rank test gives p = 0.72 — **no statistically significant difference**. The DQN's larger train→test Sharpe gap (1.44 vs 0.74) suggests the gap is partly absorbed overfitting. The richer story is in the per-stratum breakdown.

---

## Table of Contents

1. [What this does](#what-this-does)
2. [Project layout](#project-layout)
3. [Quick start](#quick-start)
4. [Pipeline at a glance](#pipeline-at-a-glance)
5. [Methodology — coursework hooks](#methodology--coursework-hooks)
6. [Results — postbugfix run](#results--postbugfix-run)
7. [Reproducibility](#reproducibility)
8. [File reference](#file-reference)

---

## What this does

For each of 30 candidate equity pairs (stratified Strong / Moderate / Weak by cointegration strength) the pipeline:

1. Downloads adjusted-close prices from Yahoo Finance.
2. Runs the Engle–Granger cointegration test + ADF on the residual.
3. Estimates the hedge ratio β by OLS **on training data only** (no look-ahead).
4. Builds the log-spread and rolling Z-score.
5. Computes a battery of Lecture-5 diagnostics (AR(1), predictability table, persistence build-up, present-value variance decomposition, Law-of-One-Price test, CAPM market-neutrality).
6. Trains the **Z-Score baseline** by grid-searching `(entry, exit, stop)` thresholds on the training window.
7. Trains the **DQN agent** for 500 episodes on the training window.
8. Evaluates both strictly out-of-sample on the test window and reports a Bellman-error convergence diagnostic for the DQN.
9. Saves per-pair charts, equity curves, trade logs, decision-step appendices, and an aggregate presentation chart.

All Z-Score thresholds, hedge ratios, and DQN parameters are fit **only** on training data. Test-period numbers are honest out-of-sample.

---

## Project layout

```
.
├── main.py                  # Orchestrator — runs the full 30-pair pipeline
├── config.py                # All hyperparameters, candidate pairs, strata
├── data_utils.py            # Yahoo download, cointegration test, spread & z-score
├── coursework_math.py       # Lecture/slide-anchored mathematical primitives
├── zscore_baseline.py       # Grid-searched threshold strategy
├── trading_env.py           # Custom Gym-like environment for the DQN
├── dqn_agent.py             # Double-DQN with target net, replay buffer, Huber loss
├── metrics.py               # Sharpe, Sortino, Calmar, drawdown, trade stats
├── methodology_skeleton.docx # Write-up scaffold (paper / report)
├── requirements.txt
├── LICENSE                  # MIT
└── results/
    └── run_30pairs_postbugfix_<timestamp>/
        ├── aggregate_summary.png      # Headline chart across all 30 pairs
        ├── summary.csv                # Per-pair Sharpe / drawdown / diagnostics
        ├── <PAIR>_results.png         # 5-panel per-pair diagnostic chart
        ├── <PAIR>_equity_curves.csv   # Z-Score & DQN equity paths
        ├── <PAIR>_metrics.csv         # OOS performance metrics
        ├── <PAIR>_zscore_grid.csv     # Full grid-search results
        ├── <PAIR>_dqn_training.csv    # DQN episode-return curve
        ├── <PAIR>_predictability.csv  # Multi-horizon predictability table
        └── <PAIR>_appendix.csv        # Step-by-step decision log (both agents)
```

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/reastep3/pairs-trading-dqn-vs-zscore.git
cd pairs-trading-dqn-vs-zscore

# 2. (Recommended) virtual env
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline (~30 pairs × 500 DQN episodes — takes a while on CPU)
python main.py
```

Each run writes to `results/run_30pairs_postbugfix_<UTC-timestamp>/`, so re-runs never clobber previous output.

---

## Pipeline at a glance

```
                         CANDIDATE_PAIRS (config.py)
                                   │
                                   ▼
        download_data ─►  Engle–Granger + ADF cointegration tests
                                   │
                                   ▼
                       70 / 30  train / test split
                                   │
            ┌──────────────────────┴──────────────────────┐
            ▼                                             ▼
   Lecture-5 diagnostics                         Hedge-ratio OLS β (train only)
   (AR(1), persistence,                                   │
    PV variance decomp,                                   ▼
    LOP test, CAPM β)                       Build spread & rolling z-score
            │                                             │
            └──────────────────────┬──────────────────────┘
                                   ▼
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
            Z-Score grid search           DQN training (500 ep)
            (entry, exit, stop)           ε-greedy, replay buffer,
            on training data              Double-DQN, target net
                    │                             │
                    └──────────────┬──────────────┘
                                   ▼
                    Out-of-sample evaluation (test set)
                                   │
                                   ▼
        Per-pair report  ◄──── Sharpe / Sortino / Calmar / DD / trades
                                   │
                                   ▼
        Aggregate chart  ◄──── Wilcoxon, win counts, train→test gap
```

---

## Methodology — coursework hooks

Every diagnostic in `coursework_math.py` is wired to a specific lecture/slide. The point isn't to recompute the same numbers in two places — it's to make the empirical pipeline **legible as theory**.

| Lecture / Slide                       | Concept                                        | Implementation                       |
| ------------------------------------- | ---------------------------------------------- | ------------------------------------ |
| #1/2 Building Blocks (8–9)            | AR(1), stationarity, `Var[x] = σ²/(1−ρ²)`      | `ar1_fit`                            |
| #2 Arbitrage (11, 14)                 | Law of One Price spread test                   | `lop_spread_test`                    |
| #3 Pricing (8–9)                      | `p = E[mx]`, SDF, excess-return pricing        | reward design in `trading_env.py`    |
| #3 Pricing (14–18)                    | Bellman optimality / dynamic programming       | `DQNAgent.update`, `bellman_convergence` |
| #4 Contingency (8–9)                  | `E[Rᵉ] = −R_f·Cov[m, Rᵉ]`, idiosyncratic risk  | `market_neutrality` (CAPM β)         |
| #5 Predictability (17, 20)            | OLS regression framework                       | `ols_predict`                        |
| #5 (18, 20)                           | AR(1) coefficient of the predictor             | `ar1_fit`                            |
| #5 (20–21)                            | Persistence build-up (geometric series)        | `persistence_buildup`                |
| #5 Slide 17 / Table 20.1              | Multi-horizon predictability table             | `predictability_table`               |
| #5 (27, 29)                           | Present-value variance decomposition           | `variance_decomposition`             |
| #5 Slide 33 + #4                      | CAPM market-neutrality test                    | `market_neutrality`                  |
| #5 Slide 19                           | Log-return definition / additivity             | `log_return`                         |

### DQN — Bellman optimality concretely

The DQN solves the **same class** of dynamic-programming problem as the consumption-investment Bellman equation in Lecture 3:

```
Bellman:        V(W_t) = max_ξ { u(c_t) + β · E[V(W_{t+1})] }
Q-learning:     Q*(s,a) = E[ r(s,a) + γ · max_{a'} Q*(s', a') ]

Mapping:
  β   (subjective discount)         ↔  γ = 0.99
  u(c_t)                            ↔  daily P&L reward
  ξ   (investment decision)         ↔  action ∈ {hold, long, short, exit}
  Optimal ξ*                        ↔  argmax_a Q(s, a)
```

The Huber loss in `dqn_agent.py:DQNAgent.update` is literally the Bellman error. A separate `bellman_convergence` diagnostic in `coursework_math.py` checks how close the trained Q-network is to the fixed point of the Bellman operator on the test trajectory.

---

## Results — postbugfix run

The numbers below come from `results/run_30pairs_postbugfix_2026-04-22_13-51-56/summary.csv`. All are **out-of-sample**.

### Aggregate

| Metric                              | Z-Score | DQN     |
| ----------------------------------- | ------- | ------- |
| Pairs won (Sharpe winner)           | **14**  | **16**  |
| Mean Sharpe                         | −0.331  | −0.215  |
| Mean total return                   | −6.0%   | −9.1%   |
| Mean max drawdown                   | −18.4%  | −26.3%  |
| Train→Test Sharpe gap (overfitting) | 0.74    | 1.44    |

**Wilcoxon signed-rank test on Sharpe differences:** *p = 0.72*. The DQN's nominal edge is **not** statistically significant — the larger train→test gap suggests a meaningful chunk of the apparent advantage is overfitting to the training period.

### By cointegration stratum

| Stratum  | n  | Z mean Sharpe | DQN mean Sharpe | Wins (Z–DQN) |
| -------- | -- | ------------- | --------------- | ------------ |
| Strong   | 14 | −0.539        | −0.285          | 5–9          |
| Moderate | 10 | −0.132        | −0.067          | 5–5          |
| Weak     | 6  | −0.179        | −0.299          | 4–2          |

Two patterns worth pulling out:

- **Strong stratum** — DQN dominates 9–5 in win count. Plausible reading: clean mean-reverting spreads have enough signal for the agent to learn timing improvements over a fixed threshold.
- **Weak stratum** — Z-Score wins 4–2. When cointegration is shaky the DQN has less true signal to exploit and its extra flexibility starts hurting it.

The full per-pair table is in `results/<run>/summary.csv`. The presentation-ready aggregate chart (Sharpe by pair, win counts by stratum, cointegration vs strategy edge, headline statistics) is at `results/<run>/aggregate_summary.png`.

### Per-pair diagnostics

For every pair the run produces a 5-panel chart (`<PAIR>_results.png`):

1. Out-of-sample equity curves (Z-Score, DQN, dollar-neutral B&H of the spread).
2. Z-score over the test period with the chosen entry/exit thresholds overlaid.
3. DQN training curve (episode rewards + rolling average).
4. Drawdown comparison.
5. Rolling 60-day Sharpe.

Plus full CSVs for the equity paths, the threshold grid, the DQN training curve, the predictability table, and a step-by-step decision appendix that logs both agents' actions, rewards, and Q-values per test day.

---

## Reproducibility

- Single random seed (`SEED = 42`) across `random`, `numpy`, and `torch` (incl. `cudnn.deterministic`).
- Hedge ratio is OLS on the training window only — never refit on test.
- Z-Score thresholds are grid-searched on training Sharpe only.
- DQN target network is sync'd every 250 steps; replay buffer is 60k.
- Each run writes to a fresh, timestamped subdirectory so prior outputs are preserved.

> **Note on results folders.** This repo intentionally ships only the `run_30pairs_postbugfix_*` directory. Earlier runs (`run_20pairs_prior`, `run_30pairs_final`) were superseded by a bug fix and are excluded via `.gitignore` to avoid confusion.

---

## File reference

| File                        | Purpose                                                                  |
| --------------------------- | ------------------------------------------------------------------------ |
| `main.py`                   | End-to-end orchestrator. Run this.                                       |
| `config.py`                 | Hyperparameters, candidate pairs, stratum mapping, dates, capital.       |
| `data_utils.py`             | Yahoo download, Engle–Granger / ADF, hedge ratio, spread, rolling z.     |
| `coursework_math.py`        | Lecture-anchored mathematical primitives (every function cites a slide). |
| `zscore_baseline.py`        | Grid-searched threshold backtest with stop-loss & step log.              |
| `trading_env.py`            | Custom environment — earn-then-act P&L, action masking, stop-loss, 6-dim state. |
| `dqn_agent.py`              | Double DQN: replay buffer, target net, Huber loss, ε-greedy.             |
| `metrics.py`                | Sharpe, Sortino, Calmar, drawdown, win-rate, profit factor.              |
| `methodology_skeleton.docx` | Write-up scaffold for the accompanying paper / report.                   |
| `requirements.txt`          | Pinned minimum versions for Python deps.                                 |

---

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Mathematical scaffolding follows the Columbia *Financial Economics* curriculum (Arka Prava Bandyopadhyay). Empirical pipeline is original work; price data is sourced from Yahoo Finance via `yfinance`.
