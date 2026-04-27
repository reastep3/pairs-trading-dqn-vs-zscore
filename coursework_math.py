"""
coursework_math.py
==================
Mathematical core for the DQN vs Z-Score pairs-trading experiment.
Columbia University — Financial Economics (Arka Prava Bandyopadhyay)

Every function cites the exact lecture and slide(s) it implements.
Nothing here is invented — it is a direct translation of the lecture math
into code that drives the pairs trading analysis.

Lecture / slide map
-------------------
  #1/2 Building Blocks  8–9   : AR(1) process, stationarity, Var[x]=σ²/(1-ρ²)
  #1   Arbitrage         15   : Statistical arbitrage & cointegration
  #2   Arbitrage II     11–16 : Law of One Price, no-arbitrage, discount factor
  #3   Pricing           8–9  : p=E[mx], SDF, excess return pricing
  #4   Contingency       8–9  : E[R^e]=-R_f·Cov[m,R^e], idiosyncratic risk
  #5   Cross Sect / TS  17–29 : Predictability, persistence, variance decomp

  #5 Slide 19        : Log return definition and additivity
  #5 Slides 17, 20   : OLS predictability regression framework
  #5 Slides 18, 20   : AR(1) coefficient of the predictor
  #5 Slides 20–21    : Persistence build-up (geometric series)
  #5 Slide 17 / Table: Multi-horizon predictability table (Table 20.1 analog)
  #5 Slides 27, 29   : Present-value variance decomposition
  #2 Slides 11, 14   : Law of One Price spread test
  #5 Slide 33 + #4   : CAPM market-neutrality test
  #3 Slides 14–18    : Bellman equation / dynamic programming convergence
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ===========================================================================
# SLIDE 19 — Log Return Definition
#
#   r_t := log(p_t) - log(p_{t-1})
#
#   Key property (slide 19):
#     r_{t→t+h} = r_{t+1} + r_{t+2} + ... + r_{t+h}
#               = log(p_{t+h}) - log(p_t)
#
#   Used throughout: hedge ratio residuals, spread daily P&L, DQN reward.
# ===========================================================================

def log_return(log_prices: pd.Series) -> pd.Series:
    """
    Slide 19: one-period log return.

        r_t = log(p_t) - log(p_{t-1})

    Additivity: r_{t→t+h} = Σ_{k=1}^h r_{t+k}
    """
    return log_prices.diff().rename("log_return")


def forward_cumulative_return(spread: pd.Series, h: int) -> pd.Series:
    """
    Slide 19: h-period forward log return at each time t.

        r_{t→t+h} = log(p_{t+h}) - log(p_t)
                  = spread[t+h] - spread[t]   (for log-space spread)

    Implementation: spread.diff(h) gives spread[t] - spread[t-h].
    Shifting by -h gives spread[t+h] - spread[t], i.e. r_{t→t+h}.

    Used in the predictability regressions (slide 17 Table 20.1 analog).
    """
    return spread.diff(h).shift(-h).rename(f"fwd_ret_h{h}")


# ===========================================================================
# SLIDES 17, 20 — OLS Predictability Regression
#
#   Slide 17 (Table 20.1):  R_{t→t+k} = a + b·(D_t/P_t) + ε
#   Slide 20:               r_{t+1}    = b·x_t            + ε_{t+1}
#
#   Same framework, different predictor.  For pairs trading, x_t = z-score.
#
#   Standard errors: slide 17 footnote says "GMM to correct for
#   heteroskedasticity and serial correlation."  We use Newey-West HAC.
# ===========================================================================

def ols_predict(y: pd.Series, x: pd.Series,
                newey_west_lags: int = 0) -> dict:
    """
    Slides 17, 20: OLS regression  y_t = a + b·x_t + ε_t

    For multi-horizon regressions with overlapping observations (h > 1),
    set newey_west_lags = h - 1 (same HAC correction as slide 17's Table 20.1
    footnote: "standard errors use GMM to correct for serial correlation").

    Returns
    -------
    dict : a, b, se_b, tstat, R2, nobs
    """
    combined = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(combined) < 10:
        return {"a": np.nan, "b": np.nan, "se_b": np.nan,
                "tstat": np.nan, "R2": np.nan, "nobs": len(combined)}

    Y = combined["y"].values
    X = sm.add_constant(combined["x"].values)

    try:
        if newey_west_lags > 0:
            fit = sm.OLS(Y, X).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": newey_west_lags, "use_correction": True},
            )
        else:
            fit = sm.OLS(Y, X).fit()
        return {
            "a":     float(fit.params[0]),
            "b":     float(fit.params[1]),
            "se_b":  float(fit.bse[1]),
            "tstat": float(fit.tvalues[1]),
            "R2":    float(fit.rsquared),
            "nobs":  int(fit.nobs),
        }
    except Exception:
        return {"a": np.nan, "b": np.nan, "se_b": np.nan,
                "tstat": np.nan, "R2": np.nan, "nobs": len(combined)}


# ===========================================================================
# SLIDES 18, 20 — AR(1) Coefficient of the Predictor / Spread
#
#   Slide 18: D/P has AR(1) coefficient ρ ≈ 0.9  (very persistent)
#   Slide 20: If x_{t+1} = ρ·x_t + δ_{t+1}, then the h-horizon predictive
#             coefficient is b·Σρ^k  (slides 20–21).
#
#   For pairs trading: the z-score IS the predictor x_t.
#   We need ρ < 1 (stationarity) for mean reversion to be possible.
#   The ADF test in cointegration testing is literally testing H₀: ρ = 1.
# ===========================================================================

def ar1_fit(series: pd.Series) -> dict:
    """
    Slides 18, 20: AR(1) fit to the z-score (or any predictor).

        x_{t+1} = ρ · x_t + δ_{t+1}

    Slide 18 notes D/P has ρ ≈ 0.9.  For pairs trading spreads,
    we want ρ << 1 (fast mean reversion = short optimal holding period).

    Also computes the half-life: number of days for the spread to revert
    halfway to zero.  half_life = -log(2) / log(ρ).

    Returns
    -------
    dict : rho, se_rho, tstat, half_life_days
    """
    s = series.dropna()
    x_lag = s.shift(1)
    result = ols_predict(s, x_lag, newey_west_lags=0)
    rho = result["b"]

    if 0 < rho < 1:
        half_life = -np.log(2) / np.log(rho)
    elif rho <= 0:
        half_life = 1.0    # over-shoots mean — immediate reversion
    else:
        half_life = np.inf  # unit root / explosive

    # Building Blocks slide 9: Var[x_t] = σ²_ε / (1 - ρ²)
    # Compute innovation variance from AR(1) residuals
    residuals = s.values[1:] - rho * s.values[:-1]
    innovation_var = float(np.var(residuals, ddof=1))

    return {
        "rho":            round(float(rho),          4),
        "se_rho":         round(float(result["se_b"]), 4),
        "tstat":          round(float(result["tstat"]), 3),
        "half_life_days": round(float(half_life),     1),
        "innovation_var": round(innovation_var,       6),
    }


# ===========================================================================
# SLIDES 20–21 — Persistence Build-up
#
#   Slides 20–21 derive:
#
#     ∂E[r_{t→t+h} | x_t] / ∂x_t = b · Σ_{k=0}^{h-1} ρ^k
#
#   As h → ∞:  b · 1/(1-ρ)  (geometric series limit — slide 21 figure)
#
#   For D/P with ρ = 0.9: coefficient grows dramatically with horizon,
#   explaining why R² rises from 15% (1yr) to 60% (5yr) in Table 20.1.
#
#   For a pairs-trading z-score with ρ = 0.3: saturates in ~5 days.
#   → Optimal strategy horizon is SHORT.  Rule of thumb: trade for ≈ half-life.
# ===========================================================================

def persistence_buildup(b: float, rho: float,
                         max_horizon: int = 60) -> pd.DataFrame:
    """
    Slides 20–21: Cumulative predictive coefficient at each horizon h.

        coef(h) = b · Σ_{k=0}^{h-1} ρ^k = b · (1 - ρ^h) / (1 - ρ)

    limit (h→∞) = b / (1 - ρ)   [slide 21 formula]

    Returns DataFrame: horizon, cumulative_coef, fraction_of_limit
    """
    hs = np.arange(1, max_horizon + 1, dtype=float)
    if abs(rho - 1.0) < 1e-9:
        geo = hs
        limit = np.inf
    else:
        geo   = (1.0 - rho ** hs) / (1.0 - rho)
        limit = b / (1.0 - rho)

    coefs = b * geo
    frac  = coefs / limit if np.isfinite(limit) else np.full_like(coefs, np.nan)

    return pd.DataFrame({
        "horizon":           hs.astype(int),
        "cumulative_coef":   coefs,
        "fraction_of_limit": frac,
    })


# ===========================================================================
# SLIDE 17 (Table 20.1 analog) — Multi-Horizon Predictability Table
#
#   Slide 17, Table 20.1: OLS regressions of k-year NYSE excess returns on D/P:
#       R_{t→t+k} = a + b·(D_t/P_t) + ε      b rises with k, R² rises 15%→60%
#
#   Our analog: regress h-day spread return on today's z-score:
#       r_{t→t+h} = a + b·z_t + ε_{t+h}
#
#   Predicted sign: b < 0  (z-score negatively predicts future spread return)
#       High z → spread expected to fall → mean reversion
#       Contrast with slide 17: b > 0 for D/P (high D/P → market rises)
#
#   Both capture the same logic: a "value" signal predicts a corrective return.
#   For D/P it is the equity risk premium; for pairs it is stat-arb convergence.
#
#   R²-horizon pattern differs:
#       D/P:    R² rises with h  (slow-moving predictor, slide 21's build-up)
#       z-score: R² peaks at short h then declines  (fast mean-reverting spread)
# ===========================================================================

def predictability_table(spread: pd.Series,
                          spread_returns: pd.Series,
                          z_scores: pd.Series,
                          horizons: list = None) -> pd.DataFrame:
    """
    Slide 17 (Table 20.1 analog): multi-horizon predictability regressions.

    For each horizon h, regresses the forward h-period cumulative spread return
    on the current z-score, using Newey-West HAC standard errors with h-1 lags
    (same correction as slide 17's GMM footnote).

    Parameters
    ----------
    spread        : log-space spread series (log P1 - β·log P2 - α)
    spread_returns: daily first-difference of spread
    z_scores      : rolling z-score of spread  [the predictor x_t]
    horizons      : list of forecast horizons in trading days

    Returns
    -------
    DataFrame : horizon_days, b, se_b, tstat, R2, n_obs
    """
    if horizons is None:
        horizons = [1, 5, 10, 20, 40]

    base = pd.DataFrame({
        "spread": spread,
        "z":      z_scores,
    }).dropna()

    rows = []
    for h in horizons:
        # Slide 19: r_{t→t+h} = spread[t+h] - spread[t]
        cum_ret = forward_cumulative_return(base["spread"], h)

        result = ols_predict(cum_ret, base["z"],
                             newey_west_lags=max(1, h - 1))
        rows.append({
            "horizon_days": int(h),
            "b":            round(result["b"],    5),
            "se_b":         round(result["se_b"], 5),
            "tstat":        round(result["tstat"], 2),
            "R2":           round(result["R2"],    4),
            "n_obs":        result["nobs"],
        })

    return pd.DataFrame(rows)


# ===========================================================================
# SLIDES 27, 29 — Present-Value Variance Decomposition
#
#   Slide 27 derives (for the market):
#     Var[p_t - d_t] = Cov[p-d, Σρ^{j-1}·Δd_{t+j}]
#                    - Cov[p-d, Σρ^{j-1}·r_{t+j}]
#
#   Slide 29 upshot: since D/P does NOT predict Δd (dividend growth),
#   the return covariance term must account for all of Var[p-d].
#   → High prices ↔ low expected returns (not high expected dividends).
#
#   Pairs trading analog:
#     Two cointegrated stocks share the same fundamental cashflow driver,
#     so ΔD_1 ≈ ΔD_2 and the dividend-growth covariance cancels.
#     Therefore: Var[z_t] ≈ -Cov[z_t, Σρ^{j-1}·spread_return_{t+j}]
#
#   This is the ECONOMIC JUSTIFICATION for pairs trading:
#     if cointegration holds (no fundamental divergence),
#     ALL variation in the spread must be explained by return predictability —
#     the spread MUST mean-revert.  The fraction below should → 100%.
# ===========================================================================

def variance_decomposition(z_scores: pd.Series,
                            spread_returns: pd.Series,
                            rho: float,
                            max_horizon: int = 60) -> dict:
    """
    Slides 27, 29: fraction of Var[z] explained by discounted future returns.

    Computes: C = Σ_{j=1}^H ρ^{j-1} · Cov[z_t, spread_return_{t+j}]

    For a perfectly cointegrated pair: C ≈ -Var[z_t]  →  fraction ≈ 1.0
    Departure from 1.0 signals either non-stationarity or fundamental divergence.

    Returns
    -------
    dict : var_z, return_cov_sum, return_fraction
    """
    common = pd.DataFrame({"z": z_scores, "sr": spread_returns}).dropna()
    z  = common["z"].values
    sr = common["sr"].values
    n  = len(z)

    var_z    = float(np.var(z, ddof=1))
    cov_sum  = 0.0

    for j in range(1, max_horizon + 1):
        if j >= n:
            break
        z_t    = z[:n - j]
        sr_fwd = sr[j:]
        if len(z_t) < 30:
            break
        cov_j   = float(np.cov(z_t, sr_fwd)[0, 1])
        cov_sum += (rho ** (j - 1)) * cov_j

    fraction = -cov_sum / var_z if var_z > 1e-12 else np.nan

    return {
        "var_z":           round(var_z,    6),
        "return_cov_sum":  round(cov_sum,  6),
        "return_fraction": round(fraction, 4),
    }


# ===========================================================================
# LECTURE #2 SLIDES 11, 14 — Law of One Price / No-Arbitrage Spread Test
#
#   Slide 11: "Two ways of acquiring the same payoffs cost the same."
#   Slide 14: No arbitrage ⇔ every non-negative payoff with some upside
#             has positive price.
#
#   For cointegrated pairs, both assets track the same fundamentals.
#   The LOP implies the hedge-ratio-adjusted spread has zero expected
#   drift — deviations from LOP are temporary and must mean-revert.
#
#   Lecture 1 slide 15: "Combination of assets must mean-revert —
#   cointegration is key."  If the spread has significant drift, the
#   LOP is violated and mean-reversion strategies carry directional risk.
# ===========================================================================

def lop_spread_test(spread_returns: pd.Series) -> dict:
    """
    Lecture 2, slides 11/14 + Lecture 1, slide 15:
    Test whether the spread has zero expected drift (consistent with LOP).

    H₀: E[r_spread] = 0  (LOP holds, spread is drift-free)
    H₁: E[r_spread] ≠ 0  (LOP violated, directional risk)

    A rejection means the spread is trending, so mean-reversion trades
    carry the additional risk of trading against a persistent drift.
    """
    from scipy.stats import t as t_dist

    sr = spread_returns.dropna()
    n = len(sr)
    mean_ret = float(sr.mean())
    std_ret = float(sr.std(ddof=1))
    se = std_ret / np.sqrt(n) if n > 1 else np.inf
    t_stat = mean_ret / se if se > 0 else 0.0
    p_value = float(2 * (1 - t_dist.cdf(abs(t_stat), df=max(n - 1, 1))))
    annual_drift = mean_ret * 252

    return {
        "mean_daily_ret": round(mean_ret, 8),
        "annual_drift":   round(annual_drift, 4),
        "t_stat":         round(t_stat, 3),
        "p_value":        round(p_value, 4),
        "lop_holds":      bool(p_value > 0.05),
    }


# ===========================================================================
# LECTURE #5 SLIDE 33 (CAPM) + LECTURE #4 SLIDES 8–9 — Market Neutrality
#
#   Slide 33: E[Ri] = γ + β_{i,Rw} (E[Rw] - γ)
#     → Expected return of any asset = risk-free + beta × market premium
#
#   Slides 8–9: E[R^e] = -Rf · Cov[m, R^e]
#     → Expected excess return depends on covariance with the SDF.
#     → "If the payoff is uncorrelated with m, it pays the risk-free rate
#        even if super volatile.  Idiosyncratic risk is not priced."
#
#   For pairs trading: if the spread has β ≈ 0, the strategy is
#   market-neutral and any positive expected return is genuine alpha —
#   it cannot be explained by market beta exposure.
# ===========================================================================

def market_neutrality(spread_returns: pd.Series,
                      market_returns: pd.Series) -> dict:
    """
    Lecture 5 slide 33 (CAPM) + Lecture 4 slides 8-9:
    Regress spread returns on market returns.

        r_spread = α + β · r_market + ε

    β ≈ 0 confirms the pair is market-neutral.
    α > 0 indicates genuine statistical-arbitrage alpha that
    cannot be attributed to the market risk premium.
    """
    combined = pd.concat([
        spread_returns.rename("spread"),
        market_returns.rename("market"),
    ], axis=1).dropna()

    if len(combined) < 30:
        return {"beta": np.nan, "alpha_daily": np.nan,
                "alpha_annual": np.nan, "beta_tstat": np.nan, "R2": np.nan}

    result = ols_predict(combined["spread"], combined["market"])

    return {
        "beta":         round(float(result["b"]),           4),
        "alpha_daily":  round(float(result["a"]),           8),
        "alpha_annual": round(float(result["a"]) * 252,    4),
        "beta_tstat":   round(float(result["tstat"]),       3),
        "R2":           round(float(result["R2"]),          4),
    }


# ===========================================================================
# LECTURE #3 SLIDES 14–18 — Bellman Equation / Dynamic Programming
#
#   Slide 14: The infinite-horizon investor solves
#       V(W_t) = max_ξ { u(c_t) + β E[V(W_{t+1})] }
#
#   This is the Bellman equation.  The DQN solves the analogous problem
#   for the action-value function Q:
#       Q*(s, a) = E[ r + γ · max_a' Q*(s', a') ]
#
#   The Bellman error δ = Q(s,a) - [r + γ max_a' Q_target(s',a')]
#   measures how far the agent is from the optimal policy.
#   A converged agent has near-zero mean Bellman error.
#
#   This function provides a diagnostic: given the trained agent's
#   Q-values and observed (s, a, r, s') transitions, compute the
#   residual Bellman error to assess convergence quality.
# ===========================================================================

def bellman_convergence(q_log: list, step_log: list,
                         rewards: list, gamma: float) -> dict:
    """
    Lecture 3, slides 14-18: Bellman equation convergence diagnostic.

    Given the agent's Q-values from evaluation, compute how well they
    satisfy the Bellman optimality equation:

        Q*(s,a) = r + γ · max_a' Q*(s',a')

    A well-converged agent will have:
      - Mean Bellman error ≈ 0  (unbiased)
      - Low std of Bellman error (consistent)

    Parameters
    ----------
    q_log    : list of dicts with keys dqn_q_hold, dqn_q_enter_long,
               dqn_q_enter_short, dqn_q_exit (one per timestep)
    step_log : list of env step dicts; uses dqn_action_resolved so the
               Q-value indexed matches the action actually executed
               (env masking may replace argmax with HOLD)
    rewards  : list of float rewards (one per timestep, same length)
    gamma    : discount factor (DQN_GAMMA)
    """
    import numpy as np

    n = min(len(q_log), len(step_log), len(rewards))
    if n < 2:
        return {"mean_bellman_error": np.nan, "std_bellman_error": np.nan,
                "mean_abs_error": np.nan, "n_steps": 0}

    action_idx = {"hold": 0, "enter_long": 1, "enter_short": 2, "exit": 3}
    errors = []
    for t in range(n - 1):
        q_t = q_log[t]
        q_next = q_log[t + 1]

        q_vals_t = [q_t["dqn_q_hold"], q_t["dqn_q_enter_long"],
                    q_t["dqn_q_enter_short"], q_t["dqn_q_exit"]]
        # Q(s_t, a_t) — indexed by the action actually executed after masking
        a_t = action_idx.get(step_log[t].get("dqn_action_resolved", "hold"), 0)
        q_chosen = q_vals_t[a_t]

        # max_a' Q(s_{t+1}, a') — the RHS of Bellman
        q_vals_next = [q_next["dqn_q_hold"], q_next["dqn_q_enter_long"],
                       q_next["dqn_q_enter_short"], q_next["dqn_q_exit"]]
        q_max_next = max(q_vals_next)

        # Bellman error: δ = Q(s,a) - [r + γ max Q(s',a')]
        bellman_target = rewards[t] + gamma * q_max_next
        error = q_chosen - bellman_target
        errors.append(error)

    errors = np.array(errors)

    return {
        "mean_bellman_error": round(float(errors.mean()), 6),
        "std_bellman_error":  round(float(errors.std()),  6),
        "mean_abs_error":     round(float(np.abs(errors).mean()), 6),
        "n_steps":            len(errors),
    }
