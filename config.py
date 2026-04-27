"""Central configuration for the DQN vs Z-Score pairs trading research."""

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
CANDIDATE_PAIRS = [
    # ── Extremely cointegrated (near-identical benchmark / product) ──
    ("SPY",  "IVV"),   # Both track S&P 500 — near-perfect cointegration
    ("GLD",  "IAU"),   # Both track gold spot — near-perfect cointegration
    ("AGG",  "BND"),   # Both track US aggregate bond index
    ("EFA",  "VEA"),   # Both track international developed equities
    ("XLE",  "VDE"),   # Both track US energy sector

    # ── Strongly cointegrated (sector duopolies / same commodity driver) ──
    ("XOM",  "CVX"),   # Integrated oil majors — same commodity driver
    ("KO",   "PEP"),   # Beverages duopoly — textbook pairs trade
    ("V",    "MA"),    # Payment network duopoly — identical revenue model
    ("UNP",  "CSX"),   # Class I railroads — same macro drivers
    ("LIN",  "APD"),   # Industrial gases duopoly — oligopoly pricing
    ("GS",   "MS"),    # Bulge-bracket investment banks
    ("HD",   "LOW"),   # Home improvement retail duopoly
    ("JPM",  "BAC"),   # Mega-cap commercial banks
    ("ADI",  "TXN"),   # Analog semiconductors — same end markets

    # ── Moderately cointegrated (same sector, different positioning) ──
    ("PG",   "CL"),    # Consumer staples — PG broader portfolio
    ("MCD",  "YUM"),   # Fast food — different franchise structures
    ("DE",   "CAT"),   # Heavy equipment — ag vs construction emphasis
    ("WMT",  "TGT"),   # Mass retail — different demographics
    ("NEE",  "DUK"),   # Large-cap utilities — regulated, same macro
    ("MMM",  "HON"),   # Diversified industrials — similar conglomerate model
    ("COP",  "EOG"),   # E&P oil companies — different basin exposure
    ("CME",  "ICE"),   # Exchange operators — different product mixes
    ("ABT",  "MDT"),   # Medical devices — different therapeutic focus
    ("COST", "KR"),    # Grocery / mass retail — different store formats

    # ── Weakly cointegrated (different business models / geographies) ──
    ("TSLA", "NIO"),   # EVs — US vs China, wildly different scale
    ("META", "SNAP"),  # Social media — monopoly vs niche, 100x revenue gap
    ("AMZN", "EBAY"),  # E-commerce — dominant platform vs declining marketplace
    ("AAPL", "MSFT"),  # Mega tech — hardware ecosystem vs enterprise software
    ("NFLX", "DIS"),   # Streaming — pure-play vs media conglomerate
    ("GOOGL","BIDU"),  # Internet search — US monopoly vs China market
]
# Maps "TICKER1-TICKER2" → expected cointegration stratum (for presentation)
PAIR_STRATA = {
    # Strong (14 pairs: 5 ETF-tracking + 9 sector duopolies)
    "SPY-IVV": "Strong", "GLD-IAU": "Strong", "AGG-BND": "Strong",
    "EFA-VEA": "Strong", "XLE-VDE": "Strong",
    "XOM-CVX": "Strong", "KO-PEP": "Strong", "V-MA": "Strong",
    "UNP-CSX": "Strong", "LIN-APD": "Strong", "GS-MS": "Strong",
    "HD-LOW": "Strong", "JPM-BAC": "Strong", "ADI-TXN": "Strong",
    # Moderate (10 pairs: same sector, different positioning)
    "PG-CL": "Moderate", "MCD-YUM": "Moderate", "DE-CAT": "Moderate",
    "WMT-TGT": "Moderate", "NEE-DUK": "Moderate", "MMM-HON": "Moderate",
    "COP-EOG": "Moderate", "CME-ICE": "Moderate", "ABT-MDT": "Moderate",
    "COST-KR": "Moderate",
    # Weak (6 pairs: different business models / geographies)
    "TSLA-NIO": "Weak", "META-SNAP": "Weak", "AMZN-EBAY": "Weak",
    "AAPL-MSFT": "Weak", "NFLX-DIS": "Weak", "GOOGL-BIDU": "Weak",
}

DATA_START  = "2018-01-01"
DATA_END    = "2024-12-31"
TRAIN_RATIO = 0.70          # 70 % train, 30 % test

# ---------------------------------------------------------------------------
# Spread / Z-score
# ---------------------------------------------------------------------------
ZSCORE_WINDOW = 60          # rolling window (trading days)

# ---------------------------------------------------------------------------
# Trading / simulation
# ---------------------------------------------------------------------------
INITIAL_CAPITAL  = 100_000.0
POSITION_SIZE    = 0.95      # fraction of capital deployed per trade
TRANSACTION_COST = 0.001     # one-way per leg (≈10 bps); round-trip = 0.004

# ---------------------------------------------------------------------------
# Z-Score baseline: grid-search space
# ---------------------------------------------------------------------------
ENTRY_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0]
EXIT_THRESHOLDS  = [0.0, 0.25, 0.5, 0.75, 1.0]
STOP_THRESHOLDS  = [3.5, 4.0]

# ---------------------------------------------------------------------------
# DQN hyperparameters
# ---------------------------------------------------------------------------
DQN_STATE_DIM      = 6
DQN_ACTION_DIM     = 4       # 0=Hold, 1=Enter Long, 2=Enter Short, 3=Exit
DQN_HIDDEN_DIMS    = [128, 64]
DQN_LR             = 3e-4
DQN_GAMMA          = 0.99
DQN_EPSILON_START  = 1.0
DQN_EPSILON_MIN    = 0.02
DQN_EPSILON_DECAY  = 0.992
DQN_BUFFER_SIZE    = 60_000
DQN_BATCH_SIZE     = 64
DQN_TARGET_UPDATE  = 250     # steps between target-network syncs
DQN_TRAIN_EPISODES = 500
DQN_STOP_LOSS      = 4.0     # force-exit if |z| exceeds this
