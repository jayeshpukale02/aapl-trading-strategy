# =============================================================================
# config.py — Central Configuration File
# All strategy parameters and global settings live here.
# Change values here to adjust behaviour across the entire project.
# =============================================================================

# ---------------------------------------------------------------------------
# Data Settings
# ---------------------------------------------------------------------------
SYMBOL        = "AAPL"                # Stock ticker (Yahoo Finance format)
START_DATE    = "2018-01-01"          # Historical data start date
END_DATE      = "2024-12-31"          # Historical data end date
DATA_DIR      = "data"                # Folder where CSV files are stored
DATA_FILE     = f"data/{SYMBOL}.csv"  # Path to cached price CSV

# ---------------------------------------------------------------------------
# Broker / Capital Settings
# ---------------------------------------------------------------------------
STARTING_CAPITAL = 100_000.0   # Starting capital in USD
COMMISSION       = 0.001       # 0.1% commission per trade (realistic)
STAKE_FRACTION   = 0.95        # Fraction of available cash used per trade

# ---------------------------------------------------------------------------
# Strategy Parameters (used in strategy and WFA optimization)
# ---------------------------------------------------------------------------
# SMA periods
FAST_SMA   = 20      # Fast Simple Moving Average period (days)
SLOW_SMA   = 50      # Slow Simple Moving Average period (days)
TREND_SMA  = 200     # Long-term trend filter SMA period (days)

# RSI settings
RSI_PERIOD      = 14   # RSI lookback period
RSI_ENTRY_MAX   = 60   # Only enter if RSI is BELOW this (not overbought)
RSI_EXIT_MIN    = 75   # Exit if RSI RISES ABOVE this (overbought exit)

# Risk management
STOP_LOSS_PCT = 0.05   # Stop-loss: exit if price drops 5% below entry

# ---------------------------------------------------------------------------
# Walk-Forward Analysis (WFA) Settings
# ---------------------------------------------------------------------------
WFA_IS_YEARS   = 2     # In-sample window length (years)
WFA_OOS_MONTHS = 6     # Out-of-sample window length (months)

# Parameter search space for WFA optimisation
WFA_PARAM_GRID = {
    "fast_sma":  [10, 15, 20, 25, 30],
    "slow_sma":  [40, 50, 60, 70],
    "rsi_entry": [55, 60, 65],
    "rsi_exit":  [70, 75, 80],
}

# ---------------------------------------------------------------------------
# Robustness Score Thresholds
# ---------------------------------------------------------------------------
ROBUSTNESS_TARGET        = 75    # Minimum acceptable robustness score
DRAWDOWN_BEST            = 10.0  # Max drawdown % for full drawdown score (25 pts)
DRAWDOWN_WORST           = 30.0  # Max drawdown % for zero drawdown score (0 pts)
SENSITIVITY_PARAM_DELTA  = 0.10  # ±10% parameter perturbation for sensitivity test
