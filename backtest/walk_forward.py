# =============================================================================
# backtest/walk_forward.py — Walk-Forward Analysis (WFA) Engine
#
# Methodology:
#   1. Slice full history into sequential IS (in-sample) + OOS (out-of-sample) windows
#   2. Grid-search strategy parameters on IS window → pick best by Sharpe Ratio
#   3. Apply best params to the subsequent OOS window (unseen data)
#   4. Roll the window forward by OOS length and repeat
#   5. Aggregate results → compute WFA Efficiency Score
#
# WFA Efficiency Score = mean(OOS Sharpe) / mean(IS Sharpe) * 100
#   > 50  → good generalization
#   > 70  → excellent generalization
#
# Usage (standalone):
#   python backtest/walk_forward.py
#
# Usage (from other modules):
#   from backtest.walk_forward import run_walk_forward
#   wfa_results = run_walk_forward()
# =============================================================================

import os
import sys
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import (
    DATA_FILE, STARTING_CAPITAL, COMMISSION,
    WFA_IS_YEARS, WFA_OOS_MONTHS, SYMBOL,
    TREND_SMA,
)
from backtest.run_backtest import run_backtest


# =============================================================================
# Reduced param grid for WFA (keeps runtime manageable: ~36 combos/fold)
# =============================================================================
WFA_GRID = {
    "fast_sma":      [10, 15, 20, 25, 30],
    "slow_sma":      [40, 50, 60, 70],
    "rsi_entry_max": [55, 60, 65],
    "rsi_exit_min":  [70, 75, 80],
}


# =============================================================================
# Window splitter
# =============================================================================
def _get_wfa_windows(
    df: pd.DataFrame,
    is_years: int = WFA_IS_YEARS,
    oos_months: int = WFA_OOS_MONTHS,
) -> list:
    """
    Split `df` into sequential IS / OOS fold pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Full OHLCV DataFrame with DatetimeIndex.
    is_years : int
        In-sample window length in years.
    oos_months : int
        Out-of-sample window length in months. Also used as step size.

    Returns
    -------
    List of dicts: [{fold, is_start, is_end, oos_start, oos_end, is_df, oos_df}, ...]
    """
    from dateutil.relativedelta import relativedelta

    # Warmup bars prepended to OOS slice so indicators (esp. 200-day trend SMA)
    # can fully warm up before the actual OOS period starts.
    # During these prepended bars Backtrader stays in prenext() — no trades fire.
    WARMUP_BARS = TREND_SMA + 60   # 260 bars (~1 year) is sufficient

    windows = []
    fold    = 1

    # First IS window starts at the beginning of the data
    is_start = df.index[0]

    while True:
        # Compute window boundaries using calendar offsets
        is_end    = is_start + relativedelta(years=is_years)
        oos_start = is_end
        oos_end   = oos_start + relativedelta(months=oos_months)

        # Stop if OOS end exceeds available data
        if oos_end > df.index[-1]:
            break

        # Slice DataFrames
        is_df  = df.loc[(df.index >= is_start) & (df.index < is_end)].copy()
        oos_df = df.loc[(df.index >= oos_start) & (df.index < oos_end)].copy()

        # Need enough bars for indicator warmup (at least TREND_SMA + 50 bars)
        min_bars = TREND_SMA + 50
        if len(is_df) < min_bars or len(oos_df) < 20:
            is_start = is_start + relativedelta(months=oos_months)
            fold += 1
            continue

        # Pad OOS slice with the last WARMUP_BARS rows from IS data.
        # This ensures all indicators warm up before OOS bars begin,
        # so next() fires correctly during the OOS period.
        warmup_rows  = min(WARMUP_BARS, len(is_df))
        oos_df_padded = pd.concat([is_df.tail(warmup_rows), oos_df])

        windows.append({
            "fold":         fold,
            "is_start":     is_start.date(),
            "is_end":       is_end.date(),
            "oos_start":    oos_start.date(),
            "oos_end":      oos_end.date(),
            "is_df":        is_df,
            "oos_df":       oos_df,
            "oos_df_padded": oos_df_padded,
        })

        # Step forward by OOS length
        is_start = is_start + relativedelta(months=oos_months)
        fold += 1

    return windows


# =============================================================================
# IS optimizer — grid search
# =============================================================================
def _optimize_is(is_df: pd.DataFrame, param_grid: dict = WFA_GRID) -> tuple:
    """
    Run a grid search over param_grid on the in-sample DataFrame.

    Returns
    -------
    (best_params dict, best_sharpe float, best_return float)
    """
    # Generate all valid parameter combinations
    keys   = list(param_grid.keys())
    values = list(param_grid.values())

    best_params = None
    best_score  = -999.0  # optimise by Sharpe Ratio
    best_return = 0.0

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))

        # Skip invalid combinations (fast must be strictly less than slow)
        if params["fast_sma"] >= params["slow_sma"]:
            continue

        try:
            m = run_backtest(
                df=is_df,
                fast_sma=params["fast_sma"],
                slow_sma=params["slow_sma"],
                rsi_entry_max=params["rsi_entry_max"],
                rsi_exit_min=params["rsi_exit_min"],
                save_plot=False,
                quiet=True,
            )
        except Exception:
            continue

        # Primary metric: Sharpe Ratio (penalise None/NaN as -999)
        sharpe = m.get("sharpe_ratio") or -999.0
        pct_ret = m.get("pct_return", 0.0)

        # Use a composite score: Sharpe dominates, return as tiebreaker
        score = sharpe if sharpe > -999.0 else (pct_ret / 100.0 - 10.0)

        if score > best_score:
            best_score  = score
            best_params = params
            best_return = pct_ret

    # Fallback: if no valid params found, use defaults
    if best_params is None:
        best_params = {
            "fast_sma": 20, "slow_sma": 50,
            "rsi_entry_max": 60, "rsi_exit_min": 75,
        }
        best_score  = -999.0
        best_return = 0.0

    return best_params, best_score, best_return


# =============================================================================
# Core WFA runner
# =============================================================================
def run_walk_forward(
    df: pd.DataFrame = None,
    is_years: int = WFA_IS_YEARS,
    oos_months: int = WFA_OOS_MONTHS,
    param_grid: dict = None,
    save_plot: bool = True,
    plot_path: str = "results/wfa_results.png",
    quiet: bool = False,
) -> dict:
    """
    Run Walk-Forward Analysis and return aggregated results.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Full OHLCV DataFrame. Loaded from DATA_FILE if None.
    is_years : int
        In-sample window length in years.
    oos_months : int
        Out-of-sample window length in months (also the step size).
    param_grid : dict, optional
        Parameter search grid. Uses WFA_GRID if None.
    save_plot : bool
        Save WFA summary bar chart if True.
    plot_path : str
        Output path for WFA chart.
    quiet : bool
        Suppress printed output if True.

    Returns
    -------
    dict with keys:
        folds            - list of per-fold result dicts
        wfa_efficiency   - OOS/IS Sharpe efficiency score (%)
        mean_oos_return  - mean OOS return across all folds (%)
        mean_is_return   - mean IS return across all folds (%)
        positive_folds   - number of folds with positive OOS return
        total_folds      - total number of folds
    """
    if df is None:
        df = pd.read_csv(DATA_FILE, index_col="Date", parse_dates=True)

    if param_grid is None:
        param_grid = WFA_GRID

    # ------------------------------------------------------------------
    # Generate windows
    # ------------------------------------------------------------------
    windows = _get_wfa_windows(df, is_years=is_years, oos_months=oos_months)

    if not windows:
        raise ValueError(
            "No valid WFA windows found. Check date range and IS/OOS lengths."
        )

    if not quiet:
        combo_count = sum(
            1 for combo in itertools.product(*param_grid.values())
            if dict(zip(param_grid.keys(), combo))["fast_sma"]
               < dict(zip(param_grid.keys(), combo))["slow_sma"]
        )
        print(f"\n[WFA] {len(windows)} folds x {combo_count} param combos "
              f"= {len(windows) * combo_count} backtests")
        print(f"[WFA] IS: {is_years}yr  |  OOS: {oos_months}mo  |  "
              f"Step: {oos_months}mo\n")

    # ------------------------------------------------------------------
    # Run each fold
    # ------------------------------------------------------------------
    fold_results = []

    for w in windows:
        fold = w["fold"]
        if not quiet:
            print(f"  Fold {fold:>2} | IS: {w['is_start']} -> {w['is_end']} "
                  f"({len(w['is_df'])} bars) | "
                  f"OOS: {w['oos_start']} -> {w['oos_end']} "
                  f"({len(w['oos_df'])} bars)")
            print(f"         Optimising IS ... ", end="", flush=True)

        # --- Optimise on IS ---
        best_params, is_sharpe, is_return = _optimize_is(w["is_df"], param_grid)

        if not quiet:
            print(f"best params: {best_params}  IS Sharpe={is_sharpe:.3f}")
            print(f"         Running OOS ... ", end="", flush=True)

        # --- Evaluate on OOS ---
        # Use oos_df_padded: IS tail prepended so indicators warm up before
        # the OOS period starts. Trades only fire during the true OOS window.
        try:
            oos_m = run_backtest(
                df=w["oos_df_padded"],
                fast_sma=best_params["fast_sma"],
                slow_sma=best_params["slow_sma"],
                rsi_entry_max=best_params["rsi_entry_max"],
                rsi_exit_min=best_params["rsi_exit_min"],
                save_plot=False,
                quiet=True,
            )
            oos_return = oos_m["pct_return"]
            oos_sharpe = oos_m["sharpe_ratio"] or 0.0
            oos_trades = oos_m["total_trades"]
            oos_dd     = oos_m["max_drawdown"]
        except Exception as e:
            oos_return = 0.0
            oos_sharpe = 0.0
            oos_trades = 0
            oos_dd     = 0.0
            if not quiet:
                print(f"[WARN] OOS failed: {e}")

        if not quiet:
            sign = "+" if oos_return >= 0 else ""
            print(f"OOS Return={sign}{oos_return:.2f}%  "
                  f"Sharpe={oos_sharpe:.3f}  Trades={oos_trades}")

        fold_results.append({
            "fold":        fold,
            "is_start":    str(w["is_start"]),
            "is_end":      str(w["is_end"]),
            "oos_start":   str(w["oos_start"]),
            "oos_end":     str(w["oos_end"]),
            "best_params": best_params,
            "is_sharpe":   round(is_sharpe, 4),
            "is_return":   round(is_return, 2),
            "oos_sharpe":  round(oos_sharpe, 4),
            "oos_return":  round(oos_return, 2),
            "oos_trades":  oos_trades,
            "oos_dd":      round(oos_dd, 2),
        })

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    oos_returns = [r["oos_return"] for r in fold_results]
    is_returns  = [r["is_return"]  for r in fold_results]
    oos_sharpes = [r["oos_sharpe"] for r in fold_results]
    is_sharpes  = [r["is_sharpe"]  for r in fold_results if r["is_sharpe"] > -999.0]

    mean_oos_return = round(float(np.mean(oos_returns)), 2)
    mean_is_return  = round(float(np.mean(is_returns)),  2)
    mean_oos_sharpe = round(float(np.mean(oos_sharpes)), 4)
    mean_is_sharpe  = round(float(np.mean(is_sharpes)) if is_sharpes else 0.0, 4)

    positive_folds = sum(1 for r in oos_returns if r > 0)
    total_folds    = len(fold_results)

    # WFA Efficiency: how much of IS Sharpe carries over to OOS
    # Clamp IS denominator to avoid div-by-zero or meaningless negatives
    if mean_is_sharpe > 0:
        wfa_efficiency = round((mean_oos_sharpe / mean_is_sharpe) * 100, 1)
    elif mean_is_return > 0:
        wfa_efficiency = round((mean_oos_return / mean_is_return) * 100, 1)
    else:
        wfa_efficiency = 0.0

    # Clamp to [0, 100] for display purposes
    wfa_efficiency_clamped = max(0.0, min(100.0, wfa_efficiency))

    wfa_results = {
        "folds":           fold_results,
        "wfa_efficiency":  wfa_efficiency_clamped,
        "mean_oos_return": mean_oos_return,
        "mean_is_return":  mean_is_return,
        "mean_oos_sharpe": mean_oos_sharpe,
        "mean_is_sharpe":  mean_is_sharpe,
        "positive_folds":  positive_folds,
        "total_folds":     total_folds,
    }

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    if not quiet:
        _print_wfa_table(fold_results, wfa_results)

    # ------------------------------------------------------------------
    # Save WFA chart
    # ------------------------------------------------------------------
    if save_plot:
        _save_wfa_chart(fold_results, wfa_results, plot_path)

    return wfa_results


# =============================================================================
# Pretty printer
# =============================================================================
def _print_wfa_table(fold_results: list, agg: dict):
    sep = "-" * 90
    print(f"\n{'=' * 90}")
    print(f"  WALK-FORWARD ANALYSIS RESULTS — {SYMBOL}")
    print(f"{'=' * 90}")
    print(f"  {'Fold':>4}  {'IS Period':>23}  {'OOS Period':>23}  "
          f"{'IS Ret%':>8}  {'OOS Ret%':>8}  {'OOS Sharpe':>10}  {'Trades':>6}")
    print(sep)

    for r in fold_results:
        is_period  = f"{r['is_start']} -> {r['is_end']}"
        oos_period = f"{r['oos_start']} -> {r['oos_end']}"
        oos_sign   = "+" if r["oos_return"] >= 0 else ""
        is_sign    = "+" if r["is_return"]  >= 0 else ""
        print(
            f"  {r['fold']:>4}  {is_period:>23}  {oos_period:>23}  "
            f"{is_sign}{r['is_return']:>6.2f}%  "
            f"{oos_sign}{r['oos_return']:>6.2f}%  "
            f"{r['oos_sharpe']:>10.3f}  "
            f"{r['oos_trades']:>6}"
        )

    print(sep)
    print(f"\n  Mean IS Return   : {agg['mean_is_return']:+.2f}%")
    print(f"  Mean OOS Return  : {agg['mean_oos_return']:+.2f}%")
    print(f"  Mean IS Sharpe   : {agg['mean_is_sharpe']:.3f}")
    print(f"  Mean OOS Sharpe  : {agg['mean_oos_sharpe']:.3f}")
    print(f"  Positive Folds   : {agg['positive_folds']} / {agg['total_folds']}")
    print(f"  WFA Efficiency   : {agg['wfa_efficiency']:.1f}%")
    print(f"{'=' * 90}\n")


# =============================================================================
# WFA bar chart
# =============================================================================
def _save_wfa_chart(fold_results: list, agg: dict, plot_path: str):
    os.makedirs("results", exist_ok=True)

    folds      = [r["fold"]      for r in fold_results]
    is_returns = [r["is_return"] for r in fold_results]
    oos_returns= [r["oos_return"]for r in fold_results]

    x     = np.arange(len(folds))
    width = 0.38

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0f1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0f1117")

    # --- Bar chart: IS vs OOS returns ---
    bars_is  = ax1.bar(x - width/2, is_returns,  width, label="IS Return %",
                       color="#4fc3f7", alpha=0.85, edgecolor="#0f1117", linewidth=0.5)
    bars_oos = ax1.bar(x + width/2, oos_returns, width, label="OOS Return %",
                       color=[("#00e676" if v >= 0 else "#ff5252") for v in oos_returns],
                       alpha=0.85, edgecolor="#0f1117", linewidth=0.5)

    ax1.axhline(0, color="#ffffff", linewidth=0.6, linestyle="--", alpha=0.4)

    # Annotate OOS bars
    for bar, val in zip(bars_oos, oos_returns):
        sign = "+" if val >= 0 else ""
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.3 if val >= 0 else -0.8),
            f"{sign}{val:.1f}%",
            ha="center", va="bottom" if val >= 0 else "top",
            fontsize=7, color="#ffffff", alpha=0.85
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"F{f}" for f in folds], color="#aaaaaa", fontsize=9)
    ax1.set_ylabel("Return (%)", color="#aaaaaa", fontsize=10)
    ax1.tick_params(colors="#aaaaaa")
    ax1.grid(axis="y", color="#222222", linewidth=0.6, linestyle="--")
    for sp in ax1.spines.values():
        sp.set_edgecolor("#333333")

    legend = ax1.legend(facecolor="#1a1d27", edgecolor="#333333",
                        labelcolor="#cccccc", fontsize=9)

    ax1.set_title(
        f"{SYMBOL} Walk-Forward Analysis  |  "
        f"{agg['total_folds']} Folds  |  "
        f"WFA Efficiency: {agg['wfa_efficiency']:.1f}%  |  "
        f"Positive OOS Folds: {agg['positive_folds']}/{agg['total_folds']}",
        color="#ffffff", fontsize=11, pad=12
    )

    # --- Summary scoreboard ---
    ax2.axis("off")
    summary_text = (
        f"Mean IS Return: {agg['mean_is_return']:+.2f}%   |   "
        f"Mean OOS Return: {agg['mean_oos_return']:+.2f}%   |   "
        f"Mean OOS Sharpe: {agg['mean_oos_sharpe']:.3f}   |   "
        f"WFA Efficiency: {agg['wfa_efficiency']:.1f}%"
    )
    ax2.text(0.5, 0.5, summary_text, transform=ax2.transAxes,
             ha="center", va="center", fontsize=10,
             color="#cccccc",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#1a1d27",
                       edgecolor="#333333", alpha=0.9))

    plt.tight_layout(pad=1.5)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] WFA chart saved -> {plot_path}")


# =============================================================================
# Script entry point
# =============================================================================
if __name__ == "__main__":
    print("[INFO] Starting Walk-Forward Analysis ...")
    print("[INFO] This may take several minutes (grid search per fold) ...")
    wfa = run_walk_forward(save_plot=True)
    print(f"[DONE] WFA Efficiency Score : {wfa['wfa_efficiency']:.1f}%")
    print(f"[DONE] Positive OOS Folds   : {wfa['positive_folds']} / {wfa['total_folds']}")
