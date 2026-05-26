# =============================================================================
# main.py — Algorithmic Trading Strategy: Complete Pipeline
#
# Runs the full end-to-end pipeline in one command:
#   Step 1 — Fetch / load historical price data
#   Step 2 — Full-period backtest (all metrics + equity curve)
#   Step 3 — Walk-Forward Analysis (IS/OOS fold table + WFA chart)
#   Step 4 — Robustness Score (4-component breakdown + gauge chart)
#
# Usage:
#   python main.py                          # default settings from config.py
#   python main.py --symbol MSFT            # different stock
#   python main.py --capital 50000          # different starting capital
#   python main.py --start 2019-01-01       # different start date
#   python main.py --skip-wfa               # skip WFA (quick smoke-test)
#   python main.py --no-plot                # disable chart saving
#
# All results are printed to stdout. Charts are saved to results/.
# =============================================================================

import os
import sys
import time
import argparse

# Ensure project root is on path regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from config import (
    SYMBOL, START_DATE, END_DATE, STARTING_CAPITAL,
    FAST_SMA, SLOW_SMA, TREND_SMA,
    RSI_PERIOD, RSI_ENTRY_MAX, RSI_EXIT_MIN,
    STOP_LOSS_PCT, ROBUSTNESS_TARGET,
)
from data.fetch_data import fetch_data
from backtest.run_backtest import run_backtest
from backtest.walk_forward import run_walk_forward
from analysis.robustness import compute_robustness_score


# =============================================================================
# Banner
# =============================================================================
BANNER = r"""
  ╔══════════════════════════════════════════════════════════╗
  ║       Algorithmic Trading Strategy — Full Pipeline       ║
  ║       SMA/RSI Hybrid  |  Backtrader  |  Walk-Forward     ║
  ╚══════════════════════════════════════════════════════════╝
"""


def _header(step: int, title: str):
    print(f"\n{'─' * 60}")
    print(f"  STEP {step}: {title}")
    print(f"{'─' * 60}")


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{int(s // 60)}m {int(s % 60)}s"


# =============================================================================
# Argument parser
# =============================================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full algorithmic trading strategy pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol",   type=str,   default=SYMBOL,
                        help="Yahoo Finance ticker symbol")
    parser.add_argument("--start",    type=str,   default=START_DATE,
                        help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end",      type=str,   default=END_DATE,
                        help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--capital",  type=float, default=STARTING_CAPITAL,
                        help="Starting capital in USD")
    parser.add_argument("--skip-wfa", action="store_true",
                        help="Skip Walk-Forward Analysis (faster, no robustness score)")
    parser.add_argument("--no-plot",  action="store_true",
                        help="Disable saving charts to results/")
    parser.add_argument("--force-download", action="store_true",
                        help="Force re-download of price data (ignore cache)")
    return parser.parse_args()


# =============================================================================
# Summary table (printed at the very end)
# =============================================================================
def _print_final_summary(
    symbol: str,
    start_date: str,
    end_date: str,
    capital: float,
    full_metrics: dict,
    wfa_results: dict | None,
    robustness_score: float | None,
):
    line = "=" * 60
    pass_fail = (
        f"{robustness_score:.1f} (> {ROBUSTNESS_TARGET})"
        if robustness_score is not None
        else "N/A — WFA skipped"
    )
    wfa_score = (
        f"{wfa_results['wfa_efficiency']:.1f}%"
        if wfa_results
        else "N/A"
    )

    print(f"\n{line}")
    print(f"  RESULTS SUMMARY")
    print(line)
    print(f"  {'Stock Symbol':<28}: {symbol}")
    print(f"  {'Backtest Period':<28}: {start_date} -> {end_date}")
    print(f"  {'Starting Capital':<28}: ${capital:,.2f}")
    print(f"  {'Percentage Return on Capital':<28}: {full_metrics['pct_return']:+.2f}%")
    print(f"  {'Maximum Drawdown':<28}: {full_metrics['max_drawdown']:.2f}%")
    print(f"  {'Sharpe Ratio':<28}: {full_metrics['sharpe_ratio']:.3f}" if full_metrics['sharpe_ratio'] else f"  {'Sharpe Ratio':<28}: N/A")
    print(f"  {'Total Closed Trades':<28}: {full_metrics['total_trades']}")
    print(f"  {'Win Rate':<28}: {full_metrics['win_rate']:.1f}%")
    print(f"  {'Walk-Forward Analysis Score':<28}: {wfa_score}")
    print(f"  {'Robustness Score':<28}: {pass_fail}")
    print(f"{line}\n")


# =============================================================================
# Main pipeline
# =============================================================================
def main():
    t_total = time.time()
    args    = _parse_args()

    print(BANNER)
    print(f"  Symbol  : {args.symbol}")
    print(f"  Period  : {args.start} -> {args.end}")
    print(f"  Capital : ${args.capital:,.2f}")
    print(f"  Skip WFA: {args.skip_wfa}")
    print()

    # ------------------------------------------------------------------
    # Step 1 — Data
    # ------------------------------------------------------------------
    _header(1, "Fetch / Load Historical Price Data")
    t0 = time.time()

    df = fetch_data(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        force_download=args.force_download,
    )
    print(f"  Loaded {len(df):,} trading days  "
          f"({df.index[0].date()} -> {df.index[-1].date()})  "
          f"[{_elapsed(t0)}]")

    # ------------------------------------------------------------------
    # Step 2 — Full-period backtest
    # ------------------------------------------------------------------
    _header(2, "Full-Period Backtest")
    t0 = time.time()

    full_metrics = run_backtest(
        df=df,
        starting_capital=args.capital,
        printlog=False,
        save_plot=(not args.no_plot),
        plot_path="results/equity_curve.png",
        quiet=False,
    )
    print(f"  Backtest complete  [{_elapsed(t0)}]")
    if not args.no_plot:
        print("  Equity curve -> results/equity_curve.png")

    # ------------------------------------------------------------------
    # Step 3 — Walk-Forward Analysis
    # ------------------------------------------------------------------
    wfa_results = None

    if not args.skip_wfa:
        _header(3, "Walk-Forward Analysis")
        print("  This runs a grid search over IS windows — may take ~8 min ...")
        t0 = time.time()

        wfa_results = run_walk_forward(
            df=df,
            save_plot=(not args.no_plot),
            plot_path="results/wfa_results.png",
            quiet=False,
        )
        print(f"  WFA complete  [{_elapsed(t0)}]")
        if not args.no_plot:
            print("  WFA chart -> results/wfa_results.png")
    else:
        _header(3, "Walk-Forward Analysis  [SKIPPED]")
        print("  Pass --skip-wfa=False to enable.")

    # ------------------------------------------------------------------
    # Step 4 — Robustness Score
    # ------------------------------------------------------------------
    robustness_score = None

    if wfa_results is not None:
        _header(4, "Robustness Score")
        t0 = time.time()

        robustness_score, breakdown = compute_robustness_score(
            wfa_results=wfa_results,
            full_metrics=full_metrics,
            df=df,
            save_plot=(not args.no_plot),
            plot_path="results/robustness_score.png",
            quiet=False,
        )
        print(f"  Robustness Score complete  [{_elapsed(t0)}]")
        if not args.no_plot:
            print("  Robustness chart -> results/robustness_score.png")

        # Warn if score is below threshold (should not happen with current params)
        if not breakdown["passes_threshold"]:
            print(f"\n  [WARN] Robustness Score {robustness_score:.2f} < {ROBUSTNESS_TARGET}.")
            print("  Consider revisiting strategy params or adjusting the WFA grid.")
    else:
        _header(4, "Robustness Score  [SKIPPED — WFA not run]")

    # ------------------------------------------------------------------
    # Final summary table
    # ------------------------------------------------------------------
    _print_final_summary(
        symbol=args.symbol,
        start_date=str(df.index[0].date()),
        end_date=str(df.index[-1].date()),
        capital=args.capital,
        full_metrics=full_metrics,
        wfa_results=wfa_results,
        robustness_score=robustness_score,
    )

    print(f"  Total pipeline time: {_elapsed(t_total)}")
    print("  All results saved to  results/\n")


# =============================================================================
if __name__ == "__main__":
    main()
