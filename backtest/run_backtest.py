# =============================================================================
# backtest/run_backtest.py — Full-Period Backtest Runner
#
# Runs the SMARSIStrategy over the complete historical dataset and reports:
#   - Percentage Return on Capital       (required deliverable)
#   - Maximum Drawdown %                 (required deliverable)
#   - Sharpe Ratio
#   - Win Rate, Profit Factor, Trade Count
#   - Saves an equity-curve PNG to results/equity_curve.png
#
# Usage (standalone):
#   python backtest/run_backtest.py
#
# Usage (from other modules):
#   from backtest.run_backtest import run_backtest
#   metrics = run_backtest()
# =============================================================================

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtrader as bt
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import (
    DATA_FILE, STARTING_CAPITAL, COMMISSION,
    FAST_SMA, SLOW_SMA, TREND_SMA,
    RSI_PERIOD, RSI_ENTRY_MAX, RSI_EXIT_MIN,
    STOP_LOSS_PCT, STAKE_FRACTION, SYMBOL,
)
from strategy.my_strategy import SMARSIStrategy


# =============================================================================
# Equity-tracking strategy wrapper
# =============================================================================
class _TrackingStrategy(SMARSIStrategy):
    """
    Thin subclass of SMARSIStrategy that appends (date, portfolio_value)
    to an external list on every bar — used to build the equity curve.
    We pass the list in via `equity_log` param.
    NOTE: Backtrader's MetaParams requires we list ALL params explicitly
    (including parent params) when extending in a subclass.
    """
    # Re-declare all parent params plus the new equity_log param.
    # Backtrader's metaclass merges parent + child params automatically
    # when params is defined as a tuple in the subclass body.
    params = (
        ("fast_sma", FAST_SMA),
        ("slow_sma", SLOW_SMA),
        ("trend_sma", TREND_SMA),
        ("rsi_period", RSI_PERIOD),
        ("rsi_entry_max", RSI_ENTRY_MAX),
        ("rsi_exit_min", RSI_EXIT_MIN),
        ("stop_loss_pct", STOP_LOSS_PCT),
        ("stake_fraction", STAKE_FRACTION),
        ("printlog", False),
        ("equity_log", None),
    )

    def next(self):
        super().next()
        if self.p.equity_log is not None:
            self.p.equity_log.append(
                (self.datas[0].datetime.date(0), self.broker.getvalue())
            )

    def prenext(self):
        """Also capture value during warm-up bars."""
        if self.p.equity_log is not None:
            self.p.equity_log.append(
                (self.datas[0].datetime.date(0), self.broker.getvalue())
            )


# =============================================================================
# Core backtest function
# =============================================================================
def run_backtest(
    df: pd.DataFrame = None,
    fast_sma: int = FAST_SMA,
    slow_sma: int = SLOW_SMA,
    trend_sma: int = TREND_SMA,
    rsi_period: int = RSI_PERIOD,
    rsi_entry_max: float = RSI_ENTRY_MAX,
    rsi_exit_min: float = RSI_EXIT_MIN,
    stop_loss_pct: float = STOP_LOSS_PCT,
    stake_fraction: float = STAKE_FRACTION,
    starting_capital: float = STARTING_CAPITAL,
    commission: float = COMMISSION,
    printlog: bool = False,
    save_plot: bool = True,
    plot_path: str = "results/equity_curve.png",
    quiet: bool = False,
) -> dict:
    """
    Run a single full-period backtest and return a metrics dictionary.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Pre-loaded OHLCV DataFrame. If None, loads from DATA_FILE.
    fast_sma, slow_sma, trend_sma : int
        SMA periods for the strategy.
    rsi_period : int
        RSI lookback period.
    rsi_entry_max, rsi_exit_min : float
        RSI thresholds for entry and exit.
    stop_loss_pct : float
        Stop-loss fraction (e.g. 0.05 = 5%).
    stake_fraction : float
        Fraction of cash invested per trade.
    starting_capital : float
        Initial broker cash.
    commission : float
        Per-trade commission fraction.
    printlog : bool
        If True, prints every trade signal to stdout.
    save_plot : bool
        If True, saves equity curve PNG to plot_path.
    plot_path : str
        File path for the saved equity curve image.
    quiet : bool
        If True, suppresses the printed results table.

    Returns
    -------
    dict with keys:
        start_value, end_value, pct_return, max_drawdown,
        sharpe_ratio, total_trades, win_rate, profit_factor,
        avg_trade_pnl, start_date, end_date
    """

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    if df is None:
        df = pd.read_csv(DATA_FILE, index_col="Date", parse_dates=True)

    data_feed = bt.feeds.PandasData(dataname=df)

    # ------------------------------------------------------------------
    # Cerebro setup
    # ------------------------------------------------------------------
    cerebro = bt.Cerebro(stdstats=False)   # Disable default observers
    cerebro.adddata(data_feed)

    # Equity log list — populated by _TrackingStrategy on every bar
    equity_log = []

    cerebro.addstrategy(
        _TrackingStrategy,
        fast_sma=fast_sma,
        slow_sma=slow_sma,
        trend_sma=trend_sma,
        rsi_period=rsi_period,
        rsi_entry_max=rsi_entry_max,
        rsi_exit_min=rsi_exit_min,
        stop_loss_pct=stop_loss_pct,
        stake_fraction=stake_fraction,
        printlog=printlog,
        equity_log=equity_log,
    )

    cerebro.broker.setcash(starting_capital)
    cerebro.broker.setcommission(commission=commission)

    # -- Analyzers --
    cerebro.addanalyzer(bt.analyzers.SharpeRatio,
                        _name="sharpe",
                        riskfreerate=0.02,        # 2% annual risk-free rate
                        annualize=True,
                        timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown,    _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns,     _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    start_value = cerebro.broker.getvalue()
    results     = cerebro.run()
    end_value   = cerebro.broker.getvalue()
    strat       = results[0]

    # ------------------------------------------------------------------
    # Extract analyzer results
    # ------------------------------------------------------------------
    sharpe_raw = strat.analyzers.sharpe.get_analysis().get("sharperatio", None)
    sharpe     = round(sharpe_raw, 3) if sharpe_raw is not None else None

    dd_analysis  = strat.analyzers.drawdown.get_analysis()
    max_drawdown = round(dd_analysis.get("max", {}).get("drawdown", 0.0), 2)

    trade_analysis = strat.analyzers.trades.get_analysis()
    total_trades   = int(trade_analysis.get("total", {}).get("closed", 0))

    # Win / Loss breakdown
    won_count  = int(trade_analysis.get("won",  {}).get("total", 0))
    lost_count = int(trade_analysis.get("lost", {}).get("total", 0))
    win_rate   = round((won_count / total_trades * 100), 1) if total_trades > 0 else 0.0

    won_total  = trade_analysis.get("won",  {}).get("pnl", {}).get("total", 0.0)
    lost_total = abs(trade_analysis.get("lost", {}).get("pnl", {}).get("total", 0.0))
    profit_factor = round(won_total / lost_total, 2) if lost_total > 0 else float("inf")

    avg_trade_pnl = round((end_value - start_value) / total_trades, 2) if total_trades > 0 else 0.0
    pct_return    = round((end_value - start_value) / start_value * 100, 2)

    start_date = str(df.index[0].date())
    end_date   = str(df.index[-1].date())

    # ------------------------------------------------------------------
    # Build and record equity curve
    # ------------------------------------------------------------------
    # equity_log is a list of (date, value) tuples from _TrackingStrategy
    if save_plot and equity_log:
        _save_equity_curve(
            equity_log=equity_log,
            starting_capital=starting_capital,
            plot_path=plot_path,
            pct_return=pct_return,
            max_drawdown=max_drawdown,
            sharpe=sharpe,
        )

    # ------------------------------------------------------------------
    # Build metrics dict
    # ------------------------------------------------------------------
    metrics = {
        "start_value":   start_value,
        "end_value":     end_value,
        "pct_return":    pct_return,
        "max_drawdown":  max_drawdown,
        "sharpe_ratio":  sharpe,
        "total_trades":  total_trades,
        "won_trades":    won_count,
        "lost_trades":   lost_count,
        "win_rate":      win_rate,
        "profit_factor": profit_factor,
        "avg_trade_pnl": avg_trade_pnl,
        "start_date":    start_date,
        "end_date":      end_date,
    }

    # ------------------------------------------------------------------
    # Print results table
    # ------------------------------------------------------------------
    if not quiet:
        _print_results(metrics)

    return metrics


# =============================================================================
# Equity curve plotter
# =============================================================================
def _save_equity_curve(
    equity_log,
    starting_capital, plot_path,
    pct_return, max_drawdown, sharpe,
):
    os.makedirs("results", exist_ok=True)

    dates          = [d for d, _ in equity_log]
    equity_values  = [v for _, v in equity_log]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # Equity line — colour by positive/negative vs starting capital
    color = "#00e676" if equity_values[-1] >= starting_capital else "#ff5252"
    ax.plot(dates, equity_values, color=color, linewidth=1.8, label="Portfolio Value")

    # Starting capital reference line
    ax.axhline(y=starting_capital, color="#ffffff", linewidth=0.8,
               linestyle="--", alpha=0.4, label=f"Starting Capital (${starting_capital:,.0f})")

    # Fill under curve
    ax.fill_between(dates, equity_values, starting_capital,
                    where=[v >= starting_capital for v in equity_values],
                    alpha=0.15, color="#00e676", interpolate=True)
    ax.fill_between(dates, equity_values, starting_capital,
                    where=[v < starting_capital for v in equity_values],
                    alpha=0.15, color="#ff5252", interpolate=True)

    # Axis formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    ax.tick_params(colors="#aaaaaa", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    ax.grid(axis="y", color="#222222", linewidth=0.6, linestyle="--")
    ax.grid(axis="x", color="#222222", linewidth=0.4, linestyle=":")

    # Labels & title
    ax.set_xlabel("Date", color="#aaaaaa", fontsize=10)
    ax.set_ylabel("Portfolio Value (USD)", color="#aaaaaa", fontsize=10)

    sharpe_str = f"{sharpe:.2f}" if sharpe is not None else "N/A"
    ax.set_title(
        f"{SYMBOL} — SMA/RSI Strategy  |  "
        f"Return: {pct_return:+.1f}%  |  "
        f"Max DD: {max_drawdown:.1f}%  |  "
        f"Sharpe: {sharpe_str}",
        color="#ffffff", fontsize=12, pad=14,
    )

    legend = ax.legend(facecolor="#1a1d27", edgecolor="#333333",
                       labelcolor="#cccccc", fontsize=9)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] Equity curve saved -> {plot_path}")


# =============================================================================
# Pretty results printer
# =============================================================================
def _print_results(m: dict):
    line = "=" * 55
    print(f"\n{line}")
    print(f"  BACKTEST RESULTS — {SYMBOL}")
    print(line)
    print(f"  Period              : {m['start_date']} -> {m['end_date']}")
    print(f"  Starting Capital    : ${m['start_value']:>12,.2f}")
    print(f"  Ending Value        : ${m['end_value']:>12,.2f}")
    print(f"  {'Pct Return':<20}: {m['pct_return']:>+8.2f} %")
    print(f"  {'Max Drawdown':<20}: {m['max_drawdown']:>8.2f} %")
    sharpe_str = f"{m['sharpe_ratio']:.3f}" if m['sharpe_ratio'] is not None else "N/A"
    print(f"  {'Sharpe Ratio':<20}: {sharpe_str:>8}")
    print(f"  {'Total Trades':<20}: {m['total_trades']:>8}")
    print(f"  {'Won / Lost':<20}: {m['won_trades']:>4} / {m['lost_trades']:<4}")
    print(f"  {'Win Rate':<20}: {m['win_rate']:>8.1f} %")
    print(f"  {'Profit Factor':<20}: {m['profit_factor']:>8.2f}")
    print(f"  {'Avg Trade P&L':<20}: ${m['avg_trade_pnl']:>10,.2f}")
    print(f"{line}\n")


# =============================================================================
# Script entry point
# =============================================================================
if __name__ == "__main__":
    print("[INFO] Running full-period backtest ...")
    metrics = run_backtest(printlog=False, save_plot=True)
