# =============================================================================
# strategy/my_strategy.py — SMARSIStrategy
#
# Strategy: Hybrid Trend-Following + Momentum Filter
#
# ENTRY (Long only):
#   1. Fast SMA crosses ABOVE Slow SMA  → trend turning bullish
#   2. Close price is ABOVE the 200-day SMA → macro uptrend confirmed
#   3. RSI < rsi_entry_max              → not yet overbought (room to run)
#
# EXIT (any one condition):
#   1. Fast SMA crosses BELOW Slow SMA  → trend turning bearish
#   2. RSI > rsi_exit_min               → overbought, take profit
#   3. Price drops > stop_loss_pct from entry price → hard stop-loss
#
# POSITION SIZING:
#   - Fixed fractional: invest `stake_fraction` of available cash per trade
#   - Never hold more than one position at a time (long-only)
#
# DESIGN NOTES:
#   - All parameters are exposed via `params` so the WFA optimizer can
#     grid-search over them without touching this file.
#   - Trade logging is verbose by default to aid debugging and audit.
# =============================================================================

import backtrader as bt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FAST_SMA, SLOW_SMA, TREND_SMA,
    RSI_PERIOD, RSI_ENTRY_MAX, RSI_EXIT_MIN,
    STOP_LOSS_PCT, STAKE_FRACTION,
)


class SMARSIStrategy(bt.Strategy):
    """
    Dual-SMA crossover strategy filtered by RSI and a 200-day trend SMA.

    Parameters
    ----------
    fast_sma : int
        Period for the fast Simple Moving Average.
    slow_sma : int
        Period for the slow Simple Moving Average.
    trend_sma : int
        Period for the long-term trend filter SMA (usually 200).
    rsi_period : int
        Lookback period for RSI calculation.
    rsi_entry_max : float
        Maximum RSI value allowed on entry (avoid buying overbought).
    rsi_exit_min : float
        RSI level above which we exit the position (overbought exit).
    stop_loss_pct : float
        Fraction below entry price that triggers a hard stop-loss.
        E.g. 0.05 = 5% below entry.
    stake_fraction : float
        Fraction of available broker cash to deploy per trade.
        E.g. 0.95 uses 95% of cash, leaving a small buffer for commissions.
    printlog : bool
        If True, prints a log line for every trade signal. Default True.
    """

    params = (
        ("fast_sma",      FAST_SMA),
        ("slow_sma",      SLOW_SMA),
        ("trend_sma",     TREND_SMA),
        ("rsi_period",    RSI_PERIOD),
        ("rsi_entry_max", RSI_ENTRY_MAX),
        ("rsi_exit_min",  RSI_EXIT_MIN),
        ("stop_loss_pct", STOP_LOSS_PCT),
        ("stake_fraction", STAKE_FRACTION),
        ("printlog",      True),
    )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def __init__(self):
        # Reference to closing price series (shortcut)
        self.dataclose = self.datas[0].close

        # --- Indicators ---
        # Fast and Slow SMA for crossover signal
        self.fast_ma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.p.fast_sma
        )
        self.slow_ma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.p.slow_sma
        )
        # Long-term trend filter: only go long above this SMA
        self.trend_ma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.p.trend_sma
        )
        # RSI momentum oscillator
        self.rsi = bt.indicators.RelativeStrengthIndex(
            self.datas[0], period=self.p.rsi_period
        )
        # Crossover signal: +1 when fast crosses above slow, -1 vice versa
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

        # Internal state: track entry price for stop-loss
        self.entry_price = None

        # Order reference (to avoid double-ordering while an order is pending)
        self.order = None

    # ------------------------------------------------------------------
    # Order notification callback
    # ------------------------------------------------------------------
    def notify_order(self, order):
        """Called by Backtrader whenever an order status changes."""
        if order.status in [order.Submitted, order.Accepted]:
            # Order sent to broker — nothing to do yet
            return

        if order.status == order.Completed:
            if order.isbuy():
                self.entry_price = order.executed.price
                self._log(
                    f"BUY  EXECUTED  | Price: {order.executed.price:.2f} "
                    f"| Size: {order.executed.size:.0f} "
                    f"| Cost: {order.executed.value:.2f} "
                    f"| Comm: {order.executed.comm:.2f}"
                )
            elif order.issell():
                self._log(
                    f"SELL EXECUTED  | Price: {order.executed.price:.2f} "
                    f"| Size: {order.executed.size:.0f} "
                    f"| Cost: {order.executed.value:.2f} "
                    f"| Comm: {order.executed.comm:.2f}"
                )
                self.entry_price = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self._log(f"Order CANCELLED / MARGIN / REJECTED")

        # Reset pending order reference
        self.order = None

    # ------------------------------------------------------------------
    # Trade notification callback
    # ------------------------------------------------------------------
    def notify_trade(self, trade):
        """Called when a round-trip trade (buy→sell) is closed."""
        if not trade.isclosed:
            return
        self._log(
            f"TRADE CLOSED   | Gross P&L: {trade.pnl:.2f} "
            f"| Net P&L: {trade.pnlcomm:.2f}"
        )

    # ------------------------------------------------------------------
    # Core logic — called on every new bar
    # ------------------------------------------------------------------
    def next(self):
        # Skip if an order is already pending (avoid double orders)
        if self.order:
            return

        # ---- STOP-LOSS CHECK (before signal logic) --------------------
        if self.position and self.entry_price is not None:
            stop_price = self.entry_price * (1.0 - self.p.stop_loss_pct)
            if self.dataclose[0] <= stop_price:
                self._log(
                    f"STOP-LOSS HIT  | Close: {self.dataclose[0]:.2f} "
                    f"<= Stop: {stop_price:.2f}"
                )
                self.order = self.close()   # close() liquidates exact position, no shorting
                return

        # ---- ENTRY LOGIC (only when flat / no open position) ----------
        if not self.position:
            entry_signal = (
                self.crossover[0] > 0                        # Fast SMA crossed above Slow SMA
                and self.dataclose[0] > self.trend_ma[0]    # Price above 200-day trend MA
                and self.rsi[0] < self.p.rsi_entry_max      # RSI not overbought on entry
            )
            if entry_signal:
                # Size: invest stake_fraction of available cash
                cash      = self.broker.getcash()
                price     = self.dataclose[0]
                size      = int((cash * self.p.stake_fraction) / price)

                if size > 0:
                    self._log(
                        f"BUY  SIGNAL    | Close: {price:.2f} "
                        f"| FastSMA: {self.fast_ma[0]:.2f} "
                        f"| SlowSMA: {self.slow_ma[0]:.2f} "
                        f"| RSI: {self.rsi[0]:.1f} "
                        f"| Size: {size}"
                    )
                    self.order = self.buy(size=size)

        # ---- EXIT LOGIC (only when in a position) ---------------------
        else:
            exit_crossdown = self.crossover[0] < 0               # Fast crossed below Slow
            exit_overbought = self.rsi[0] > self.p.rsi_exit_min  # RSI overbought

            if exit_crossdown or exit_overbought:
                reason = "SMA CROSSDOWN" if exit_crossdown else "RSI OVERBOUGHT"
                self._log(
                    f"SELL SIGNAL    | Reason: {reason} "
                    f"| Close: {self.dataclose[0]:.2f} "
                    f"| RSI: {self.rsi[0]:.1f}"
                )
                self.order = self.close()  # close() liquidates exact position, no shorting

    # ------------------------------------------------------------------
    # End-of-run summary
    # ------------------------------------------------------------------
    def stop(self):
        """Called once at the very end of the backtest."""
        self._log(
            f"[STRATEGY END] Params: fast={self.p.fast_sma}, "
            f"slow={self.p.slow_sma}, rsi_entry<{self.p.rsi_entry_max}, "
            f"rsi_exit>{self.p.rsi_exit_min}, "
            f"stop_loss={self.p.stop_loss_pct*100:.1f}%"
        )

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------
    def _log(self, message: str):
        """Print a timestamped log line (only when printlog=True)."""
        if self.p.printlog:
            dt = self.datas[0].datetime.date(0)
            print(f"  {dt}  {message}")
