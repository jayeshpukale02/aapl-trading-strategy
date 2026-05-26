# backtest/__init__.py
from backtest.run_backtest import run_backtest
from backtest.walk_forward import run_walk_forward

__all__ = ["run_backtest", "run_walk_forward"]
