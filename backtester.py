"""
Algorithmic Trading Backtester — entry point.

Run this file to start the app. The actual implementation is split into:
- app.py: BacktesterApp UI and orchestration
- data_handler.py: fetch_data (yfinance + column normalization)
- strategy.py: BaseStrategy, MACrossoverStrategy, MeanReversionStrategy
- engine.py: BacktestEngine (returns, equity, metrics, vectorized trades)
- logger.py: _agent_log for debugging
"""

from app import BacktesterApp


def main():
    app = BacktesterApp()
    app.mainloop()


if __name__ == "__main__":
    main()
