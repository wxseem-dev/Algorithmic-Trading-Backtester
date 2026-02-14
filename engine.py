"""
Backtest engine: strategy returns, equity curves, and metrics.
Uses vectorized Pandas operations (no loop over entry indices).
"""
import numpy as np
import pandas as pd

from logger import _agent_log


class BacktestEngine:
    """
    Runs backtest math on strategy output: Strategy_Return, Equity, drawdown, trade-level returns.
    """

    def run(self, df: pd.DataFrame) -> dict:
        """
        Compute strategy returns, equity curves, and metrics from a DataFrame
        that already has Close, Signal, and Position_Change (from a strategy).

        :param df: DataFrame with Close, Signal, Position_Change; optional Short_MA, Long_MA.
        :returns: Dict with keys: df, total_return_strategy, total_return_buyhold,
                  max_drawdown, num_trades, win_rate, avg_trade_return, trade_results.
        """
        df = df.copy()

        _agent_log(
            {
                "runId": "pre-fix",
                "hypothesisId": "C",
                "location": "engine.py:BacktestEngine.run:start",
                "message": "Starting backtest engine",
                "data": {
                    "rows": int(df.shape[0]),
                    "has_signal": "Signal" in df.columns,
                },
            }
        )

        # Daily percent change
        df["Return"] = df["Close"].pct_change().fillna(0)

        # Strategy return = yesterday's holding decision * today's price change
        df["Strategy_Return"] = df["Signal"].shift(1).fillna(0) * df["Return"]

        # Equity curves
        df["BuyHold_Equity"] = (1 + df["Return"]).cumprod()
        df["Strategy_Equity"] = (1 + df["Strategy_Return"]).cumprod()

        # Trade-level returns: vectorized using entry/exit and .shift()-style logic
        entry_mask = df["Position_Change"] == 1
        exit_mask = df["Position_Change"] == -1

        # At each entry date, store Close; forward-fill so each row has "last entry price"
        entry_price_series = df["Close"].where(entry_mask).ffill()
        # At exit rows, return = exit_price / entry_price - 1
        trade_return_at_exit = (df["Close"] / entry_price_series - 1).where(exit_mask)
        trade_results = list(trade_return_at_exit.dropna().values)

        # If still in position at last bar, count final trade (exit at last day)
        if len(df) > 0 and df["Signal"].iloc[-1] == 1 and pd.notna(entry_price_series.iloc[-1]):
            last_entry_price = float(entry_price_series.iloc[-1])
            if last_entry_price > 0:
                trade_results.append(float(df["Close"].iloc[-1]) / last_entry_price - 1)

        trade_results = np.array(trade_results)

        # Metrics
        total_return_strategy = float(df["Strategy_Equity"].iloc[-1] - 1)
        total_return_buyhold = float(df["BuyHold_Equity"].iloc[-1] - 1)

        running_max = df["Strategy_Equity"].cummax()
        drawdown = df["Strategy_Equity"] / running_max - 1
        max_drawdown = float(drawdown.min())

        num_trades = len(trade_results)
        win_rate = float((trade_results > 0).mean()) if num_trades > 0 else np.nan
        avg_trade_return = float(trade_results.mean()) if num_trades > 0 else np.nan

        _agent_log(
            {
                "runId": "pre-fix",
                "hypothesisId": "C",
                "location": "engine.py:BacktestEngine.run:metrics",
                "message": "Computed backtest metrics",
                "data": {
                    "total_return_strategy": total_return_strategy,
                    "total_return_buyhold": total_return_buyhold,
                    "max_drawdown": max_drawdown,
                    "num_trades": int(num_trades),
                    "win_rate": win_rate if np.isfinite(win_rate) else None,
                    "avg_trade_return": avg_trade_return if np.isfinite(avg_trade_return) else None,
                },
            }
        )

        return {
            "df": df,
            "total_return_strategy": total_return_strategy,
            "total_return_buyhold": total_return_buyhold,
            "max_drawdown": max_drawdown,
            "num_trades": num_trades,
            "win_rate": win_rate,
            "avg_trade_return": avg_trade_return,
            "trade_results": trade_results,
        }
