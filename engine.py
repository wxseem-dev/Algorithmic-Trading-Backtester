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

    def run(
        self,
        df: pd.DataFrame,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
    ) -> dict:
        """
        Compute strategy returns, equity curves, and metrics from a DataFrame
        that already has Close, Signal, and Position_Change (from a strategy).
        Transaction costs (commission + slippage) are applied on position changes.

        :param df: DataFrame with Close, Signal, Position_Change; optional Short_MA, Long_MA.
        :param commission_pct: Commission per trade as a decimal (e.g. 0.001 = 10 bps).
        :param slippage_pct: Slippage per trade as a decimal (e.g. 0.0005 = 5 bps).
        :returns: Dict with keys: df, total_return_strategy, total_return_buyhold,
                  max_drawdown, num_trades, win_rate, avg_trade_return, trade_results.
        """
        df = df.copy()

        df["Return"] = df["Close"].pct_change().fillna(0)
        
        df["Strategy_Return_Gross"] = df["Signal"].shift(1).fillna(0) * df["Return"]
        
        cost_per_trade = commission_pct + slippage_pct
        df["Transaction_Costs"] = df["Position_Change"].abs() * cost_per_trade

        df["Strategy_Return"] = df["Strategy_Return_Gross"] - df["Transaction_Costs"]

        df["Strategy_Equity"] = (1 + df["Strategy_Return"]).cumprod()

        df["Strategy_Equity_Gross"] = (1 + df["Strategy_Return_Gross"]).cumprod()

        df["BuyHold_Equity"] = (1 + df["Return"]).cumprod()

        signals = df["Signal"].values
        prices = df["Close"].values
        trades = []
        entry_price = 0.0
        in_trade = False

        for i in range(1, len(signals)):
            if signals[i-1] == 0 and signals[i] == 1:
                entry_price = prices[i]
                in_trade = True

            elif signals[i-1] == 1 and signals[i] == 0:
                if in_trade:
                    trade_ret = (prices[i] / entry_price) - 1
                    trades.append(trade_ret)
                    in_trade = False
        
        if in_trade:
            trade_ret = (prices[-1] / entry_price) - 1
            trades.append(trade_ret)

        trade_results = pd.Series(trades)

        total_return_strategy = float(df["Strategy_Equity"].iloc[-1] - 1)
        total_return_buyhold = float(df["BuyHold_Equity"].iloc[-1] - 1)

        running_max = df["Strategy_Equity"].cummax()
        drawdown = df["Strategy_Equity"] / running_max - 1
        max_drawdown = float(drawdown.min())

        daily_mean = df["Strategy_Return"].mean()
        daily_std = df["Strategy_Return"].std()
        sharpe_ratio = (daily_mean / daily_std) * np.sqrt(252) if daily_std > 0 else 0.0

        downside_returns = df.loc[df["Strategy_Return"] < 0, "Strategy_Return"]
        downside_std = downside_returns.std()
        sortino_ratio = (daily_mean / downside_std) * np.sqrt(252) if downside_std > 0 else 0.0

        num_trades = len(trade_results)
        wins = trade_results[trade_results > 0]
        losses = trade_results[trade_results <= 0]
        
        win_rate = float(len(wins) / num_trades) if num_trades > 0 else 0.0
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

        total_gross_win = wins.sum()
        total_gross_loss = abs(losses.sum())
        profit_factor = float(total_gross_win / total_gross_loss) if total_gross_loss > 0 else np.inf
        
        total_costs = float(df["Transaction_Costs"].sum())

        avg_trade_return = float(trade_results.mean()) if num_trades > 0 else 0.0

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
                    "total_costs": total_costs,
                    "gross_return": float(df["Strategy_Equity_Gross"].iloc[-1] - 1),
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
            "total_costs": total_costs,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
        }
