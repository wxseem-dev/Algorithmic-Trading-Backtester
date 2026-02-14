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

        #_agent_log(
            #{
                #"runId": "pre-fix",
                #"hypothesisId": "C",
                #"location": "engine.py:BacktestEngine.run:start",
                #"message": "Starting backtest engine",
                #"data": {
                    #"rows": int(df.shape[0]),
                    #"has_signal": "Signal" in df.columns,
                #},
            #}
        #)

        # Daily percent change
        #df["Return"] = df["Close"].pct_change().fillna(0)

        # Raw strategy return (before costs)
        #df["Strategy_Return_Gross"] = df["Signal"].shift(1).fillna(0) * df["Return"]

        # Transaction costs: paid on both entry (+1) and exit (-1); cost = |Position_Change| * (commission + slippage)
        #cost_per_turnover = commission_pct + slippage_pct
        #df["Transaction_Costs"] = df["Position_Change"].abs() * cost_per_turnover

        # Net strategy return
        #df["Strategy_Return"] = df["Strategy_Return_Gross"] - df["Transaction_Costs"]

        # Equity curves
        #df["BuyHold_Equity"] = (1 + df["Return"]).cumprod()
        #df["Strategy_Equity"] = (1 + df["Strategy_Return"]).cumprod()

        #df["Strategy_Equity_Gross"] = (1 + df["Strategy_Return_Gross"]).cumprod()

        # Trade-level returns: vectorized using entry/exit and .shift()-style logic
        #entry_mask = df["Position_Change"] == 1
        #exit_mask = df["Position_Change"] == -1

        # At each entry date, store Close; forward-fill so each row has "last entry price"
        #entry_price_series = df["Close"].where(entry_mask).ffill()
        # At exit rows, return = exit_price / entry_price - 1
        #trade_return_at_exit = (df["Close"] / entry_price_series - 1).where(exit_mask)
        #trade_results = list(trade_return_at_exit.dropna().values)

        # If still in position at last bar, count final trade (exit at last day)
        #if len(df) > 0 and df["Signal"].iloc[-1] == 1 and pd.notna(entry_price_series.iloc[-1]):
            #last_entry_price = float(entry_price_series.iloc[-1])
            #if last_entry_price > 0:
                #trade_results.append(float(df["Close"].iloc[-1]) / last_entry_price - 1)

        #trade_results = np.array(trade_results)

        # Metrics
        #total_return_strategy = float(df["Strategy_Equity"].iloc[-1] - 1)
        #total_return_buyhold = float(df["BuyHold_Equity"].iloc[-1] - 1)

        #running_max = df["Strategy_Equity"].cummax()
        #drawdown = df["Strategy_Equity"] / running_max - 1
        #max_drawdown = float(drawdown.min())

        #num_trades = len(trade_results)
        #win_rate = float((trade_results > 0).mean()) if num_trades > 0 else np.nan
        #avg_trade_return = float(trade_results.mean()) if num_trades > 0 else np.nan
        #total_costs = float(df["Transaction_Costs"].sum())

        # Annualised risk metrics (252 trading days)
        #risk_free_rate = 0.04
        #daily_rf = risk_free_rate / 252
        #excess_returns = df["Strategy_Return"] - daily_rf
        #std_dev = df["Strategy_Return"].std()

        #sharpe_ratio = 0.0
        #if std_dev > 0 and np.isfinite(std_dev):
            #sharpe_ratio = float((excess_returns.mean() / std_dev) * np.sqrt(252))

        #negative_returns = df.loc[df["Strategy_Return"] < 0, "Strategy_Return"]
        #downside_std = negative_returns.std()
        #sortino_ratio = 0.0
        #if downside_std > 0 and np.isfinite(downside_std):
            #sortino_ratio = float((excess_returns.mean() / downside_std) * np.sqrt(252))

        # Trade statistics for profit factor
        #wins = trade_results[trade_results > 0]
        #losses = trade_results[trade_results <= 0]
        #avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        #avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
        #gross_profit = float(wins.sum())
        #gross_loss = abs(float(losses.sum()))
        #profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else np.inf

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
