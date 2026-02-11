"""
This file is a tiny "trading game" app.

You pick:
- a stock ticker (like AAPL)
- a start date and end date
- two moving averages (a short one and a long one)

Then the app:
1) downloads the stock prices
2) pretends to "buy" when the short average is above the long average
3) pretends to "sell" when the short average is below the long average
4) draws pictures (charts) and prints a simple report
"""

# Tkinter makes the window and buttons.
import tkinter as tk
from tkinter import ttk, messagebox

# datetime understands dates like "2026-02-03".
import datetime as dt

# json/time are used only for our tiny debug log helper.
import json
import time

# numpy/pandas help us do math with tables of numbers.
import numpy as np
import pandas as pd

# matplotlib draws charts.
import matplotlib

# Tell matplotlib to draw inside a Tkinter window.
matplotlib.use("TkAgg")

# These connect matplotlib drawings into Tkinter.
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# yfinance downloads stock data from Yahoo Finance.
import yfinance as yf


# region agent log
# This is where we save debug notes (one JSON line per note).
DEBUG_LOG_PATH = r"c:\Users\Waseem\Downloads\backtester\.cursor\debug.log"


def _agent_log(payload: dict) -> None:
    """
    Tiny debug logger.

    It writes one JSON object per line to DEBUG_LOG_PATH.
    This helps us see what happened if the app has a problem.
    """
    base = {
        "sessionId": "debug-session",
        "timestamp": int(time.time() * 1000),
    }
    try:
        # Open the file, add one line, close it.
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({**base, **payload}) + "\n")
    except Exception:
        # If logging fails, we do NOT want the whole app to crash.
        pass


# endregion


class BacktesterApp(tk.Tk):
    def __init__(self):
        # Build the main window.
        super().__init__()
        self.title("Simple Algorithmic Trading Backtester")
        self.geometry("1100x700")

        # Build the buttons, inputs, and text areas.
        self._build_ui()

        # Make a drawing area for charts.
        # We make 2 charts stacked on top of each other:
        # - top: price + moving averages + buy/sell dots
        # - bottom: "equity" (how $1 would grow/shrink)
        self.fig = Figure(figsize=(8, 5), dpi=75)
        self.fig.tight_layout(pad=2.0)
        self.price_ax = self.fig.add_subplot(211)
        self.equity_ax = self.fig.add_subplot(212, sharex=self.price_ax)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_ui(self):
        """Create the widgets you can click/type in."""
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Pick which stock/ETF you want (ticker symbol).
        ttk.Label(control_frame, text="Ticker:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.ticker_var = tk.StringVar(value="AAPL")
        ticker_choices = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "META",
            "NVDA",
            "SPY",
            "QQQ",
        ]
        self.ticker_box = ttk.Combobox(control_frame, textvariable=self.ticker_var, values=ticker_choices, width=10)
        self.ticker_box.grid(row=0, column=1, padx=5, pady=2)

        # Pick the time range to download.
        ttk.Label(control_frame, text="Start (YYYY-MM-DD):").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        # Default: about 3 years ago.
        self.start_var = tk.StringVar(value=(dt.date.today() - dt.timedelta(days=365 * 3)).strftime("%Y-%m-%d"))
        ttk.Entry(control_frame, textvariable=self.start_var, width=12).grid(row=0, column=3, padx=5, pady=2)

        ttk.Label(control_frame, text="End (YYYY-MM-DD):").grid(row=0, column=4, sticky="w", padx=5, pady=2)
        # Default: today.
        self.end_var = tk.StringVar(value=dt.date.today().strftime("%Y-%m-%d"))
        ttk.Entry(control_frame, textvariable=self.end_var, width=12).grid(row=0, column=5, padx=5, pady=2)

        # Pick the moving average sizes (how many days to average).
        ttk.Label(control_frame, text="Short MA (days):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.short_ma_var = tk.StringVar(value="50")
        ttk.Entry(control_frame, textvariable=self.short_ma_var, width=8).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(control_frame, text="Long MA (days):").grid(row=1, column=2, sticky="w", padx=5, pady=2)
        self.long_ma_var = tk.StringVar(value="200")
        ttk.Entry(control_frame, textvariable=self.long_ma_var, width=8).grid(row=1, column=3, padx=5, pady=2)

        # Button: do the backtest now.
        run_button = ttk.Button(control_frame, text="Run Backtest", command=self.run_backtest)
        run_button.grid(row=1, column=5, padx=10, pady=2, sticky="e")

        # Text box: show results (like a report card).
        self.metrics_text = tk.Text(self, height=8, width=80, state="disabled")
        self.metrics_text.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        # Big blank space where charts will be drawn.
        self.chart_frame = ttk.Frame(self)
        self.chart_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

    def run_backtest(self):
        """
        Runs when you click the "Run Backtest" button.

        Big steps:
        - read what you typed/selected
        - download prices
        - do the pretend trading math
        - draw charts + show a report
        """
        try:
            # Read ticker (like "AAPL"), remove spaces, make uppercase.
            ticker = self.ticker_var.get().strip().upper()
            # Turn the start/end text into real datetime values.
            start = dt.datetime.strptime(self.start_var.get().strip(), "%Y-%m-%d")
            end = dt.datetime.strptime(self.end_var.get().strip(), "%Y-%m-%d")
            # Turn the MA text boxes into integers.
            short_ma = int(self.short_ma_var.get().strip())
            long_ma = int(self.long_ma_var.get().strip())
        except Exception as e:
            # If something is wrong, show a popup and stop.
            messagebox.showerror("Input Error", f"Invalid input: {e}")
            return

        # region agent log
        _agent_log(
            {
                "runId": "pre-fix",
                "hypothesisId": "A",
                "location": "backtester.py:run_backtest:inputs",
                "message": "Parsed user inputs",
                "data": {
                    "ticker": ticker,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "short_ma": short_ma,
                    "long_ma": long_ma,
                },
            }
        )
        # endregion

        # MA sizes must be positive numbers.
        if short_ma <= 0 or long_ma <= 0:
            messagebox.showerror("Input Error", "MA windows must be positive integers.")
            return

        # The "short" average must be smaller than the "long" average.
        if short_ma >= long_ma:
            messagebox.showerror("Input Error", "Short MA must be less than Long MA.")
            return

        try:
            # Download stock prices from the internet.
            data = yf.download(ticker, start=start, end=end)
        except Exception as e:
            # region agent log
            _agent_log(
                {
                    "runId": "pre-fix",
                    "hypothesisId": "B",
                    "location": "backtester.py:run_backtest:download_error",
                    "message": "Download error from yfinance",
                    "data": {"ticker": ticker, "error": str(e)},
                }
            )
            # endregion
            messagebox.showerror("Download Error", f"Failed to download data: {e}")
            return

        # Sometimes yfinance returns columns like ('Close','AAPL') instead of just 'Close'.
        # That can confuse the rest of our code, so we flatten it here.
        if isinstance(getattr(data, "columns", None), pd.MultiIndex):
            try:
                # If the ticker exists in the last level, select just that ticker's columns.
                if ticker in data.columns.get_level_values(-1):
                    data = data.xs(ticker, axis=1, level=-1)
                else:
                    # Otherwise, drop the last level to make columns simple.
                    data = data.droplevel(-1, axis=1)
            except Exception:
                # region agent log
                _agent_log(
                    {
                        "runId": "pre-fix",
                        "hypothesisId": "D",
                        "location": "backtester.py:run_backtest:normalize_columns_error",
                        "message": "Failed to normalize MultiIndex columns",
                        "data": {
                            "ticker": ticker,
                            "columns_repr": repr(data.columns),
                        },
                    }
                )
                # endregion
                messagebox.showerror("Data Error", "Unexpected data format returned from yfinance.")
                return

        if data.empty:
            # region agent log
            _agent_log(
                {
                    "runId": "pre-fix",
                    "hypothesisId": "B",
                    "location": "backtester.py:run_backtest:no_data",
                    "message": "Downloaded data is empty",
                    "data": {"ticker": ticker},
                }
            )
            # endregion
            messagebox.showwarning("No Data", "No data returned for given inputs.")
            return

        # region agent log
        _agent_log(
            {
                "runId": "pre-fix",
                "hypothesisId": "B",
                "location": "backtester.py:run_backtest:download_success",
                "message": "Downloaded data summary",
                "data": {
                    "ticker": ticker,
                    "rows": int(getattr(data, "shape", (0, 0))[0]),
                    "cols": int(getattr(data, "shape", (0, 0))[1]),
                    "columns": list(map(str, getattr(data, "columns", []))),
                },
            }
        )
        # endregion

        # Do the strategy math (this returns a dictionary of results).
        results = self._backtest_ma_crossover(data, short_ma, long_ma)

        # Draw the pictures (charts).
        self._update_charts(data, results, ticker, short_ma, long_ma)
        # Show the text results (metrics).
        self._update_metrics(results, ticker, short_ma, long_ma)

    @staticmethod
    def _backtest_ma_crossover(data: pd.DataFrame, short_ma: int, long_ma: int):
        """
        Pretend trading strategy (moving average crossover):

        - If the short average line is ABOVE the long average line: we pretend we are "holding" the stock.
        - If the short average line is BELOW the long average line: we pretend we are "not holding" it.

        Then we calculate:
        - daily returns
        - strategy returns (only earn/lose money on days we were holding)
        - equity curves (how $1 changes over time)
        """
        # Copy the data so we can add new columns safely.
        df = data.copy()

        # We want one main price column called "Close".
        # If "Adj Close" exists, it is often better because it accounts for stock splits/dividends.
        df["Close"] = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]

        # region agent log
        _agent_log(
            {
                "runId": "pre-fix",
                "hypothesisId": "C",
                "location": "backtester.py:_backtest_ma_crossover:start",
                "message": "Starting MA crossover backtest",
                "data": {
                    "rows": int(df.shape[0]),
                    "short_ma": short_ma,
                    "long_ma": long_ma,
                    "has_adj_close": "Adj Close" in df.columns,
                },
            }
        )
        # endregion

        # Moving averages = smooth lines made by averaging the last N days.
        df["Short_MA"] = df["Close"].rolling(window=short_ma).mean()
        df["Long_MA"] = df["Close"].rolling(window=long_ma).mean()

        # Signal: 1 means "we hold the stock", 0 means "we don't hold it".
        df["Signal"] = 0
        # If the short line is above the long line, set Signal to 1.
        df.loc[df["Short_MA"] > df["Long_MA"], "Signal"] = 1

        # Position_Change tells us when we switched:
        # +1 => buy (0 -> 1)
        # -1 => sell (1 -> 0)
        df["Position_Change"] = df["Signal"].diff().fillna(0)

        # Daily percent change (today compared to yesterday).
        df["Return"] = df["Close"].pct_change().fillna(0)

        # Strategy return = yesterday's holding decision * today's price change.
        # We use shift(1) so we "decide" first, then experience the return next day.
        df["Strategy_Return"] = df["Signal"].shift(1).fillna(0) * df["Return"]

        # Equity curves: start at 1 and multiply forward each day.
        df["BuyHold_Equity"] = (1 + df["Return"]).cumprod()
        df["Strategy_Equity"] = (1 + df["Strategy_Return"]).cumprod()

        # Find buy and sell moments.
        trades = df[df["Position_Change"] != 0].copy()
        # Entries are buy signals (+1).
        entry_indices = trades[trades["Position_Change"] == 1].index
        # Exits are sell signals (-1).
        exit_indices = trades[trades["Position_Change"] == -1].index

        # We'll store each trade's percent profit/loss here.
        trade_results = []
        for entry_idx in entry_indices:
            # Find the first sell after this buy.
            possible_exits = exit_indices[exit_indices > entry_idx]
            if len(possible_exits) == 0:
                # No sell found: pretend we sell at the last day.
                exit_idx = df.index[-1]
            else:
                # Sell at the first sell signal after entry.
                exit_idx = possible_exits[0]

            # Price at buy time and sell time.
            entry_price = df.at[entry_idx, "Close"]
            exit_price = df.at[exit_idx, "Close"]
            # Profit percent for this trade.
            trade_return = exit_price / entry_price - 1
            trade_results.append(trade_return)

        trade_results = np.array(trade_results) if len(trade_results) > 0 else np.array([])

        # --- Scorecard numbers ---
        total_return_strategy = df["Strategy_Equity"].iloc[-1] - 1
        total_return_buyhold = df["BuyHold_Equity"].iloc[-1] - 1

        # Max drawdown = biggest drop from a previous high.
        running_max = df["Strategy_Equity"].cummax()
        drawdown = df["Strategy_Equity"] / running_max - 1
        max_drawdown = drawdown.min()

        num_trades = len(trade_results)
        if num_trades > 0:
            # Win rate = percent of trades that were positive.
            win_rate = (trade_results > 0).mean()
            # Average profit/loss per trade.
            avg_trade_return = trade_results.mean()
        else:
            # If we never traded, these are not available.
            win_rate = np.nan
            avg_trade_return = np.nan

        # region agent log
        _agent_log(
            {
                "runId": "pre-fix",
                "hypothesisId": "C",
                "location": "backtester.py:_backtest_ma_crossover:metrics",
                "message": "Computed backtest metrics",
                "data": {
                    "total_return_strategy": float(total_return_strategy),
                    "total_return_buyhold": float(total_return_buyhold),
                    "max_drawdown": float(max_drawdown),
                    "num_trades": int(num_trades),
                    "win_rate": float(win_rate) if np.isfinite(win_rate) else None,
                    "avg_trade_return": float(avg_trade_return) if np.isfinite(avg_trade_return) else None,
                },
            }
        )
        # endregion

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

    def _update_charts(self, data: pd.DataFrame, results, ticker: str, short_ma: int, long_ma: int):
        """Draw the charts in the window."""
        df = results["df"]

        # Clear old drawings first (so we don't draw on top of old stuff).
        self.price_ax.clear()
        self.equity_ax.clear()

        # Top chart: price and moving averages.
        self.price_ax.plot(df.index, df["Close"], label="Close", color="black", linewidth=1)
        self.price_ax.plot(df.index, df["Short_MA"], label=f"Short MA ({short_ma})", color="blue", linewidth=1)
        self.price_ax.plot(df.index, df["Long_MA"], label=f"Long MA ({long_ma})", color="red", linewidth=1)

        # Buy and sell dots.
        buys = df[df["Position_Change"] == 1]
        sells = df[df["Position_Change"] == -1]

        self.price_ax.scatter(buys.index, buys["Close"], marker="^", color="green", label="BUY", s=60)
        self.price_ax.scatter(sells.index, sells["Close"], marker="v", color="red", label="SELL", s=60)

        # Labels make the chart easy to read.
        self.price_ax.set_title(f"{ticker} Price and Moving Averages")
        self.price_ax.set_ylabel("Price")
        self.price_ax.legend(loc="upper left")
        self.price_ax.grid(True, linestyle="--", alpha=0.3)

        # Bottom chart: how $1 grows/shrinks.
        self.equity_ax.plot(df.index, df["BuyHold_Equity"], label="Buy & Hold", color="gray", linestyle="--")
        self.equity_ax.plot(df.index, df["Strategy_Equity"], label="Strategy", color="purple")

        self.equity_ax.set_title("Equity Curve")
        self.equity_ax.set_ylabel("Equity (normalized)")
        self.equity_ax.legend(loc="upper left")
        self.equity_ax.grid(True, linestyle="--", alpha=0.3)

        # Make the date labels nicer.
        self.fig.autofmt_xdate()
        # Draw on the screen now.
        self.canvas.draw()

    def _update_metrics(self, results, ticker: str, short_ma: int, long_ma: int):
        """Show the numbers in the text box at the bottom."""
        total_return_strategy = results["total_return_strategy"]
        total_return_buyhold = results["total_return_buyhold"]
        max_drawdown = results["max_drawdown"]
        num_trades = results["num_trades"]
        win_rate = results["win_rate"]
        avg_trade_return = results["avg_trade_return"]

        def pct(x):
            # Turn 0.12 into "12.00%". If not a real number, show "N/A".
            return f"{x*100:.2f}%" if np.isfinite(x) else "N/A"

        lines = [
            f"Ticker: {ticker}",
            f"Short MA: {short_ma} days, Long MA: {long_ma} days",
            "",
            f"Total Return (Strategy): {pct(total_return_strategy)}",
            f"Total Return (Buy & Hold): {pct(total_return_buyhold)}",
            f"Max Drawdown (Strategy): {pct(max_drawdown)}",
            "",
            f"Number of Trades: {num_trades}",
            f"Winning %: {pct(win_rate)}",
            f"Average Trade Return: {pct(avg_trade_return)}",
        ]

        self.metrics_text.config(state="normal")
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, "\n".join(lines))
        self.metrics_text.config(state="disabled")


def main():
    # Create and run the app window.
    app = BacktesterApp()
    app.mainloop()


if __name__ == "__main__":
    main()

