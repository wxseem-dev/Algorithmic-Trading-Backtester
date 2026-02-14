"""
Tkinter UI for the algorithmic trading backtester.
Orchestrates: data_handler -> strategy -> engine -> charts/metrics.
"""
import datetime as dt
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from data_handler import fetch_data
from engine import BacktestEngine
from logger import _agent_log
from strategy import MACrossoverStrategy, MeanReversionStrategy


# Strategy key used in UI vs class mapping
STRATEGY_CHOICES = [
    "MA Crossover",
    "Mean Reversion",
]


def _get_strategy_class(name: str):
    if name == "MA Crossover":
        return MACrossoverStrategy
    if name == "Mean Reversion":
        return MeanReversionStrategy
    return MACrossoverStrategy  # default


class BacktesterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simple Algorithmic Trading Backtester")
        self.geometry("1100x700")
        self._build_ui()

        self.fig = Figure(figsize=(8, 5), dpi=75)
        self.fig.tight_layout(pad=2.0)
        self.price_ax = self.fig.add_subplot(211)
        self.equity_ax = self.fig.add_subplot(212, sharex=self.price_ax)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_ui(self):
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Label(control_frame, text="Ticker:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.ticker_var = tk.StringVar(value="AAPL")
        ticker_choices = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "SPY", "QQQ", "KO", "PEP"
        ]
        self.ticker_box = ttk.Combobox(
            control_frame, textvariable=self.ticker_var, values=ticker_choices, width=10
        )
        self.ticker_box.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(control_frame, text="Start (YYYY-MM-DD):").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        self.start_var = tk.StringVar(
            value=(dt.date.today() - dt.timedelta(days=365 * 3)).strftime("%Y-%m-%d")
        )
        ttk.Entry(control_frame, textvariable=self.start_var, width=12).grid(row=0, column=3, padx=5, pady=2)

        ttk.Label(control_frame, text="End (YYYY-MM-DD):").grid(row=0, column=4, sticky="w", padx=5, pady=2)
        self.end_var = tk.StringVar(value=dt.date.today().strftime("%Y-%m-%d"))
        ttk.Entry(control_frame, textvariable=self.end_var, width=12).grid(row=0, column=5, padx=5, pady=2)

        # Strategy dropdown
        ttk.Label(control_frame, text="Strategy:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.strategy_var = tk.StringVar(value="MA Crossover")
        self.strategy_box = ttk.Combobox(
            control_frame,
            textvariable=self.strategy_var,
            values=STRATEGY_CHOICES,
            state="readonly",
            width=14,
        )
        self.strategy_box.grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(control_frame, text="Short MA (days):").grid(row=1, column=2, sticky="w", padx=5, pady=2)
        self.short_ma_var = tk.StringVar(value="50")
        ttk.Entry(control_frame, textvariable=self.short_ma_var, width=8).grid(row=1, column=3, padx=5, pady=2)

        ttk.Label(control_frame, text="Long MA (days):").grid(row=1, column=4, sticky="w", padx=5, pady=2)
        self.long_ma_var = tk.StringVar(value="200")
        ttk.Entry(control_frame, textvariable=self.long_ma_var, width=8).grid(row=1, column=5, padx=5, pady=2)

        run_button = ttk.Button(control_frame, text="Run Backtest", command=self.run_backtest)
        run_button.grid(row=1, column=6, padx=10, pady=2, sticky="e")

        self.metrics_text = tk.Text(self, height=8, width=80, state="disabled")
        self.metrics_text.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        self.chart_frame = ttk.Frame(self)
        self.chart_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

    def run_backtest(self):
        try:
            ticker = self.ticker_var.get().strip().upper()
            start = dt.datetime.strptime(self.start_var.get().strip(), "%Y-%m-%d")
            end = dt.datetime.strptime(self.end_var.get().strip(), "%Y-%m-%d")
            short_ma = int(self.short_ma_var.get().strip())
            long_ma = int(self.long_ma_var.get().strip())
        except Exception as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")
            return

        strategy_name = self.strategy_var.get().strip()
        if strategy_name not in STRATEGY_CHOICES:
            strategy_name = "MA Crossover"

        _agent_log(
            {
                "runId": "pre-fix",
                "hypothesisId": "A",
                "location": "app.py:run_backtest:inputs",
                "message": "Parsed user inputs",
                "data": {
                    "ticker": ticker,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "short_ma": short_ma,
                    "long_ma": long_ma,
                    "strategy": strategy_name,
                },
            }
        )

        if short_ma <= 0 or long_ma <= 0:
            messagebox.showerror("Input Error", "MA windows must be positive integers.")
            return
        if short_ma >= long_ma and strategy_name == "MA Crossover":
            messagebox.showerror("Input Error", "Short MA must be less than Long MA.")
            return

        try:
            data = fetch_data(ticker, start, end)
        except ValueError as e:
            messagebox.showerror("Data Error", str(e))
            return
        except Exception as e:
            _agent_log(
                {
                    "runId": "pre-fix",
                    "hypothesisId": "B",
                    "location": "app.py:run_backtest:download_error",
                    "message": "Download error from yfinance",
                    "data": {"ticker": ticker, "error": str(e)},
                }
            )
            messagebox.showerror("Download Error", f"Failed to download data: {e}")
            return

        if data.empty:
            _agent_log(
                {
                    "runId": "pre-fix",
                    "hypothesisId": "B",
                    "location": "app.py:run_backtest:no_data",
                    "message": "Downloaded data is empty",
                    "data": {"ticker": ticker},
                }
            )
            messagebox.showwarning("No Data", "No data returned for given inputs.")
            return

        _agent_log(
            {
                "runId": "pre-fix",
                "hypothesisId": "B",
                "location": "app.py:run_backtest:download_success",
                "message": "Downloaded data summary",
                "data": {
                    "ticker": ticker,
                    "rows": int(getattr(data, "shape", (0, 0))[0]),
                    "cols": int(getattr(data, "shape", (0, 0))[1]),
                    "columns": list(map(str, getattr(data, "columns", []))),
                },
            }
        )

        # Instantiate chosen strategy
        strategy_cls = _get_strategy_class(strategy_name)
        if strategy_cls == MACrossoverStrategy:
            strategy = MACrossoverStrategy(short_ma=short_ma, long_ma=long_ma)
        else:
            strategy = MeanReversionStrategy()

        df_with_signals = strategy.generate_signals(data)

        # Run engine
        engine = BacktestEngine()
        results = engine.run(df_with_signals)

        # Update UI
        short_ma_plot = short_ma if "Short_MA" in results["df"].columns else None
        long_ma_plot = long_ma if "Long_MA" in results["df"].columns else None
        self._update_charts(results["df"], ticker, short_ma_plot, long_ma_plot)
        self._update_metrics(results, ticker, strategy_name, short_ma, long_ma)

    def _update_charts(self, df: pd.DataFrame, ticker: str, short_ma: int | None, long_ma: int | None):
        self.price_ax.clear()
        self.equity_ax.clear()

        self.price_ax.plot(df.index, df["Close"], label="Close", color="black", linewidth=1)
        if short_ma is not None and "Short_MA" in df.columns:
            self.price_ax.plot(
                df.index, df["Short_MA"], label=f"Short MA ({short_ma})", color="blue", linewidth=1
            )
        if long_ma is not None and "Long_MA" in df.columns:
            self.price_ax.plot(
                df.index, df["Long_MA"], label=f"Long MA ({long_ma})", color="red", linewidth=1
            )

        buys = df[df["Position_Change"] == 1]
        sells = df[df["Position_Change"] == -1]
        self.price_ax.scatter(buys.index, buys["Close"], marker="^", color="green", label="BUY", s=60)
        self.price_ax.scatter(sells.index, sells["Close"], marker="v", color="red", label="SELL", s=60)

        self.price_ax.set_title(f"{ticker} Price and Moving Averages")
        self.price_ax.set_ylabel("Price")
        self.price_ax.legend(loc="upper left")
        self.price_ax.grid(True, linestyle="--", alpha=0.3)

        self.equity_ax.plot(df.index, df["BuyHold_Equity"], label="Buy & Hold", color="gray", linestyle="--")
        self.equity_ax.plot(df.index, df["Strategy_Equity"], label="Strategy", color="purple")
        self.equity_ax.set_title("Equity Curve")
        self.equity_ax.set_ylabel("Equity (normalized)")
        self.equity_ax.legend(loc="upper left")
        self.equity_ax.grid(True, linestyle="--", alpha=0.3)

        self.fig.autofmt_xdate()
        self.canvas.draw()

    def _update_metrics(self, results: dict, ticker: str, strategy_name: str, short_ma: int, long_ma: int):
        total_return_strategy = results["total_return_strategy"]
        total_return_buyhold = results["total_return_buyhold"]
        max_drawdown = results["max_drawdown"]
        num_trades = results["num_trades"]
        win_rate = results["win_rate"]
        avg_trade_return = results["avg_trade_return"]

        def pct(x):
            return f"{x*100:.2f}%" if np.isfinite(x) else "N/A"

        ma_line = f"Short MA: {short_ma} days, Long MA: {long_ma} days" if strategy_name == "MA Crossover" else "Strategy: Mean Reversion (skeleton)"
        lines = [
            f"Ticker: {ticker}",
            f"Strategy: {strategy_name}",
            ma_line,
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
