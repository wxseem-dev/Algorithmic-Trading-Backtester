# Simple Python Algorithmic Trading Backtester (GUI)

This is a small educational backtester with a **Tkinter** GUI.  
It downloads historical price data using **yfinance**, runs a **moving average crossover** strategy, and shows:

- Price chart with short/long moving averages
- BUY/SELL markers when the short MA crosses the long MA
- Strategy vs Buy-and-Hold equity curves
- Summary statistics (total return, max drawdown, number of trades, win rate, average trade return)

<img width="1372" height="907" alt="image" src="https://github.com/user-attachments/assets/c429abd8-e909-40fe-ba00-df0ca9f14a90" />


## 1. Installation

From the project folder (where `backtester.py` is located), create a virtual environment (optional but recommended) and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate      # on Windows
pip install -r requirements.txt
```

## 2. Running the App

```bash
python backtester.py
```

## 3. Using the Backtester

1. **Ticker**: Choose a ticker from the dropdown (or type your own, e.g. `AAPL`, `MSFT`, `SPY`).
2. **Date Range**: Set `Start` and `End` dates in `YYYY-MM-DD` format.
3. **Moving Averages**:
   - `Short MA (days)`: e.g. `50`
   - `Long MA (days)`: e.g. `200`  
   (Short MA must be less than Long MA.)
4. Click **Run Backtest**.

The app will:

- Download data via `yfinance`
- Compute short and long moving averages
- Generate BUY/SELL signals when short MA crosses above/below long MA
- Simulate a long-only strategy (in or out of the market)
- Plot:
  - Price + MAs + BUY/SELL markers
  - Equity curves (Strategy vs Buy & Hold)
- Display metrics:
  - Total Return (Strategy)
  - Total Return (Buy & Hold)
  - Max Drawdown (Strategy)
  - Number of Trades
  - Winning %
  - Average Trade Return

This is meant as a **simple, educational tool** and not as investment advice.

