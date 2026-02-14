"""
Data fetching and normalization for the backtester.
Handles yfinance download and MultiIndex column flattening.
"""
import pandas as pd
import yfinance as yf

from logger import _agent_log


def fetch_data(ticker: str, start, end) -> pd.DataFrame:
    """
    Download OHLCV data for a ticker and normalize column structure.

    - Downloads via yfinance.
    - Flattens MultiIndex columns (e.g. ('Close', 'AAPL') -> 'Close') using
      xs(ticker) when the ticker is in the last level, otherwise droplevel(-1).

    :param ticker: Symbol to download (e.g. 'AAPL').
    :param start: Start date (datetime or date).
    :param end: End date (datetime or date).
    :returns: DataFrame with single-level columns; may be empty.
    :raises: Exception on download failure (caller should handle and show messagebox).
    """
    data = yf.download(ticker, start=start, end=end)

    if isinstance(getattr(data, "columns", None), pd.MultiIndex):
        try:
            if ticker in data.columns.get_level_values(-1):
                data = data.xs(ticker, axis=1, level=-1)
            else:
                data = data.droplevel(-1, axis=1)
        except Exception:
            _agent_log(
                {
                    "runId": "pre-fix",
                    "hypothesisId": "D",
                    "location": "data_handler.py:fetch_data:normalize_columns_error",
                    "message": "Failed to normalize MultiIndex columns",
                    "data": {
                        "ticker": ticker,
                        "columns_repr": repr(data.columns),
                    },
                }
            )
            raise ValueError("Unexpected data format returned from yfinance.")

    return data
