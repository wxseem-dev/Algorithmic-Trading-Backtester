"""
Strategy definitions for the backtester.
BaseStrategy is abstract; concrete strategies add signals to the data.
"""
from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """Abstract base for all strategies. Subclasses must implement generate_signals."""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add signal columns to a copy of the data and return it.

        The returned DataFrame must include at least:
        - Close: price series (use Adj Close if available)
        - Signal: 1 = hold, 0 = not hold
        - Position_Change: diff of Signal (e.g. +1 buy, -1 sell)

        Optional columns for plotting (e.g. MA crossover): Short_MA, Long_MA.
        """
        pass


class MACrossoverStrategy(BaseStrategy):
    """Moving average crossover: hold when Short_MA > Long_MA."""

    def __init__(self, short_ma: int, long_ma: int):
        self.short_ma = short_ma
        self.long_ma = long_ma

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["Close"] = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]

        df["Short_MA"] = df["Close"].rolling(window=self.short_ma).mean()
        df["Long_MA"] = df["Close"].rolling(window=self.long_ma).mean()

        df["Signal"] = 0
        df.loc[df["Short_MA"] > df["Long_MA"], "Signal"] = 1

        df["Position_Change"] = df["Signal"].diff().fillna(0)
        return df


class MeanReversionStrategy(BaseStrategy):
    """Skeleton: always flat (signal 0). To be filled later."""

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["Close"] = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
        df["Signal"] = 0
        df["Position_Change"] = df["Signal"].diff().fillna(0)
        return df
