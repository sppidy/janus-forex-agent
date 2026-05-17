"""London Breakout Strategy.

Rules:
1. Define the pre-London range (Asian session consolidation)
2. Wait for London open (3 AM ET / 1:30 PM IST)
3. Trade the breakout direction when price breaks above/below the range
4. SL at the opposite side of the range
5. TP at 1.5x the range size
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta, timezone
from dataclasses import dataclass
from logger import logger

ET = timezone(timedelta(hours=-4))

# Pre-London range: 8 PM - 3 AM ET (Asian consolidation)
RANGE_START = time(20, 0)
RANGE_END = time(3, 0)
# Breakout window: 3 AM - 8 AM ET (London open)
BREAKOUT_START = time(3, 0)
BREAKOUT_END = time(8, 0)


@dataclass
class SweepSignal:
    symbol: str
    signal: str
    confidence: float
    price: float
    reason: str
    stop_loss: float | None = None
    target: float | None = None


def _get_pre_london_range(df: pd.DataFrame) -> tuple[float, float] | None:
    """Get the high and low of the pre-London (Asian) range."""
    if df.empty or len(df) < 20:
        return None

    now = datetime.now(ET)
    range_start_dt = datetime.combine(now.date() - timedelta(days=1), RANGE_START, tzinfo=ET)
    range_end_dt = datetime.combine(now.date(), RANGE_END, tzinfo=ET)

    try:
        idx = df.index
        if hasattr(idx, 'tz') and idx.tz is not None:
            idx = idx.tz_convert(ET)
        mask = (idx >= range_start_dt) & (idx <= range_end_dt)
        range_candles = df[mask]
    except Exception:
        range_candles = df[
            ((df.index.hour >= 20) & (df.index.hour <= 23)) |
            ((df.index.hour >= 0) & (df.index.hour <= 2))
        ].tail(30)

    if range_candles.empty or len(range_candles) < 3:
        return None

    return float(range_candles["High"].max()), float(range_candles["Low"].min())


def analyze_pair(symbol: str, df: pd.DataFrame) -> SweepSignal | None:
    """Check for London breakout setup."""
    result = _get_pre_london_range(df)
    if result is None:
        return None

    range_high, range_low = result
    range_size = range_high - range_low

    if range_size <= 0:
        return None

    # Check recent candles for breakout (most recent first)
    recent = df.tail(20)
    for i in range(len(recent) - 1, -1, -1):
        close = float(recent["Close"].iloc[i])
        high = float(recent["High"].iloc[i])
        low = float(recent["Low"].iloc[i])

        # Bullish breakout: close above range high
        if close > range_high and high > range_high:
            current = float(df["Close"].iloc[-1])
            # Only valid if we're still above range
            if current >= range_high:
                return SweepSignal(
                    symbol=symbol,
                    signal="BUY",
                    confidence=0.65,
                    price=current,
                    reason=f"London breakout above {range_high:.5f} (range: {range_size:.5f})",
                    stop_loss=round(range_low - range_size * 0.1, 5),
                    target=round(current + range_size * 1.5, 5),
                )

        # Bearish breakout: close below range low
        if close < range_low and low < range_low:
            current = float(df["Close"].iloc[-1])
            if current <= range_low:
                return SweepSignal(
                    symbol=symbol,
                    signal="SELL",
                    confidence=0.65,
                    price=current,
                    reason=f"London breakout below {range_low:.5f} (range: {range_size:.5f})",
                    stop_loss=round(range_high + range_size * 0.1, 5),
                    target=round(current - range_size * 1.5, 5),
                )

    return None
