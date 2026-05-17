"""ICT-based Forex Strategy: Asian Session Sweep + CISD + Fib 50% Rejection.

Rules:
1. Track Asian session High/Low (7 PM - 12 AM ET)
2. If Asian High (AH) gets swept → look for SELLS
   If Asian Low (AL) gets swept → look for BUYS
3. After sweep, detect CISD (Change in State of Delivery) — market structure shift
4. Calculate 50% Fibonacci retracement of the CISD push
5. Entry: price trades through Fib 50%, then candle CLOSES back through it
   - SELL: price goes above 50%, closes below 50%
   - BUY: price goes below 50%, closes above 50%
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta, timezone
from dataclasses import dataclass
from logger import logger

ET = timezone(timedelta(hours=-4))

# Asian session boundaries (ET)
ASIAN_START = time(19, 0)   # 7:00 PM ET (previous day)
ASIAN_END = time(0, 0)      # 12:00 AM ET (midnight)

# Kill zones for entries (higher probability)
LONDON_OPEN = time(3, 0)
LONDON_CLOSE = time(12, 0)
NY_OPEN = time(8, 0)
NY_CLOSE = time(17, 0)


@dataclass
class AsianRange:
    date: str           # Trading day (the day after the Asian session)
    high: float
    low: float
    high_time: str
    low_time: str


@dataclass
class SweepSignal:
    symbol: str
    signal: str         # "BUY" or "SELL"
    confidence: float
    price: float
    reason: str
    stop_loss: float | None = None
    target: float | None = None
    asian_high: float | None = None
    asian_low: float | None = None
    fib_50: float | None = None
    cisd_level: float | None = None


def _get_asian_range(df: pd.DataFrame) -> AsianRange | None:
    """Extract the most recent Asian session high/low from intraday data.

    Asian session = 7:00 PM to 12:00 AM ET (of the previous calendar day).
    """
    if df.empty or len(df) < 10:
        return None

    # data_fetcher strips timezone from index, so work with naive datetimes
    idx = df.index
    if hasattr(idx, 'tz') and idx.tz is not None:
        try:
            idx = idx.tz_convert(ET)
        except Exception:
            pass
        df = df.copy()
        df.index = idx

    now = datetime.now(ET)
    asian_date = now.date()

    if hasattr(df.index, 'tz') and df.index.tz is not None:
        asian_start_dt = datetime.combine(asian_date - timedelta(days=1), ASIAN_START, tzinfo=ET)
        asian_end_dt = datetime.combine(asian_date, ASIAN_END, tzinfo=ET)
    else:
        asian_start_dt = datetime.combine(asian_date - timedelta(days=1), ASIAN_START)
        asian_end_dt = datetime.combine(asian_date, ASIAN_END)

    try:
        mask = (df.index >= asian_start_dt) & (df.index <= asian_end_dt)
        asian_candles = df[mask]
    except Exception:
        asian_candles = df[
            ((df.index.hour >= 19) & (df.index.hour <= 23)) |
            (df.index.hour == 0)
        ].tail(20)

    if asian_candles.empty or len(asian_candles) < 2:
        return None

    ah = float(asian_candles["High"].max())
    al = float(asian_candles["Low"].min())
    ah_time = str(asian_candles["High"].idxmax())
    al_time = str(asian_candles["Low"].idxmin())

    return AsianRange(
        date=str(asian_date),
        high=ah,
        low=al,
        high_time=ah_time,
        low_time=al_time,
    )


def _find_swing_points(df: pd.DataFrame, lookback: int = 3) -> tuple[list, list]:
    """Find swing highs and swing lows in the data."""
    swing_highs = []
    swing_lows = []

    highs = df["High"].values
    lows = df["Low"].values

    for i in range(lookback, len(df) - lookback):
        # Swing high: higher than `lookback` candles on each side
        if all(highs[i] >= highs[i - j] for j in range(1, lookback + 1)) and \
           all(highs[i] >= highs[i + j] for j in range(1, lookback + 1)):
            swing_highs.append((i, highs[i]))

        # Swing low: lower than `lookback` candles on each side
        if all(lows[i] <= lows[i - j] for j in range(1, lookback + 1)) and \
           all(lows[i] <= lows[i + j] for j in range(1, lookback + 1)):
            swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows


def _detect_cisd(df: pd.DataFrame, sweep_type: str, sweep_idx: int) -> tuple[int, float, float] | None:
    """Detect Change in State of Delivery after a sweep.

    For AH sweep (looking for SELLS):
      - Find where a candle closes below the most recent swing low after the sweep
      - This confirms bearish CISD

    For AL sweep (looking for BUYS):
      - Find where a candle closes above the most recent swing high after the sweep
      - This confirms bullish CISD

    Returns (cisd_bar_idx, cisd_push_high, cisd_push_low) or None.
    """
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values

    _, swing_lows = _find_swing_points(df.iloc[:sweep_idx + 5], lookback=2)
    swing_highs, _ = _find_swing_points(df.iloc[:sweep_idx + 5], lookback=2)

    if sweep_type == "AH_SWEPT":
        # Looking for bearish CISD: close below recent swing low
        recent_swing_lows = [sl for sl in swing_lows if sl[0] < sweep_idx]
        if not recent_swing_lows:
            return None
        ref_low = recent_swing_lows[-1][1]  # Most recent swing low

        for i in range(sweep_idx, min(sweep_idx + 20, len(df))):
            if closes[i] < ref_low:
                # CISD confirmed — the push is from sweep high to this close
                push_high = max(highs[sweep_idx:i + 1])
                push_low = min(lows[sweep_idx:i + 1])
                return (i, push_high, push_low)

    elif sweep_type == "AL_SWEPT":
        # Looking for bullish CISD: close above recent swing high
        recent_swing_highs = [sh for sh in swing_highs if sh[0] < sweep_idx]
        if not recent_swing_highs:
            return None
        ref_high = recent_swing_highs[-1][1]

        for i in range(sweep_idx, min(sweep_idx + 20, len(df))):
            if closes[i] > ref_high:
                push_high = max(highs[sweep_idx:i + 1])
                push_low = min(lows[sweep_idx:i + 1])
                return (i, push_high, push_low)

    return None


def _check_fib_rejection(df: pd.DataFrame, cisd_idx: int, fib_50: float,
                          sweep_type: str) -> int | None:
    """Check if price rejected off the Fib 50% level after CISD.

    SELL setup (AH swept):
      - Price trades ABOVE fib_50 (wick above), then candle CLOSES BELOW fib_50

    BUY setup (AL swept):
      - Price trades BELOW fib_50 (wick below), then candle CLOSES ABOVE fib_50

    Returns the bar index of the rejection candle, or None.
    """
    for i in range(cisd_idx + 1, min(cisd_idx + 30, len(df))):
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]
        close = df["Close"].iloc[i]

        if sweep_type == "AH_SWEPT":
            # Wick touched above fib, closed below
            if high >= fib_50 and close < fib_50:
                return i
        elif sweep_type == "AL_SWEPT":
            # Wick touched below fib, closed above
            if low <= fib_50 and close > fib_50:
                return i

    return None


def analyze_pair(symbol: str, df_15m: pd.DataFrame) -> SweepSignal | None:
    """Run the full ICT strategy on a forex pair using 15m candles.

    Returns a SweepSignal if all conditions are met, else None.
    """
    if df_15m.empty or len(df_15m) < 50:
        return None

    # Step 1: Get Asian range
    asian = _get_asian_range(df_15m)
    if asian is None:
        return None

    # Step 2: Check for sweeps in post-Asian price action
    closes = df_15m["Close"].values
    highs = df_15m["High"].values
    lows = df_15m["Low"].values

    ah_swept_idx = None
    al_swept_idx = None

    # Look at the most recent candles (post-Asian, during London/NY)
    lookback_start = max(0, len(df_15m) - 80)  # ~20 hours of 15m candles
    for i in range(lookback_start, len(df_15m)):
        if highs[i] > asian.high and ah_swept_idx is None:
            ah_swept_idx = i
        if lows[i] < asian.low and al_swept_idx is None:
            al_swept_idx = i

    # Prioritize the most recent sweep
    sweep_type = None
    sweep_idx = None

    if ah_swept_idx is not None and al_swept_idx is not None:
        # Both swept — use the later one
        if ah_swept_idx > al_swept_idx:
            sweep_type = "AH_SWEPT"
            sweep_idx = ah_swept_idx
        else:
            sweep_type = "AL_SWEPT"
            sweep_idx = al_swept_idx
    elif ah_swept_idx is not None:
        sweep_type = "AH_SWEPT"
        sweep_idx = ah_swept_idx
    elif al_swept_idx is not None:
        sweep_type = "AL_SWEPT"
        sweep_idx = al_swept_idx
    else:
        return None  # No sweep occurred

    # Step 3: Detect CISD
    cisd = _detect_cisd(df_15m, sweep_type, sweep_idx)
    if cisd is None:
        return None

    cisd_idx, push_high, push_low = cisd

    # Step 4: Calculate Fib 50%
    fib_50 = (push_high + push_low) / 2

    # Step 5: Check for Fib 50% rejection
    rejection_idx = _check_fib_rejection(df_15m, cisd_idx, fib_50, sweep_type)
    if rejection_idx is None:
        return None

    # All conditions met — generate signal
    current_price = float(closes[-1])
    latest_high = float(highs[-1])
    latest_low = float(lows[-1])

    if sweep_type == "AH_SWEPT":
        # SELL signal
        stop_loss = push_high * 1.0005  # Just above the push high
        risk = abs(current_price - stop_loss)
        target = current_price - (risk * 2)  # 2:1 RR

        confidence = _calculate_confidence(df_15m, rejection_idx, sweep_type, asian)

        return SweepSignal(
            symbol=symbol,
            signal="SELL",
            confidence=confidence,
            price=current_price,
            reason=f"AH {asian.high:.5f} swept, bearish CISD confirmed, Fib 50% ({fib_50:.5f}) rejection",
            stop_loss=round(stop_loss, 5),
            target=round(target, 5),
            asian_high=asian.high,
            asian_low=asian.low,
            fib_50=round(fib_50, 5),
            cisd_level=round(push_low, 5),
        )

    elif sweep_type == "AL_SWEPT":
        # BUY signal
        stop_loss = push_low * 0.9995  # Just below the push low
        risk = abs(stop_loss - current_price)
        target = current_price + (risk * 2)  # 2:1 RR

        confidence = _calculate_confidence(df_15m, rejection_idx, sweep_type, asian)

        return SweepSignal(
            symbol=symbol,
            signal="BUY",
            confidence=confidence,
            price=current_price,
            reason=f"AL {asian.low:.5f} swept, bullish CISD confirmed, Fib 50% ({fib_50:.5f}) rejection",
            stop_loss=round(stop_loss, 5),
            target=round(target, 5),
            asian_high=asian.high,
            asian_low=asian.low,
            fib_50=round(fib_50, 5),
            cisd_level=round(push_high, 5),
        )

    return None


def _calculate_confidence(df: pd.DataFrame, rejection_idx: int,
                           sweep_type: str, asian: AsianRange) -> float:
    """Score the setup quality 0.0-1.0."""
    score = 0.5  # Base confidence

    # Bonus: rejection happened recently (fresh setup)
    bars_since_rejection = len(df) - 1 - rejection_idx
    if bars_since_rejection <= 3:
        score += 0.15
    elif bars_since_rejection <= 8:
        score += 0.08

    # Bonus: in a kill zone (London or NY open)
    now = datetime.now(ET).time()
    if LONDON_OPEN <= now <= LONDON_CLOSE or NY_OPEN <= now <= NY_CLOSE:
        score += 0.10

    # Bonus: tight Asian range (more liquidity above/below to grab)
    asian_range_pips = abs(asian.high - asian.low)
    avg_price = (asian.high + asian.low) / 2
    if avg_price > 0:
        range_pct = asian_range_pips / avg_price
        if range_pct < 0.003:  # Tight range
            score += 0.10
        elif range_pct < 0.005:
            score += 0.05

    # Bonus: volume confirmation (if available)
    if "Volume" in df.columns:
        recent_vol = df["Volume"].iloc[-5:].mean()
        avg_vol = df["Volume"].mean()
        if avg_vol > 0 and recent_vol > avg_vol * 1.3:
            score += 0.05

    return min(score, 0.95)


def scan_all_pairs(pairs: list[str], get_data_fn) -> list[dict]:
    """Scan all forex pairs for ICT setups.

    Args:
        pairs: list of forex symbols
        get_data_fn: function(symbol, period, interval) -> DataFrame
    """
    signals = []
    for symbol in pairs:
        try:
            df = get_data_fn(symbol, period="5d", interval="15m")
            if df.empty or len(df) < 50:
                signals.append({
                    "symbol": symbol,
                    "signal": "HOLD",
                    "confidence": 0,
                    "price": 0,
                    "reason": "Insufficient data",
                })
                continue

            result = analyze_pair(symbol, df)
            if result:
                signals.append({
                    "symbol": result.symbol,
                    "signal": result.signal,
                    "confidence": result.confidence,
                    "price": result.price,
                    "reason": result.reason,
                    "stop_loss": result.stop_loss,
                    "target": result.target,
                    "position_size_pct": min(result.confidence * 0.05, 0.05),
                    "asian_high": result.asian_high,
                    "asian_low": result.asian_low,
                    "fib_50": result.fib_50,
                })
            else:
                # No setup — return HOLD with Asian range info
                df_15m = get_data_fn(symbol, period="5d", interval="15m")
                asian = _get_asian_range(df_15m) if not df_15m.empty else None
                price = float(df["Close"].iloc[-1]) if not df.empty else 0
                signals.append({
                    "symbol": symbol,
                    "signal": "HOLD",
                    "confidence": 0,
                    "price": price,
                    "reason": f"No ICT setup. AH: {asian.high:.5f} AL: {asian.low:.5f}" if asian else "No Asian range data",
                    "asian_high": asian.high if asian else None,
                    "asian_low": asian.low if asian else None,
                })
        except Exception as e:
            logger.warning(f"Error analyzing {symbol}: {e}")
            signals.append({
                "symbol": symbol,
                "signal": "HOLD",
                "confidence": 0,
                "price": 0,
                "reason": f"Error: {str(e)[:80]}",
            })

    return signals
