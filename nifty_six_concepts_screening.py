"""
Six Additional Oliver Velez Concepts -- SCREENING backtest
=================================================================
Standalone, 2-year backtest across all six concepts from the doc, to
answer "which one is worth optimizing further" -- this is a SCREENING
pass, not a full sweep on each concept (that's the natural next step
for whichever concept(s) show promise here, same as Velez/Concepts2&4/
Crash Play got their own dedicated sweeps after an initial signal).

SCOPE DECISIONS (read before trusting the numbers):

  - Several sections reference companion documents not provided here
    (the standalone elephant-bar system doc, the 8-step narrow-state
    doc, the three-finger-spread doc, the 20-MA-halt doc) -- their exact
    thresholds aren't available, so this script defines its own
    reasonable, clearly-labeled operational versions of: "near the 20
    MA", "picture of power", "position zone", elephant-bar size
    threshold, and narrow/hovering state. These are documented inline.
  - Section 3 (elephant bar position sizing) is NOT run as its own
    standalone strategy -- it's an execution/sizing detail (50% of
    capital on entry), not an independent signal. Its "first vs
    follow-through elephant" distinction IS implemented and used inside
    the Power Bar Start module (Section 5.1), where it's directly
    relevant.
  - Every module uses the SAME single stop/trail (8pt initial / 4pt
    trailing-after-profit, Velez's own already-optimized values) rather
    than a full sweep -- this keeps 5 separate concepts tractable in one
    pass. Whichever concept looks best here should get its own dedicated
    stop/target/trail sweep next, the same treatment Velez, Concepts2&4,
    and the Crash Play already received.
  - The doc's own most-specific, most-falsifiable claims are tested
    directly, per its own "General backtesting note":
      - RBI/GBI: ~87% accuracy under all 5 hard rules (Section 4.2),
        vs. a relaxed variant dropping rule 4 (the near-20MA filter).
      - Power Bar Start: ~90% follow-through when the reversal bar is
        elephant-scale (Section 5.1), vs. ordinary-sized reversal bars.
      - Section 5.4's stacking-score hypothesis (0-4 elements present).
      - Section 2.1's 50%-retracement filter, applied as an overlay to
        the RBI/GBI and swing-trade signals.
  - Trade direction: buys CE for long-biased signals, PE for
    short-biased signals (long premium only, never writes options),
    same proxy premium simulation as every other script in this project.

FIVE STRATEGY MODULES (Section 3 folded into Power Bar Start's "first
elephant" check, Section 2 implemented as an overlay rather than a
standalone module):
  1. RBI/GBI, 5 hard rules (Section 4)
  2. Power Bar Start (Section 5.1)
  3. Market Law 4 + stacking score (Section 5.2/5.4)
  4. Sloping-20MA swing trade (Section 1)
  5. Pinch Play / Apex Breakout + OMG halt (Section 6)

DIAGNOSTICS ADDED after the first real run: three modules (RBI/GBI
relaxed, Power Bar Start, Market Law 4) came back showing SQN > 6.6 and
near-identical net P&L/drawdown -- both are red flags (SQN > 6 is
essentially "holy grail" territory, and three independently-coded
signal functions landing on almost the same numbers smells like an
entry-frequency artifact, not a shared edge). The diagnostics block at
the end of main() reports, per module: what % of eligible days it
actually fired on, CE vs PE split, and entry-hour clustering, so this
can be checked directly against the raw numbers instead of taken on
faith.
"""

import datetime as dt
import math
import pandas as pd
import numpy as np

import dhan_live_bot as dlb

FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")   # 2 years
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85
BAR_RULE = "2min"

STRIKE_STEP = 50
EOD_SQUAREOFF = dt.time(15, 20)

SMA_FAST = 20
SMA_SLOW = 200
AVG_RANGE_LOOKBACK = 20
ELEPHANT_MULT = 2.0
SMALL_BAR_MULT = 0.6
NEAR_MA_MULT = 0.3          # "near the 20 MA" = within 0.3x avg_range
FIRST_ELEPHANT_LOOKBACK = 7  # bars checked for a preceding elephant bar in the same direction

STOP_PTS = 8.0    # single default for this screening pass (Velez's own optimized value)
TRAIL_PTS = 4.0

ASSUMED_DELTA = 0.6
ASSUMED_TIME_VALUE = 40.0

STARTING_CAPITAL = 25_000
BROKERAGE_PER_TRADE = 40.0


def lot_size_for_date(d):
    if d < dt.date(2024, 11, 20):
        return 25
    elif d < dt.date(2026, 1, 1):
        return 75
    else:
        return 65


def itm_call_strike(spot):
    return math.floor(spot / STRIKE_STEP) * STRIKE_STEP


def itm_put_strike(spot):
    return math.ceil(spot / STRIKE_STEP) * STRIKE_STEP


def fetch_index_chunked(from_date, to_date, chunk_days=CHUNK_DAYS):
    start = dt.datetime.strptime(from_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(to_date, "%Y-%m-%d").date()
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + dt.timedelta(days=chunk_days - 1), end)
        resp = dlb.dhan.intraday_minute_data(
            security_id=dlb.UNDERLYING_SECURITY_ID, exchange_segment=dlb.UNDERLYING_EXCHANGE_SEGMENT,
            instrument_type=dlb.UNDERLYING_INSTRUMENT_TYPE,
            from_date=cur.strftime("%Y-%m-%d"), to_date=chunk_end.strftime("%Y-%m-%d"),
        )
        data = resp.get("data", resp)
        df = pd.DataFrame({
            "open": data["open"], "high": data["high"], "low": data["low"], "close": data["close"],
            "volume": data["volume"], "timestamp": data["timestamp"],
        })
        df["time"] = pd.to_datetime(df["timestamp"], unit="s") + pd.Timedelta(hours=5, minutes=30)
        chunks.append(df.set_index("time").sort_index())
        cur = chunk_end + dt.timedelta(days=1)
    full = pd.concat(chunks)
    return full[~full.index.duplicated(keep="first")].sort_index()


def resample_2min_per_day(df, bar_rule=BAR_RULE):
    out = []
    for day in sorted(set(df.index.date)):
        day_df = df[df.index.date == day]
        r = day_df.resample(bar_rule, origin=day_df.index[0]).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        out.append(r)
    return pd.concat(out).sort_index()


def build_premium_series(day_df, strike, day_open_spot, side):
    idx_open = day_df["open"].to_numpy()
    idx_high = day_df["high"].to_numpy()
    idx_low = day_df["low"].to_numpy()
    idx_close = day_df["close"].to_numpy()
    if side == "CE":
        intrinsic0 = max(day_open_spot - strike, 0.0)
        mo, mh, ml, mc = (idx_open - day_open_spot, idx_high - day_open_spot,
                          idx_low - day_open_spot, idx_close - day_open_spot)
    else:
        intrinsic0 = max(strike - day_open_spot, 0.0)
        mo, mh, ml, mc = (day_open_spot - idx_open, day_open_spot - idx_low,
                          day_open_spot - idx_high, day_open_spot - idx_close)
    base = intrinsic0 + ASSUMED_TIME_VALUE
    p_open = np.maximum(base + mo * ASSUMED_DELTA, 0.0)
    p_high = np.maximum(base + mh * ASSUMED_DELTA, 0.0)
    p_low = np.maximum(base + ml * ASSUMED_DELTA, 0.0)
    p_close = np.maximum(base + mc * ASSUMED_DELTA, 0.0)
    return pd.DataFrame({"open": p_open, "high": p_high, "low": p_low, "close": p_close}, index=day_df.index)


# ----------------------------- shared primitives -----------------------------

def is_green(o, c):
    return c > o


def is_red(o, c):
    return c < o


def is_elephant(o, h, l, c, avgr, mult=ELEPHANT_MULT):
    if not avgr or avgr <= 0 or pd.isna(avgr):
        return False
    return (h - l) >= mult * avgr


def is_small(rng, avgr):
    return bool(avgr) and avgr > 0 and not pd.isna(avgr) and rng < SMALL_BAR_MULT * avgr


def near_ma(price, ma, avgr, mult=NEAR_MA_MULT):
    if ma is None or avgr is None or avgr <= 0 or pd.isna(ma) or pd.isna(avgr):
        return False
    return abs(price - ma) <= mult * avgr


def compute_indicators(df):
    df = df.copy()
    df["sma20"] = df["close"].rolling(SMA_FAST, min_periods=SMA_FAST).mean()
    df["sma200"] = df["close"].rolling(SMA_SLOW, min_periods=SMA_SLOW).mean()
    df["range"] = df["high"] - df["low"]
    df["avg_range"] = df["range"].shift(1).rolling(AVG_RANGE_LOOKBACK, min_periods=AVG_RANGE_LOOKBACK).mean()
    return df


def compute_slope_stage(df):
    """Vectorized: direction = sign of sma20 change over a 5-bar lookback;
    duration = consecutive bars the current direction has held (run-length);
    stage 1 = fresh slope + narrow MA separation, stage 2 = the 'sweet spot',
    stage 3 = long-running slope or wide separation ('mature/extended').
    Adds 'ma_direction', 'slope_duration', 'ma_sep_atr', 'slope_stage' columns."""
    sma20 = df["sma20"].to_numpy()
    sma200 = df["sma200"].to_numpy()
    avgr = df["avg_range"].to_numpy()
    n = len(df)

    direction = np.full(n, 0)
    for i in range(5, n):
        if pd.isna(sma20[i]) or pd.isna(sma20[i - 5]):
            continue
        if sma20[i] > sma20[i - 5]:
            direction[i] = 1
        elif sma20[i] < sma20[i - 5]:
            direction[i] = -1

    duration = np.zeros(n, dtype=int)
    for i in range(n):
        if i == 0 or direction[i] == 0 or direction[i] != direction[i - 1]:
            duration[i] = 1 if direction[i] != 0 else 0
        else:
            duration[i] = duration[i - 1] + 1

    sep_atr = np.full(n, np.nan)
    valid = (~pd.isna(sma20)) & (~pd.isna(sma200)) & (~pd.isna(avgr)) & (avgr > 0)
    sep_atr[valid] = np.abs(sma20[valid] - sma200[valid]) / avgr[valid]

    stage = np.full(n, 0)
    for i in range(n):
        if direction[i] == 0 or np.isnan(sep_atr[i]):
            continue
        if duration[i] <= 10 and sep_atr[i] < 2.0:
            stage[i] = 1
        elif duration[i] <= 40 and sep_atr[i] < 4.0:
            stage[i] = 2
        else:
            stage[i] = 3

    df = df.copy()
    df["ma_direction"] = direction
    df["slope_duration"] = duration
    df["ma_sep_atr"] = sep_atr
    df["slope_stage"] = stage
    return df


def is_first_elephant(highs, lows, opens, closes, avg_ranges, i, direction, lookback=FIRST_ELEPHANT_LOOKBACK):
    """direction=1 checks for a preceding bullish elephant bar, -1 bearish."""
    for k in range(max(0, i - lookback), i):
        if is_elephant(opens[k], highs[k], lows[k], closes[k], avg_ranges[k]):
            if direction == 1 and is_green(opens[k], closes[k]):
                return False
            if direction == -1 and is_red(opens[k], closes[k]):
                return False
    return True


def fifty_pct_zone_check(highs, lows, i, lookback=30):
    """Finds the most recent significant swing (extreme high + its preceding
    origin low, or mirror) within `lookback` bars before i, and returns
    'above_50' / 'below_50' for whether the CURRENT price sits above or below
    the 50% retracement of that swing (used as an overlay filter, Section 2.1)."""
    if i < lookback:
        return None
    window_highs = list(highs[i - lookback:i])
    window_lows = list(lows[i - lookback:i])
    swing_high = max(window_highs)
    # origin = the lowest low BEFORE the swing high within the window
    high_idx_local = window_highs.index(swing_high)
    if high_idx_local == 0:
        return None
    origin_low = min(window_lows[:high_idx_local])
    fifty = origin_low + 0.5 * (swing_high - origin_low)
    current = (highs[i] + lows[i]) / 2.0
    return "above_50" if current >= fifty else "below_50"


def run_self_checks():
    print("Running self-checks (synthetic data)...")

    assert is_green(10, 11) and not is_green(11, 10)
    assert is_red(11, 10) and not is_red(10, 11)
    assert is_elephant(100, 110, 90, 91, 5) is True     # range 20 >= 2*5
    assert is_elephant(100, 101, 99, 100.5, 5) is False
    assert is_small(2, 10) is True and is_small(8, 10) is False
    assert near_ma(100, 100.2, 1.0) is True     # within 0.3
    assert near_ma(100, 105, 1.0) is False
    print("  shared primitive sanity checks ok")

    # slope_stage: construct a clean, steadily rising SMA20 with a flat SMA200
    n = 250
    idx = pd.date_range("2025-01-06 09:15", periods=n, freq="2min")
    closes = np.concatenate([np.full(200, 100.0), 100.0 + np.arange(1, n - 199) * 0.3])
    df = pd.DataFrame({"open": closes, "high": closes + 0.2, "low": closes - 0.2, "close": closes}, index=idx)
    df = compute_indicators(df)
    df = compute_slope_stage(df)
    # near the very end, sma20 should be rising (direction=1) with a short-ish duration
    assert df["ma_direction"].iloc[-1] == 1, "expected a rising 20MA direction at the end of this ramp"
    print("  compute_slope_stage direction-detection check ok")

    # is_first_elephant: an elephant bar with NO preceding elephant in lookback -> first
    closes2 = [100] * 10
    opens2 = [100] * 10
    highs2 = [100.1] * 10
    lows2 = [99.9] * 10
    avgr2 = [0.2] * 10
    opens2[7], closes2[7], highs2[7], lows2[7] = 100, 110, 110, 100   # elephant bar at idx7
    assert is_first_elephant(highs2, lows2, opens2, closes2, avgr2, 7, direction=1) is True
    # now put ANOTHER bullish elephant bar right before it at idx5 -> idx7 is now "follow-through"
    opens2[5], closes2[5], highs2[5], lows2[5] = 100, 108, 108, 100
    assert is_first_elephant(highs2, lows2, opens2, closes2, avgr2, 7, direction=1) is False
    print("  is_first_elephant first-vs-followthrough check ok")

    # 50% zone check: swing from 100 (origin) up to 120 (high) within the window;
    # current price at 108 is below the 50% level (110) -> below_50
    highs3 = [100] * 5 + [105, 110, 120, 118, 115] + [108] * 5
    lows3 = [99] * 5 + [100, 104, 115, 113, 110] + [107] * 5
    z = fifty_pct_zone_check(highs3, lows3, i=14, lookback=14)
    assert z == "below_50", f"expected below_50, got {z}"
    print("  fifty_pct_zone_check check ok")

    print("All self-checks passed (part 1 of self-checks; more follow per module).\n")


# ----------------------------- trade management (shared) -----------------------------

def simulate_trade(premium_df, entry_idx, stop_pts=STOP_PTS, trail_pts=TRAIL_PTS):
    """Always simulated as a long-premium trade (we only ever buy CE or PE).
    Hard stop at entry-stop_pts until the position first goes into profit
    (peak premium > entry), at which point a trailing stop (peak-trail_pts)
    takes over and only ever ratchets up. EOD square-off fallback."""
    entry = premium_df["close"].iloc[entry_idx]
    stop_level = entry - stop_pts
    peak = entry
    trailing_active = False
    n = len(premium_df)
    for j in range(entry_idx + 1, n):
        bar = premium_df.iloc[j]
        t = premium_df.index[j].time()
        if bar["high"] > peak:
            peak = bar["high"]
        if peak > entry:
            trailing_active = True
        if trailing_active:
            stop_level = max(stop_level, peak - trail_pts)
        if bar["low"] <= stop_level:
            reason = "TRAIL" if trailing_active else "STOP"
            return stop_level, reason, premium_df.index[j]
        if t >= EOD_SQUAREOFF:
            return bar["close"], "EOD", premium_df.index[j]
    last = premium_df.iloc[-1]
    return last["close"], "EOD", premium_df.index[-1]


# ----------------------------- module 1: RBI / GBI five hard rules -----------------------------

def rbi_gbi_signal(df, i, relaxed=False):
    """Rally-Base-Rally (long) / Grind-Base-Instability (short) with Velez's
    5 hard rules (Section 4): (1) trade with the 20MA slope, (2) only buy
    above / short below the 200MA, (3) entries near the 20MA, (4) 'picture
    of power' -- 20MA and 200MA aligned in the trade direction, (5) a
    confirming color bar. relaxed=True drops rule (3) (the near-20MA filter),
    per the doc's own suggested relaxation test."""
    if pd.isna(df["sma20"].iloc[i]) or pd.isna(df["sma200"].iloc[i]) or pd.isna(df["avg_range"].iloc[i]):
        return None
    close = df["close"].iloc[i]
    o, c = df["open"].iloc[i], df["close"].iloc[i]
    sma20, sma200, avgr = df["sma20"].iloc[i], df["sma200"].iloc[i], df["avg_range"].iloc[i]
    direction = df["ma_direction"].iloc[i]

    long_ok = (direction == 1) and (close > sma200) and (sma20 > sma200) and is_green(o, c)
    if not relaxed:
        long_ok = long_ok and near_ma(close, sma20, avgr)
    if long_ok:
        return "LONG"

    short_ok = (direction == -1) and (close < sma200) and (sma20 < sma200) and is_red(o, c)
    if not relaxed:
        short_ok = short_ok and near_ma(close, sma20, avgr)
    if short_ok:
        return "SHORT"
    return None


# ----------------------------- module 2: Power Bar Start -----------------------------

def power_bar_start_signal(df, i):
    """A reversal bar (color-change vs the immediately preceding bar) that
    breaks the prior bar's extreme. Returns (direction, 'ELEPHANT'/'ORDINARY')
    if found, else None. ELEPHANT = the reversal bar itself is an elephant
    bar per is_elephant(); tests the doc's ~90% follow-through claim
    (elephant-scale reversal bars vs ordinary ones)."""
    if i < 1 or pd.isna(df["avg_range"].iloc[i]):
        return None
    o, h, l, c = df["open"].iloc[i], df["high"].iloc[i], df["low"].iloc[i], df["close"].iloc[i]
    po, ph, pl, pc = (df["open"].iloc[i - 1], df["high"].iloc[i - 1],
                      df["low"].iloc[i - 1], df["close"].iloc[i - 1])
    avgr = df["avg_range"].iloc[i]

    if is_red(po, pc) and is_green(o, c) and h > ph:
        scale = "ELEPHANT" if is_elephant(o, h, l, c, avgr) else "ORDINARY"
        return "LONG", scale
    if is_green(po, pc) and is_red(o, c) and l < pl:
        scale = "ELEPHANT" if is_elephant(o, h, l, c, avgr) else "ORDINARY"
        return "SHORT", scale
    return None


def follow_through_ok(closes, i, direction, bars_ahead=5):
    """Did price keep moving in `direction` for the next `bars_ahead` bars
    (close bars_ahead later beyond the signal bar's close, same direction)?"""
    j = i + bars_ahead
    if j >= len(closes):
        return None  # inconclusive, not enough data
    if direction == "LONG":
        return closes[j] > closes[i]
    else:
        return closes[j] < closes[i]


# ----------------------------- module 3: Market Law 4 + stacking score -----------------------------

def market_law4_signal(df, i, lookback=15):
    """Failed-swing reversal: price attempts a new swing low (down-move) but
    fails to sustain it and closes back above the prior minor swing low
    within `lookback` bars -- a 'failed breakdown' (long), or the mirror
    'failed breakout' (short). Simplified operational version (Section 5.2)."""
    if i < lookback:
        return None
    lows = df["low"].to_numpy()
    highs = df["high"].to_numpy()
    closes = df["close"].to_numpy()
    window_lows = lows[i - lookback:i]
    window_highs = highs[i - lookback:i]
    prior_swing_low = min(window_lows[:-3]) if len(window_lows) > 3 else None
    prior_swing_high = max(window_highs[:-3]) if len(window_highs) > 3 else None

    # failed breakdown: made a new low vs prior swing low in last 3 bars, then closed back above it
    recent_lows = lows[i - 3:i]
    if prior_swing_low is not None and min(recent_lows) < prior_swing_low and closes[i] > prior_swing_low:
        return "LONG"
    recent_highs = highs[i - 3:i]
    if prior_swing_high is not None and max(recent_highs) > prior_swing_high and closes[i] < prior_swing_high:
        return "SHORT"
    return None


def stacking_score(df, i, direction):
    """0-4 count of supportive elements present at the signal bar (Section
    5.4): (1) 20MA sloping with the trade direction, (2) price on the
    correct side of the 200MA, (3) entry near the 20MA, (4) the signal bar
    is elephant-scale."""
    score = 0
    ma_dir = df["ma_direction"].iloc[i]
    close = df["close"].iloc[i]
    sma20, sma200, avgr = df["sma20"].iloc[i], df["sma200"].iloc[i], df["avg_range"].iloc[i]
    o, h, l, c = df["open"].iloc[i], df["high"].iloc[i], df["low"].iloc[i], df["close"].iloc[i]
    if pd.isna(sma20) or pd.isna(sma200) or pd.isna(avgr):
        return 0
    if direction == "LONG":
        if ma_dir == 1:
            score += 1
        if close > sma200:
            score += 1
    else:
        if ma_dir == -1:
            score += 1
        if close < sma200:
            score += 1
    if near_ma(close, sma20, avgr):
        score += 1
    if is_elephant(o, h, l, c, avgr):
        score += 1
    return score


# ----------------------------- module 4: sloping 20MA swing trade -----------------------------

def swing_trade_signal(df, i):
    """Entry when slope_stage is freshly in stage 2 (the doc's 'sweet spot')
    with a pullback to the 20MA followed by a confirming bar in the trend
    direction (Section 1)."""
    if pd.isna(df["sma20"].iloc[i]) or pd.isna(df["avg_range"].iloc[i]):
        return None
    stage = df["slope_stage"].iloc[i]
    direction = df["ma_direction"].iloc[i]
    close = df["close"].iloc[i]
    sma20, avgr = df["sma20"].iloc[i], df["avg_range"].iloc[i]
    o, c = df["open"].iloc[i], df["close"].iloc[i]
    if stage != 2 or not near_ma(close, sma20, avgr):
        return None
    if direction == 1 and is_green(o, c):
        return "LONG"
    if direction == -1 and is_red(o, c):
        return "SHORT"
    return None


# ----------------------------- module 5: pinch play / apex breakout -----------------------------

def is_hovering(df, i, lookback=10, flat_thresh_mult=0.15):
    """20MA rising while 200MA is roughly flat, and the gap between them is
    narrowing over `lookback` bars ('pinch'/hovering state, Section 6)."""
    if i < lookback or pd.isna(df["sma200"].iloc[i]) or pd.isna(df["avg_range"].iloc[i]):
        return False
    sma20_now, sma20_prev = df["sma20"].iloc[i], df["sma20"].iloc[i - lookback]
    sma200_now, sma200_prev = df["sma200"].iloc[i], df["sma200"].iloc[i - lookback]
    avgr = df["avg_range"].iloc[i]
    if pd.isna(sma20_prev) or pd.isna(sma200_prev) or avgr <= 0:
        return False
    sma200_flat = abs(sma200_now - sma200_prev) <= flat_thresh_mult * avgr
    sma20_rising = sma20_now > sma20_prev
    sep_now = abs(sma20_now - sma200_now)
    sep_prev = abs(sma20_prev - sma200_prev)
    narrowing = sep_now < sep_prev
    return bool(sma200_flat and sma20_rising and narrowing)


def pinch_play_signal(df, i, breakout_lookback=10):
    """Apex breakout: while hovering, price breaks above the recent
    `breakout_lookback`-bar high. OMG halt: at the moment of breakout price
    is ALSO simultaneously near the 200MA (both MAs being tested at once) --
    flagged separately for comparison rather than excluded."""
    if not is_hovering(df, i, lookback=breakout_lookback):
        return None
    highs = df["high"].to_numpy()
    recent_high = highs[i - breakout_lookback:i].max()
    close = df["close"].iloc[i]
    if close <= recent_high:
        return None
    sma200, avgr = df["sma200"].iloc[i], df["avg_range"].iloc[i]
    omg_halt = near_ma(close, sma200, avgr)
    return "LONG", ("OMG_HALT" if omg_halt else "STANDARD")


def run_module_self_checks():
    print("Running module self-checks (synthetic data)...")

    # --- RBI/GBI ---
    idx = pd.date_range("2025-01-06 09:15", periods=1, freq="2min")
    row = pd.DataFrame({"open": [100], "high": [100.5], "low": [99.8], "close": [100.2],
                         "sma20": [100.1], "sma200": [98.0], "avg_range": [0.5],
                         "ma_direction": [1]}, index=idx)
    assert rbi_gbi_signal(row, 0) == "LONG", "expected a clean RBI long signal"
    row2 = row.copy()
    row2["sma200"] = 102.0  # now below 200MA -> rule 2 fails
    assert rbi_gbi_signal(row2, 0) is None
    row3 = row.copy()
    row3["close"] = 101.5   # far from sma20 -> strict rule3 fails, relaxed passes
    row3["open"] = 101.2
    assert rbi_gbi_signal(row3, 0) is None
    assert rbi_gbi_signal(row3, 0, relaxed=True) == "LONG"
    print("  rbi_gbi_signal check ok (incl. relaxed-rule variant)")

    # --- Power Bar Start ---
    df_pb = pd.DataFrame({
        "open": [100, 99], "high": [100.2, 108], "low": [98.5, 98.8], "close": [98.8, 107],
        "avg_range": [1.0, 1.0],
    })
    sig = power_bar_start_signal(df_pb, 1)
    assert sig == ("LONG", "ELEPHANT"), f"expected elephant-scale long power bar, got {sig}"
    df_pb2 = pd.DataFrame({
        "open": [100, 99], "high": [100.2, 100.5], "low": [98.5, 98.8], "close": [98.8, 100.3],
        "avg_range": [1.0, 1.0],
    })
    sig2 = power_bar_start_signal(df_pb2, 1)
    assert sig2 == ("LONG", "ORDINARY"), f"expected ordinary-scale long power bar, got {sig2}"
    closes_ft = [98.8, 100.3, 101, 102, 103, 104]
    assert follow_through_ok(closes_ft, 1, "LONG", bars_ahead=4) is True
    print("  power_bar_start_signal + follow_through_ok check ok")

    # --- Market Law 4 ---
    lows_ml4 = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 96, 95.5, 100.5]
    highs_ml4 = [x + 1 for x in lows_ml4]
    closes_ml4 = [x + 0.5 for x in lows_ml4]
    closes_ml4[-1] = 100.5
    df_ml4 = pd.DataFrame({"low": lows_ml4, "high": highs_ml4, "close": closes_ml4})
    sig_ml4 = market_law4_signal(df_ml4, 14, lookback=14)
    assert sig_ml4 == "LONG", f"expected failed-breakdown LONG signal, got {sig_ml4}"
    print("  market_law4_signal check ok")

    row_stack = pd.DataFrame({"open": [100], "high": [102], "low": [98], "close": [101],
                               "sma20": [100.9], "sma200": [95], "avg_range": [1.0],
                               "ma_direction": [1]}, index=[0])
    sc = stacking_score(row_stack, 0, "LONG")
    assert sc == 4, f"expected a max stacking score of 4, got {sc}"
    print("  stacking_score check ok")

    # --- sloping 20MA swing trade ---
    row_sw = pd.DataFrame({"open": [100], "close": [100.3], "sma20": [100.2], "avg_range": [0.5],
                            "slope_stage": [2], "ma_direction": [1]}, index=[0])
    assert swing_trade_signal(row_sw, 0) == "LONG"
    row_sw2 = row_sw.copy()
    row_sw2["slope_stage"] = 1
    assert swing_trade_signal(row_sw2, 0) is None
    print("  swing_trade_signal check ok")

    # --- pinch play / hovering ---
    n_h = 15
    df_h = pd.DataFrame({
        "sma20": np.zeros(n_h), "sma200": np.full(n_h, 100.0),
        "avg_range": np.full(n_h, 0.5),
        "high": np.full(n_h, 101.0), "close": np.full(n_h, 100.5),
    })
    sma20_series2 = 100 - np.linspace(3, 0.2, n_h)     # rising (less negative offset) toward sma200=100
    df_h["sma20"] = sma20_series2
    assert is_hovering(df_h, n_h - 1, lookback=10) is True
    breakout_level = df_h["close"].iloc[-11:-1].max() + 2  # breakout above recent range
    df_h.loc[df_h.index[-1], "high"] = breakout_level
    df_h.loc[df_h.index[-1], "close"] = breakout_level
    sig_pp = pinch_play_signal(df_h, n_h - 1, breakout_lookback=10)
    assert sig_pp is not None and sig_pp[0] == "LONG", f"expected a pinch-play LONG breakout, got {sig_pp}"
    print("  is_hovering + pinch_play_signal check ok")

    print("All module self-checks passed.\n")


# ----------------------------- per-day trade generation -----------------------------
# NOTE on timeframe: Velez's material for these six concepts is written for
# daily/swing charts (20-day & 200-day MAs). This project backtests intraday
# Nifty options, so -- consistent with how every other Velez concept in this
# project has been adapted (Crash Play, O=H/O=L, etc.) -- the 20/200 "MA"
# here is computed on the 2-min intraday bar series, NOT the daily chart.
# This is a scope adaptation, not an oversight.
#
# One trade per day per module (matches every other script in this project).
# Lookback windows (avg_range, slope duration, swing-low search, hovering
# check) are computed on the continuous multi-day bar series for MA
# continuity, but may occasionally look a few bars into the previous day
# at/near the 09:15 open -- accepted as a minor, documented approximation
# for this screening pass.

MODULE_WARMUP = SMA_SLOW + AVG_RANGE_LOOKBACK + 5


def run_module(full_df, day_groups, signal_fn, tag_fn=None, min_warmup=MODULE_WARMUP):
    """signal_fn(full_df, abs_pos) -> None, direction_str, or (direction_str, extra_tag).
    tag_fn(full_df, abs_pos, direction) -> optional dict of extra per-trade fields."""
    trades = []
    capital = STARTING_CAPITAL
    for day, positions in day_groups.items():
        if positions[0] < min_warmup:
            continue
        day_df = full_df.loc[full_df.index.date == day]
        day_open_spot = day_df["open"].iloc[0]
        found = None
        for local_i, abs_p in enumerate(positions):
            if abs_p < min_warmup or local_i < 2:
                continue
            res = signal_fn(full_df, abs_p)
            if res is None:
                continue
            if isinstance(res, tuple):
                direction, extra = res
            else:
                direction, extra = res, None
            found = (local_i, abs_p, direction, extra)
            break
        if found is None:
            continue
        local_i, abs_p, direction, extra = found
        side = "CE" if direction == "LONG" else "PE"
        strike = itm_call_strike(day_open_spot) if side == "CE" else itm_put_strike(day_open_spot)
        premium_df = build_premium_series(day_df, strike, day_open_spot, side)
        entry_price = premium_df["close"].iloc[local_i]
        exit_price, reason, exit_time = simulate_trade(premium_df, local_i)
        lot = lot_size_for_date(day)
        pnl_pts = exit_price - entry_price
        rupee_pnl = pnl_pts * lot - BROKERAGE_PER_TRADE
        capital += rupee_pnl
        rec = {
            "date": day, "direction": direction, "strike": strike, "side": side,
            "entry_time": day_df.index[local_i], "exit_time": exit_time,
            "entry_price": round(entry_price, 2), "exit_price": round(exit_price, 2),
            "exit_reason": reason, "pnl_pts": round(pnl_pts, 2),
            "rupee_pnl": round(rupee_pnl, 2), "capital_after": round(capital, 2),
        }
        if extra is not None:
            rec["tag"] = extra
        if tag_fn is not None:
            extra_fields = tag_fn(full_df, abs_p, direction)
            if extra_fields:
                rec.update(extra_fields)
        trades.append(rec)
    return pd.DataFrame(trades)


def compute_metrics(trades_df):
    if trades_df.empty:
        return {"n": 0, "win_pct": 0.0, "pf": 0.0, "net_rs": 0.0, "max_dd_rs": 0.0, "sqn": 0.0}
    pnl = trades_df["rupee_pnl"].to_numpy()
    n = len(pnl)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    win_pct = 100.0 * len(wins) / n
    gross_win = wins.sum()
    gross_loss = abs(losses.sum())
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    net_rs = pnl.sum()
    equity = STARTING_CAPITAL + np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd_rs = dd.max()
    sqn = (pnl.mean() / pnl.std() * math.sqrt(n)) if n > 1 and pnl.std() > 0 else 0.0
    return {"n": n, "win_pct": round(win_pct, 1), "pf": round(pf, 2) if pf != float("inf") else pf,
            "net_rs": round(net_rs, 2), "max_dd_rs": round(max_dd_rs, 2), "sqn": round(sqn, 2)}


def print_diagnostics(name, t, total_days):
    """Sanity-check helper: reports what fraction of eligible days a module
    actually fired on, the CE/PE split, and entry-hour clustering. A module
    that fires on nearly every single day, or whose entries cluster in one
    narrow time window, is a signal that the entry condition is too loose
    to be the specific, selective setup the doc describes -- treat a high
    PF/SQN from such a module with real suspicion, not celebration."""
    if t.empty:
        print(f"  {name}: no trades generated")
        return
    coverage = 100.0 * len(t) / total_days
    ce = int((t["side"] == "CE").sum())
    pe = int((t["side"] == "PE").sum())
    hours = pd.to_datetime(t["entry_time"]).dt.hour
    early = int((hours < 10).sum())
    mid = int(((hours >= 10) & (hours < 12)).sum())
    aft = int(((hours >= 12) & (hours < 14)).sum())
    late = int((hours >= 14).sum())
    print(f"  {name}: fires on {coverage:.1f}% of eligible days ({len(t)}/{total_days}) | "
          f"CE={ce} PE={pe} | entry-hour <10:{early} 10-12:{mid} 12-14:{aft} >=14:{late}")


def main():
    run_self_checks()
    run_module_self_checks()

    print(f"Fetching Nifty index data {FROM_DATE} to {TO_DATE} ...")
    raw = fetch_index_chunked(FROM_DATE, TO_DATE)
    full_df = resample_2min_per_day(raw)
    full_df = compute_indicators(full_df)
    full_df = compute_slope_stage(full_df)
    print(f"Loaded {len(full_df)} bars across {len(set(full_df.index.date))} days.\n")

    day_groups = {}
    for pos, d in enumerate(full_df.index.date):
        day_groups.setdefault(d, []).append(pos)
    total_days = len(day_groups)

    results = {}

    highs_list = full_df["high"].tolist()
    lows_list = full_df["low"].tolist()

    def zone_tag_fn(df, abs_p, direction):
        zone = fifty_pct_zone_check(highs_list, lows_list, abs_p)
        return {"retracement_zone": zone}

    def print_zone_breakdown(t):
        if t.empty or "retracement_zone" not in t.columns:
            return
        for zone, grp in t.groupby("retracement_zone"):
            if zone is None:
                continue
            wp = 100.0 * (grp["rupee_pnl"] > 0).sum() / len(grp)
            print(f"    50%-retracement overlay [{zone}]: n={len(grp)}, win%={wp:.1f}")

    print("Module 1: RBI/GBI (strict, all 5 hard rules)...")
    t = run_module(full_df, day_groups, lambda df, i: rbi_gbi_signal(df, i, relaxed=False), tag_fn=zone_tag_fn)
    results["RBI_GBI_strict"] = (t, compute_metrics(t))
    print_zone_breakdown(t)

    print("Module 1b: RBI/GBI (relaxed -- rule 3 near-20MA dropped)...")
    t = run_module(full_df, day_groups, lambda df, i: rbi_gbi_signal(df, i, relaxed=True), tag_fn=zone_tag_fn)
    results["RBI_GBI_relaxed"] = (t, compute_metrics(t))
    print_zone_breakdown(t)

    print("Module 2: Power Bar Start...")
    t = run_module(full_df, day_groups, power_bar_start_signal)
    results["Power_Bar_Start"] = (t, compute_metrics(t))
    if not t.empty:
        closes_arr = full_df["close"].to_numpy()
        ft_flags = []
        for _, row in t.iterrows():
            abs_pos = full_df.index.get_loc(row["entry_time"])
            fto = follow_through_ok(closes_arr, abs_pos, row["direction"], bars_ahead=5)
            ft_flags.append(fto)
        t["follow_through_5bar"] = ft_flags
        elephant_mask = t["tag"] == "ELEPHANT"
        for label, mask in [("ELEPHANT", elephant_mask), ("ORDINARY", ~elephant_mask)]:
            sub = t[mask]["follow_through_5bar"].dropna()
            if len(sub) > 0:
                pct = 100.0 * sub.sum() / len(sub)
                print(f"    {label} reversal bars: {pct:.1f}% follow-through over next 5 bars (n={len(sub)})")

    print("Module 3: Market Law 4 + stacking score...")
    def ml4_tag_fn(df, abs_p, direction):
        return {"stacking_score": stacking_score(df, abs_p, direction)}
    t = run_module(full_df, day_groups, market_law4_signal, tag_fn=ml4_tag_fn)
    results["Market_Law_4"] = (t, compute_metrics(t))
    if not t.empty:
        print("    win% by stacking score bucket:")
        for score, grp in t.groupby("stacking_score"):
            wp = 100.0 * (grp["rupee_pnl"] > 0).sum() / len(grp)
            print(f"      score={score}: n={len(grp)}, win%={wp:.1f}")

    print("Module 4: Sloping 20MA swing trade...")
    t = run_module(full_df, day_groups, swing_trade_signal, tag_fn=zone_tag_fn)
    results["Swing_Trade_20MA"] = (t, compute_metrics(t))
    print_zone_breakdown(t)

    print("Module 5: Pinch Play / Apex Breakout...")
    t = run_module(full_df, day_groups, pinch_play_signal)
    results["Pinch_Play"] = (t, compute_metrics(t))
    if not t.empty:
        for label, mask in [("OMG_HALT", t["tag"] == "OMG_HALT"), ("STANDARD", t["tag"] == "STANDARD")]:
            sub = t[mask]
            if len(sub) > 0:
                m = compute_metrics(sub)
                print(f"    {label}: n={m['n']}, win%={m['win_pct']}, PF={m['pf']}, net_Rs={m['net_rs']}")

    print("\n" + "=" * 70)
    print("COMPARISON TABLE (screening pass, single 8pt-stop/4pt-trail for all)")
    print("=" * 70)
    print(f"{'Module':<22}{'n':>6}{'win%':>8}{'PF':>8}{'net_Rs':>12}{'maxDD_Rs':>12}{'SQN':>7}")
    for name, (t, m) in results.items():
        pf_str = "inf" if m["pf"] == float("inf") else str(m["pf"])
        print(f"{name:<22}{m['n']:>6}{m['win_pct']:>8}{pf_str:>8}{m['net_rs']:>12}{m['max_dd_rs']:>12}{m['sqn']:>7}")

    print("\n" + "=" * 70)
    print(f"DIAGNOSTICS (sanity-check before trusting any 'great' number -- total eligible days={total_days})")
    print("=" * 70)
    for name, (t, m) in results.items():
        print_diagnostics(name, t, total_days)

    for name, (t, _) in results.items():
        if not t.empty:
            fname = f"six_concepts_{name.lower()}_trades.csv"
            t.to_csv(fname, index=False)
            print(f"Saved {fname}")


if __name__ == "__main__":
    main()
