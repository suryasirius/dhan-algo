"""
Unified Price Action Strategy -- Direct Dhan Trading Bot (Options version)
============================================================================
Standalone Python bot. Connects straight to your Dhan account via the
DhanHQ API. Watches Nifty's own price (the "underlying") and runs the
compression->spike / weak-arc rollover / failed-spike reversal / leg-1-leg-2
confirm / weakening-domes / air-pocket-weighted logic, combined into one
BUY/SELL decision. Every BUY buys the current nearest in-the-money Nifty
CALL, every SELL buys the current nearest in-the-money Nifty PUT, always on
Nifty's current weekly expiry.

EXITS: fixed stop-loss and fixed target, no trailing -- 5 points stop /
10 points target (1:2 risk:reward), measured in Nifty points on
the underlying. (A trailing-stop version was tried and dropped -- on
1-minute close-only backtesting, over half the trail exits gave back more
than the intended 3-point cushion because price gapped past it between
candle closes, so it underperformed the plain fixed target.) When the
stop or target is hit, the bot sells whichever option contract it
actually bought at entry.

TREND FILTER: a 1-minute signal is only allowed to fire if the 5-minute AND
15-minute charts are both trending the same direction as the signal (both up
for a BUY, both down for a SELL). Trend on each timeframe is an EMA(9) vs
EMA(21) read on that timeframe's own candles. If the two timeframes disagree,
or don't have enough bars yet to judge, no trade is taken.
------------------------------------------------------------------------
"""

import io
import time
import logging
import datetime as dt
import requests
import pandas as pd
import numpy as np
from dhanhq import DhanContext, dhanhq

from dhan_config import (
    CLIENT_ID, ACCESS_TOKEN, UNDERLYING_SECURITY_ID, UNDERLYING_EXCHANGE_SEGMENT,
    UNDERLYING_INSTRUMENT_TYPE, ITM_STRIKE_OFFSET, TRADE_QTY,
)

# ============================== CONFIG ===================================
PRODUCT_TYPE = "INTRADAY"
DRY_RUN = True

INSTRUMENT_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"
# NOTE: switched from api-scrip-master.csv (the "compact" file) to the
# "detailed" file above -- found via diagnostic that the compact file
# contains ZERO NIFTY option/futures rows (only SENSEX/BANKEX/crypto-style
# products), which meant resolve_itm_option() below would always raise
# "No NIFTY OPTIDX rows found" and the bot could never place a real trade.
# The detailed file has a clean UNDERLYING_SYMBOL column (3,968 real NIFTY
# rows confirmed) which resolve_itm_option() now filters on directly,
# instead of pattern-matching the trading symbol string.
NIFTY_STRIKE_STEP = 50

# --- Risk / exits: fixed stop + fixed target, no trailing, 1:2 ---
SL_POINTS           = 5
TP_POINTS           = 10
MAX_BARS_IN_TRADE   = 15
CONFLUENCE_REQUIRED = 1

# --- Only concepts 2 and 4 are active -- everything else turned OFF ---
# 2 = Compression spike (sell)
# 4 = Rollover (sell)
# Concept 3 (rollover buy) was dropped -- barely profitable (+4.10 net over
# 32 trades) once trading was already restricted to Tuesday/Wednesday.
# Any signal whose concept_id is in this set is skipped entirely --
# no trade gets taken for it, live or in backtest.
DISABLED_CONCEPTS = {1, 3, 5, 6, 7, 8, 9}

# --- Concept parameters (same defaults as the Pine script) ---
COMP_LEN, COMP_THRESHOLD, MIN_TIGHT_BARS = 10, 0.55, 4
ATR_LEN, SPIKE_MULT, BREAKOUT_LEN, EXHAUSTION_BARS = 14, 1.7, 10, 3
ADX_LEN, WEAK_ADX_THRESH, WEAK_ARC_BARS, NEAR_EXTREME_PCT = 14, 20, 30, 0.5
REJECTION_WICK_RATIO = 0.5
LEG_MOVE_BARS, LEG_MOVE_ATR_MULT = 5, 2.0
DOME_PIVOT_BARS, DOME_BASE_LEN, MIN_DOMES_WEAKENING = 5, 15, 2
BASE_LOOKBACK, AIR_POCKET_ATR_MULT = 200, 4.0

POLL_SECONDS = 20
MARKET_OPEN, MARKET_CLOSE = dt.time(9, 15), dt.time(15, 30)

# --- Higher-timeframe trend filter (5m + 15m must both agree) ---
REQUIRE_HTF_TREND_MATCH = True
TREND_FAST_EMA, TREND_SLOW_EMA = 9, 21
HTF_CONTEXT_DAYS = 5  # extra calendar days of history pulled just so the
                       # 5m/15m EMAs have enough bars from market open onward

# --- Day-of-week filter -- only trade on the days that actually work ---
# Based on the 6-month backtest: Tuesday (+154.45) and Wednesday (+196.90)
# were the only clearly profitable days; Thursday (-186.35) and Friday
# (-59.40) were net losers, Monday was only mildly positive (+49.70).
# Monday=0 ... Sunday=6 (Python's weekday() numbering)
ALLOWED_WEEKDAYS = {1, 2}  # Tuesday, Wednesday only
# ===========================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("dhan_bot.log"), logging.StreamHandler()],
)
log = logging.getLogger("dhan_bot")

if "PASTE_YOUR" in ACCESS_TOKEN or "YOUR_UNDERLYING" in UNDERLYING_SECURITY_ID:
    raise SystemExit(
        "Open dhan_config.py and fill in your real ACCESS_TOKEN and "
        "UNDERLYING_SECURITY_ID before running this."
    )

dhan_context = DhanContext(CLIENT_ID, ACCESS_TOKEN)
dhan = dhanhq(dhan_context)


# ---------------------------- Indicators ----------------------------------
def true_range(df):
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)


def atr(df, length):
    return true_range(df).ewm(alpha=1 / length, adjust=False).mean()


def dmi_adx(df, length):
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_ = true_range(df).ewm(alpha=1 / length, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / length, adjust=False).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / length, adjust=False).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / length, adjust=False).mean()


def pivot_high(series, left, right):
    vals = series.values
    out = [np.nan] * len(vals)
    for i in range(left + right, len(vals)):
        center = i - right
        window = vals[center - left:i + 1]
        if vals[center] == window.max():
            out[center] = vals[center]
    return pd.Series(out, index=series.index)


def pivot_low(series, left, right):
    vals = series.values
    out = [np.nan] * len(vals)
    for i in range(left + right, len(vals)):
        center = i - right
        window = vals[center - left:i + 1]
        if vals[center] == window.min():
            out[center] = vals[center]
    return pd.Series(out, index=series.index)


# ------------------- Higher-timeframe (5m / 15m) trend filter --------------
def resample_ohlc(df, rule):
    """Builds higher-timeframe candles from the 1-min data. label='right'
    means each row is timestamped with the moment that candle CLOSED, so
    later code never reads a bar's info before it actually existed."""
    o = df["open"].resample(rule, label="right", closed="left").first()
    h = df["high"].resample(rule, label="right", closed="left").max()
    l = df["low"].resample(rule, label="right", closed="left").min()
    c = df["close"].resample(rule, label="right", closed="left").last()
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna()


def ema_trend_series(close, fast, slow):
    """UP if the fast EMA is above the slow EMA, DOWN if below, None if
    there aren't enough bars yet to trust the slow EMA."""
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    trend = pd.Series(
        np.where(fast_ema > slow_ema, "UP", np.where(fast_ema < slow_ema, "DOWN", None)),
        index=close.index, dtype=object,
    )
    trend.iloc[:slow] = None
    return trend


def attach_htf_trend(df, fast=TREND_FAST_EMA, slow=TREND_SLOW_EMA):
    """Adds trend_5m / trend_15m columns to a copy of df. Each 1-min row
    gets the trend reading from the most recently CLOSED 5-min/15-min
    candle at that point in time -- no lookahead."""
    df = df.copy()
    for rule, col in (("5min", "trend_5m"), ("15min", "trend_15m")):
        htf_trend = ema_trend_series(resample_ohlc(df, rule)["close"], fast, slow)
        combined_index = df.index.union(htf_trend.index)
        aligned = htf_trend.reindex(combined_index).sort_index().ffill()
        df[col] = aligned.reindex(df.index)
    return df


# ---------------- Bar-by-bar replication of the Pine logic -----------------
def compute_latest_signal(df):
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    n = len(df)
    trend_5m_arr = df["trend_5m"].values if "trend_5m" in df.columns else None
    trend_15m_arr = df["trend_15m"].values if "trend_15m" in df.columns else None

    atr_s = atr(df, ATR_LEN).values
    adx_s = dmi_adx(df, ADX_LEN).values
    piv_low_s = pivot_low(df["low"], 2, 2).values
    piv_high_s = pivot_high(df["high"], 2, 2).values
    piv_dome_s = pivot_high(df["high"], DOME_PIVOT_BARS, DOME_PIVOT_BARS).values

    tight_count = weak_count = weak_dome_count = 0
    bars_since_spike_up = bars_since_spike_down = None
    breakout_level_up = breakout_level_down = None
    anchor_active_long = anchor_active_short = False
    anchor_low_val = leg2_low_val = None
    anchor_high_val = leg2_high_val = None
    last_dome_high = None
    rollover_buy_prev = rollover_sell_prev = False

    result = None

    for i in range(n):
        atr_i = atr_s[i]
        if np.isnan(atr_i) or atr_i == 0:
            continue

        if i >= COMP_LEN - 1:
            short_range = h[i - COMP_LEN + 1:i + 1].max() - l[i - COMP_LEN + 1:i + 1].min()
            is_tight = (short_range / (atr_i * COMP_LEN)) < COMP_THRESHOLD
        else:
            is_tight = False
        tight_count = tight_count + 1 if is_tight else 0
        compression_built = tight_count >= MIN_TIGHT_BARS

        big_range_bar = (h[i] - l[i]) > SPIKE_MULT * atr_i
        if i >= BREAKOUT_LEN:
            prior_high = h[max(0, i - BREAKOUT_LEN):i].max()
            prior_low = l[max(0, i - BREAKOUT_LEN):i].min()
        else:
            prior_high = prior_low = np.nan

        spike_up = big_range_bar and not np.isnan(prior_high) and c[i] > prior_high and c[i] > o[i]
        spike_down = big_range_bar and not np.isnan(prior_low) and c[i] < prior_low and c[i] < o[i]
        raw_buy = compression_built and spike_up
        raw_sell = compression_built and spike_down

        if raw_buy:
            breakout_level_up, bars_since_spike_up = prior_high, 0
        elif bars_since_spike_up is not None:
            bars_since_spike_up += 1
        if raw_sell:
            breakout_level_down, bars_since_spike_down = prior_low, 0
        elif bars_since_spike_down is not None:
            bars_since_spike_down += 1

        exhausted_up = (bars_since_spike_up is not None and bars_since_spike_up <= EXHAUSTION_BARS
                         and breakout_level_up is not None and c[i] < breakout_level_up)
        exhausted_down = (bars_since_spike_down is not None and bars_since_spike_down <= EXHAUSTION_BARS
                           and breakout_level_down is not None and c[i] > breakout_level_down)

        vote_spike_buy = raw_buy and not exhausted_down
        vote_spike_sell = raw_sell and not exhausted_up
        vote_reversal_buy = exhausted_down
        vote_reversal_sell = exhausted_up

        adx_i = adx_s[i]
        weak_trend = (not np.isnan(adx_i)) and adx_i < WEAK_ADX_THRESH
        weak_count = weak_count + 1 if weak_trend else 0
        stretched_weak_arc = weak_count >= WEAK_ARC_BARS

        if i >= WEAK_ARC_BARS - 1:
            extreme_high = h[i - WEAK_ARC_BARS + 1:i + 1].max()
            extreme_low = l[i - WEAK_ARC_BARS + 1:i + 1].min()
        else:
            extreme_high = extreme_low = np.nan
        near_high = (not np.isnan(extreme_high)) and h[i] >= extreme_high * (1 - NEAR_EXTREME_PCT / 100)
        near_low = (not np.isnan(extreme_low)) and l[i] <= extreme_low * (1 + NEAR_EXTREME_PCT / 100)

        rng = h[i] - l[i]
        bull_wick = min(o[i], c[i]) - l[i]
        bear_wick = h[i] - max(o[i], c[i])
        fast_rej_up = rng > 0 and (bull_wick / rng) >= REJECTION_WICK_RATIO and c[i] > o[i]
        fast_rej_down = rng > 0 and (bear_wick / rng) >= REJECTION_WICK_RATIO and c[i] < o[i]

        rollover_buy_raw = stretched_weak_arc and near_low and fast_rej_up
        rollover_sell_raw = stretched_weak_arc and near_high and fast_rej_down
        vote_rollover_buy = rollover_buy_raw and not rollover_buy_prev
        vote_rollover_sell = rollover_sell_raw and not rollover_sell_prev
        rollover_buy_prev, rollover_sell_prev = rollover_buy_raw, rollover_sell_raw

        if i >= LEG_MOVE_BARS:
            hh = h[i - LEG_MOVE_BARS + 1:i + 1].max()
            ll = l[i - LEG_MOVE_BARS + 1:i + 1].min()
            sharp_down_leg = (hh - l[i]) > LEG_MOVE_ATR_MULT * atr_i and l[i] <= ll
            sharp_up_leg = (h[i] - ll) > LEG_MOVE_ATR_MULT * atr_i and h[i] >= hh
        else:
            sharp_down_leg = sharp_up_leg = False

        piv_low_i = piv_low_s[i]
        piv_high_i = piv_high_s[i]

        if sharp_down_leg:
            anchor_active_long, anchor_low_val, leg2_low_val = True, None, None
        if anchor_active_long and not np.isnan(piv_low_i):
            if anchor_low_val is None:
                anchor_low_val = piv_low_i
            elif leg2_low_val is None and piv_low_i > anchor_low_val:
                leg2_low_val = piv_low_i
        vote_leg2_buy = (anchor_active_long and leg2_low_val is not None
                          and not np.isnan(piv_high_i) and c[i] > piv_high_i and c[i] > o[i])
        if vote_leg2_buy:
            anchor_active_long = False

        if sharp_up_leg:
            anchor_active_short, anchor_high_val, leg2_high_val = True, None, None
        if anchor_active_short and not np.isnan(piv_high_i):
            if anchor_high_val is None:
                anchor_high_val = piv_high_i
            elif leg2_high_val is None and piv_high_i < anchor_high_val:
                leg2_high_val = piv_high_i
        vote_leg2_sell = (anchor_active_short and leg2_high_val is not None
                           and not np.isnan(piv_low_i) and c[i] < piv_low_i and c[i] < o[i])
        if vote_leg2_sell:
            anchor_active_short = False

        piv_dome_i = piv_dome_s[i]
        if not np.isnan(piv_dome_i):
            weak_dome_count = weak_dome_count + 1 if (last_dome_high is not None and piv_dome_i < last_dome_high) else 0
            last_dome_high = piv_dome_i
        weakening_domes = weak_dome_count >= MIN_DOMES_WEAKENING
        recent_low = l[max(0, i - DOME_BASE_LEN):i].min() if i >= DOME_BASE_LEN else np.nan
        vote_dome_sell = weakening_domes and (not np.isnan(recent_low)) and c[i] < recent_low

        base_start = max(0, i - BASE_LOOKBACK + 1)
        long_term_base_low = l[base_start:i + 1].min()
        air_pocket_points = h[i] - long_term_base_low
        large_air_pocket = air_pocket_points > AIR_POCKET_ATR_MULT * atr_i
        vote_air_pocket_boost = (vote_rollover_sell or vote_dome_sell) and large_air_pocket

        bull_votes = int(vote_spike_buy) + int(vote_rollover_buy) + int(vote_reversal_buy) + int(vote_leg2_buy)
        bear_votes = (int(vote_spike_sell) + int(vote_rollover_sell) + int(vote_reversal_sell)
                      + int(vote_leg2_sell) + int(vote_dome_sell) + int(vote_air_pocket_boost))

        final_buy = bull_votes >= CONFLUENCE_REQUIRED and bull_votes > bear_votes
        final_sell = bear_votes >= CONFLUENCE_REQUIRED and bear_votes > bull_votes

        # which concept is responsible, by priority (matches the 1-9 key from
        # the Pine script): 1 spike-buy, 2 spike-sell, 3 rollover-buy,
        # 4 rollover-sell, 5 reversal-buy, 6 reversal-sell, 7 leg2-buy,
        # 8 leg2-sell, 9 dome-sell
        concept_id = None
        if final_buy:
            concept_id = 1 if vote_spike_buy else 3 if vote_rollover_buy else 5 if vote_reversal_buy else 7
        elif final_sell:
            concept_id = 2 if vote_spike_sell else 4 if vote_rollover_sell else 6 if vote_reversal_sell else 9 if vote_dome_sell else 8

        if concept_id in DISABLED_CONCEPTS:
            final_buy, final_sell, concept_id = False, False, None

        trend_5m_i = trend_5m_arr[i] if trend_5m_arr is not None else None
        trend_15m_i = trend_15m_arr[i] if trend_15m_arr is not None else None
        if REQUIRE_HTF_TREND_MATCH and (final_buy or final_sell):
            htf_agrees = (
                (final_buy and trend_5m_i == "UP" and trend_15m_i == "UP") or
                (final_sell and trend_5m_i == "DOWN" and trend_15m_i == "DOWN")
            )
            if not htf_agrees:
                final_buy, final_sell, concept_id = False, False, None

        if (final_buy or final_sell) and df.index[i].weekday() not in ALLOWED_WEEKDAYS:
            final_buy, final_sell, concept_id = False, False, None

        result = dict(time=df.index[i], close=c[i], bull_votes=bull_votes,
                      bear_votes=bear_votes, final_buy=final_buy, final_sell=final_sell,
                      concept_id=concept_id, trend_5m=trend_5m_i, trend_15m=trend_15m_i)

    return result


# ---------------------------- Dhan data / orders ----------------------------
def get_recent_candles():
    """Pulls today's candles PLUS a few extra days of history (HTF_CONTEXT_DAYS)
    so the 5m/15m trend EMAs already have enough bars to judge a trend from
    the moment the market opens today, instead of only warming up late in
    the day. Returns the whole window with trend_5m/trend_15m attached."""
    today = dt.datetime.now().strftime("%Y-%m-%d")
    from_date = (dt.datetime.now() - dt.timedelta(days=HTF_CONTEXT_DAYS)).strftime("%Y-%m-%d")
    resp = dhan.intraday_minute_data(
        security_id=UNDERLYING_SECURITY_ID,
        exchange_segment=UNDERLYING_EXCHANGE_SEGMENT,
        instrument_type=UNDERLYING_INSTRUMENT_TYPE,
        from_date=from_date,
        to_date=today,
    )
    data = resp.get("data", resp)
    df = pd.DataFrame({
        "open": data["open"], "high": data["high"],
        "low": data["low"], "close": data["close"],
        "timestamp": data["timestamp"],
    })
    # Dhan's timestamp is plain UTC seconds -- add 5:30 to get real IST
    # market-clock time (otherwise every candle looks 5.5 hours earlier
    # than it actually happened, e.g. a 2:45pm IST trade shows as 9:15am).
    df["time"] = pd.to_datetime(df["timestamp"], unit="s") + pd.Timedelta(hours=5, minutes=30)
    df = df.set_index("time").sort_index()
    return attach_htf_trend(df)


_instrument_cache = {"df": None, "loaded_date": None}


def load_instrument_master():
    today = dt.date.today()
    if _instrument_cache["df"] is not None and _instrument_cache["loaded_date"] == today:
        return _instrument_cache["df"]
    log.info("Downloading Dhan instrument master (once per day)...")
    resp = requests.get(INSTRUMENT_MASTER_URL, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    _instrument_cache["df"] = df
    _instrument_cache["loaded_date"] = today
    return df


def resolve_itm_option(option_type, spot_price):
    df = load_instrument_master()

    # Detailed-file columns: UNDERLYING_SYMBOL identifies NIFTY unambiguously
    # (no need to pattern-match a trading-symbol string), INSTRUMENT_TYPE
    # "OP" = options (index or stock), OPTION_TYPE is "CE"/"PE" same as before.
    nifty_opts = df[
        (df["UNDERLYING_SYMBOL"].astype(str).str.upper() == "NIFTY")
        & (df["EXCH_ID"].astype(str).str.upper() == "NSE")
        & (df["INSTRUMENT_TYPE"].astype(str).str.upper() == "OP")
        & (df["OPTION_TYPE"] == option_type)
    ].copy()
    if nifty_opts.empty:
        raise ValueError(
            "No NIFTY OPTIDX rows found with these column filters -- check "
            "the instrument master's column names/format."
        )

    nifty_opts["SM_EXPIRY_DATE"] = pd.to_datetime(nifty_opts["SM_EXPIRY_DATE"])
    today = pd.Timestamp(dt.date.today())
    nifty_opts = nifty_opts[nifty_opts["SM_EXPIRY_DATE"] >= today]
    nearest_expiry = nifty_opts["SM_EXPIRY_DATE"].min()
    nifty_opts = nifty_opts[nifty_opts["SM_EXPIRY_DATE"] == nearest_expiry]

    strikes = sorted(nifty_opts["STRIKE_PRICE"].unique())
    if option_type == "CE":
        itm_strikes = sorted([s for s in strikes if s <= spot_price], reverse=True)
    else:
        itm_strikes = sorted([s for s in strikes if s >= spot_price])

    if len(itm_strikes) <= ITM_STRIKE_OFFSET:
        raise ValueError("Not enough ITM strikes available -- check spot_price/data.")
    target_strike = itm_strikes[ITM_STRIKE_OFFSET]

    row = nifty_opts[nifty_opts["STRIKE_PRICE"] == target_strike].iloc[0]
    return {
        "security_id": str(row["SECURITY_ID"]),
        "exchange_segment": "NSE_FNO",
        "trading_symbol": row["DISPLAY_NAME"],
        "strike": target_strike,
        "expiry": nearest_expiry.date(),
    }


def place_option_order(transaction_type, contract):
    if DRY_RUN:
        log.info("[DRY RUN] Would %s %s qty=%s (strike %s, expiry %s)",
                  transaction_type, contract["trading_symbol"], TRADE_QTY,
                  contract["strike"], contract["expiry"])
        return "DRYRUN"
    order_id = dhan.place_order(
        security_id=contract["security_id"],
        exchange_segment=contract["exchange_segment"],
        transaction_type=transaction_type,
        quantity=TRADE_QTY,
        order_type="MARKET",
        product_type=PRODUCT_TYPE,
    )
    log.info("Placed %s on %s -> %s", transaction_type, contract["trading_symbol"], order_id)
    return order_id


# ---------------------------- Position management ----------------------------
# Fixed stop / fixed target, no trailing. Judged on the UNDERLYING's price
# movement; the order placed to close is the OPTION contract bought at entry.
position = {"side": None, "entry_underlying_price": None, "bars_held": 0, "contract": None}


def manage_open_position(latest_underlying_close):
    if position["side"] is None:
        return
    exited = False
    if position["side"] == "LONG":
        hard_stop = position["entry_underlying_price"] - SL_POINTS
        target = position["entry_underlying_price"] + TP_POINTS
        if latest_underlying_close <= hard_stop:
            log.info("STOP hit (underlying @ %.2f)", latest_underlying_close)
            place_option_order("SELL", position["contract"]); exited = True
        elif latest_underlying_close >= target:
            log.info("TARGET hit (underlying @ %.2f)", latest_underlying_close)
            place_option_order("SELL", position["contract"]); exited = True
        elif position["bars_held"] >= MAX_BARS_IN_TRADE:
            log.info("TIME EXIT (underlying @ %.2f)", latest_underlying_close)
            place_option_order("SELL", position["contract"]); exited = True
    elif position["side"] == "SHORT":
        hard_stop = position["entry_underlying_price"] + SL_POINTS
        target = position["entry_underlying_price"] - TP_POINTS
        if latest_underlying_close >= hard_stop:
            log.info("STOP hit (underlying @ %.2f)", latest_underlying_close)
            place_option_order("SELL", position["contract"]); exited = True
        elif latest_underlying_close <= target:
            log.info("TARGET hit (underlying @ %.2f)", latest_underlying_close)
            place_option_order("SELL", position["contract"]); exited = True
        elif position["bars_held"] >= MAX_BARS_IN_TRADE:
            log.info("TIME EXIT (underlying @ %.2f)", latest_underlying_close)
            place_option_order("SELL", position["contract"]); exited = True

    if exited:
        position.update(side=None, entry_underlying_price=None, bars_held=0, contract=None)
    else:
        position["bars_held"] += 1


def is_market_open():
    now = dt.datetime.now().time()
    return MARKET_OPEN <= now <= MARKET_CLOSE


def main():
    log.info("Starting Dhan live bot. DRY_RUN=%s, TRADE_QTY=%s", DRY_RUN, TRADE_QTY)
    last_candle_time = None
    while True:
        if not is_market_open():
            log.info("Market closed, sleeping 60s...")
            time.sleep(60)
            continue
        try:
            df = get_recent_candles()
        except Exception:
            log.exception("Failed to fetch candles, retrying...")
            time.sleep(POLL_SECONDS)
            continue

        if df.empty:
            time.sleep(POLL_SECONDS)
            continue

        latest_time = df.index[-1]
        if latest_time == last_candle_time:
            time.sleep(POLL_SECONDS)
            continue
        last_candle_time = latest_time

        # Only today's rows go into the concept logic (so compression/weak-arc
        # counters reset fresh each day, same as the backtest) -- but they
        # still carry trend_5m/trend_15m columns inherited from the wider
        # multi-day window fetched above, so the trend filter has real data
        # to work with even in the first few minutes after market open.
        today_df = df[df.index.date == dt.date.today()]
        if today_df.empty:
            time.sleep(POLL_SECONDS)
            continue

        sig = compute_latest_signal(today_df)
        if sig is None:
            time.sleep(POLL_SECONDS)
            continue

        latest_close = df["close"].iloc[-1]
        log.info("Candle %s close=%.2f bull=%d bear=%d", latest_time, latest_close,
                  sig["bull_votes"], sig["bear_votes"])

        if position["side"] is not None:
            manage_open_position(latest_close)

        if position["side"] is None:
            if sig["final_buy"]:
                try:
                    contract = resolve_itm_option("CE", latest_close)
                    place_option_order("BUY", contract)
                    position.update(side="LONG", entry_underlying_price=latest_close,
                                     bars_held=0, contract=contract)
                except Exception:
                    log.exception("Could not resolve/buy ITM Call, skipping this signal")
            elif sig["final_sell"]:
                try:
                    contract = resolve_itm_option("PE", latest_close)
                    place_option_order("BUY", contract)
                    position.update(side="SHORT", entry_underlying_price=latest_close,
                                     bars_held=0, contract=contract)
                except Exception:
                    log.exception("Could not resolve/buy ITM Put, skipping this signal")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
