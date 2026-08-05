"""
Master OI-Gated Combined Backtest -- ALL concepts ever tried in this project
================================================================================
Standalone -- does NOT modify dhan_live_bot.py / backtest.py / FINAL_*.py.

WHAT THIS IS
------------
Every concept tried in this project, re-tested with the same NIFTY-futures
OI confirmation filter built for the standalone OI concept (LONG only
allowed when OI shows LONG_BUILDUP, SHORT only when SHORT_BUILDUP -- exact
same classify_oi_regime() logic, unit-tested, used in the last two scripts).

FIDELITY WARNING -- read before trusting results
-------------------------------------------------
Concepts 2&4, Setup 2 (v1, session-filtered), Scalping Setup 1 (Trend
Pullback), Scalping Setup 3 (Breakout & Retest), and Order Flow/AMT are
EXACT reuse of their already-tested code from earlier in this same
conversation.
SMC, Price Action 3-Step, Setup 2 v2 (zone-refined), and Strike Momentum
were built in an earlier part of this conversation that has since been
summarized -- their exact original code is no longer available. What's
below is a faithful RECONSTRUCTION from the detailed parameter/logic
description in that summary, not a byte-identical rerun. Treat these four
results as a reasonable approximation, not a confirmed re-validation.
Strike Momentum in particular only had its risk parameters documented, not
its exact entry trigger rule -- the version here is a labeled best-effort
guess and should be trusted least of all nine.

DATA WINDOW: 1 month, per your instruction. (NIFTY futures OI itself is
available further back too -- confirmed ~90 days earlier -- but everything
here uses the same 1-month window you asked for, for consistency.)

STRUCTURE (to keep this from becoming unreadable/unrunnable):
  1. Each of the 9 concepts is shown individually, OI-gated, under ONE
     baseline management style (Fixed 1:2 -- this project's established
     default) so you can see each concept's own standalone contribution.
  2. ALL NINE combined into one pooled portfolio (shared capital, like the
     earlier "combined_final_strategy" scripts) is then tested across ALL
     8 risk-manager variants from your diagram (Fixed 1:1, 1:1.5, 1:2, 1:3,
     1:5, Trail-after-1:2, Trail-after-1:5, Full-trailing-from-start).
  3. The FULL analytics suite from your diagram (daily, 30-min session,
     CE/PE, MFE/MAE, win/loss stats, drawdown, expectancy, equity curve,
     weekly/monthly, volatility, gap, trend, walk-forward-style,
     Monte Carlo, full trade log) is run on the pooled portfolio's
     baseline Fixed 1:2 variant, plus a condensed comparison table across
     all 8 variants.

HONEST SCOPE NOTES on two specific analytics items:
  - "Walk-Forward Validation": this project doesn't fit/optimize
    parameters (every concept's rules are fixed, decided by hand), so
    classic walk-forward (re-optimize on train, verify on test) doesn't
    apply as-is. What's implemented instead is a walk-forward-STYLE
    consistency check: the window is split into sequential weekly folds
    and each fold's performance is shown independently, extending the
    project's existing H1/H2 stability check to more/smaller folds.
  - "Monte Carlo Simulation": there's no probabilistic trade-generating
    model here, so this reshuffles the ACTUAL sequence of real trade
    outcomes into many random orders (1000 iterations) to see how much of
    the equity curve's shape (final result, max drawdown) depends on the
    lucky/unlucky ORDER trades happened to occur in, versus the trades
    themselves. This is a legitimate, standard use of the technique for
    a fixed set of historical trade outcomes.
"""

import datetime as dt
import random
import pandas as pd
import numpy as np

import dhan_live_bot as dlb

REQUEST_DAYS_BACK = 35  # 1 month + a few days buffer
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
FROM_DATE = (dt.date.today() - dt.timedelta(days=REQUEST_DAYS_BACK)).strftime("%Y-%m-%d")

OI_CHANGE_LOOKBACK_MIN = 15
AGGR_MIN_WARMUP_BARS = 30
TRAIL_PTS = 3
SL_POINTS = dlb.SL_POINTS  # 5 -- shared risk unit across every concept in this file

RISK_VARIANTS = [
    ("Fixed_1_1", SL_POINTS * 1, False, False),
    ("Fixed_1_1_5", SL_POINTS * 1.5, False, False),
    ("Fixed_1_2", SL_POINTS * 2, False, False),
    ("Fixed_1_3", SL_POINTS * 3, False, False),
    ("Fixed_1_5", SL_POINTS * 5, False, False),
    ("Trail_after_1_2", SL_POINTS * 2, True, False),
    ("Trail_after_1_5", SL_POINTS * 5, True, False),
    ("Full_Trailing_From_Start", None, False, True),
]
BASELINE_VARIANT = "Fixed_1_2"

STARTING_CAPITAL = 10_000
LOT_SIZE = 65
ASSUMED_DELTA = 0.6
BROKERAGE_PER_TRADE = 40.0
MONTE_CARLO_ITERATIONS = 1000

SESSIONS = [
    "09:15-09:45", "09:45-10:15", "10:15-10:45", "10:45-11:15",
    "11:15-11:45", "11:45-12:15", "12:15-12:45", "12:45-13:15",
    "13:15-13:45", "13:45-14:15", "14:15-14:45", "14:45-15:15", "15:15-15:30",
]
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


# ============================== DATA FETCH ==================================

def fetch_index_chunked(from_date, to_date, chunk_days=85):
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


def find_current_nifty_future():
    df = dlb.load_instrument_master()
    futs = df[
        (df["UNDERLYING_SYMBOL"].astype(str).str.upper() == "NIFTY")
        & (df["EXCH_ID"].astype(str).str.upper() == "NSE")
        & (df["INSTRUMENT_TYPE"].astype(str).str.upper() == "FUT")
    ].copy()
    if futs.empty:
        raise SystemExit("No NIFTY futures contract found.")
    futs["SM_EXPIRY_DATE"] = pd.to_datetime(futs["SM_EXPIRY_DATE"])
    row = futs.sort_values("SM_EXPIRY_DATE").iloc[0]
    return row["SECURITY_ID"], row["DISPLAY_NAME"], row["SM_EXPIRY_DATE"].date()


def fetch_futures_oi_chunked(security_id, from_date, to_date, chunk_days=85):
    start = dt.datetime.strptime(from_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(to_date, "%Y-%m-%d").date()
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + dt.timedelta(days=chunk_days - 1), end)
        resp = dlb.dhan.intraday_minute_data(
            security_id=str(security_id), exchange_segment="NSE_FNO", instrument_type="FUTIDX",
            from_date=cur.strftime("%Y-%m-%d"), to_date=chunk_end.strftime("%Y-%m-%d"), oi=True,
        )
        if isinstance(resp, dict) and resp.get("status") == "failure":
            print(f"  [{cur} to {chunk_end}] skipped: {resp.get('remarks', resp)}")
            cur = chunk_end + dt.timedelta(days=1)
            continue
        data = resp.get("data", resp)
        if isinstance(data, dict) and "open_interest" in data and len(data.get("timestamp", [])) > 0:
            d = pd.DataFrame({"oi": data["open_interest"], "timestamp": data["timestamp"]})
            d["time"] = pd.to_datetime(d["timestamp"], unit="s") + pd.Timedelta(hours=5, minutes=30)
            chunks.append(d.set_index("time").sort_index())
            print(f"  [{cur} to {chunk_end}] got {len(d)} candles.")
        else:
            print(f"  [{cur} to {chunk_end}] no OI in response.")
        cur = chunk_end + dt.timedelta(days=1)
    if not chunks:
        return pd.DataFrame(columns=["oi"])
    full = pd.concat(chunks)
    return full[~full.index.duplicated(keep="first")].sort_index()


def classify_oi_regime(price_change, oi_change):
    if price_change > 0 and oi_change > 0:
        return "LONG_BUILDUP"
    if price_change < 0 and oi_change > 0:
        return "SHORT_BUILDUP"
    if price_change > 0 and oi_change < 0:
        return "SHORT_COVERING"
    if price_change < 0 and oi_change < 0:
        return "LONG_UNWINDING"
    return "NEUTRAL"


def compute_atr(highs, lows, closes, period=14):
    trs = [highs[0] - lows[0]]
    for i in range(1, len(highs)):
        trs.append(max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])))
    atr = []
    for i in range(len(trs)):
        w = trs[max(0, i-period+1):i+1]
        atr.append(sum(w)/len(w))
    return atr


def find_confirmed_swings(highs, lows, n=3):
    length = len(highs)
    swing_high, swing_low = {}, {}
    for i in range(n, length-n):
        wh = highs[i-n:i+n+1]
        if highs[i] == max(wh):
            swing_high[i] = highs[i]
        wl = lows[i-n:i+n+1]
        if lows[i] == min(wl):
            swing_low[i] = lows[i]
    return swing_high, swing_low


def is_bullish_engulfing(o, c, j):
    if j < 1:
        return False
    return (c[j] > o[j] and c[j-1] < o[j-1] and min(o[j], c[j]) <= min(o[j-1], c[j-1])
            and max(o[j], c[j]) >= max(o[j-1], c[j-1]))


def is_bearish_engulfing(o, c, j):
    if j < 1:
        return False
    return (c[j] < o[j] and c[j-1] > o[j-1] and min(o[j], c[j]) <= min(o[j-1], c[j-1])
            and max(o[j], c[j]) >= max(o[j-1], c[j-1]))


def is_bullish_pinbar(o, h, l, c, j, ratio=2.0):
    body = abs(c[j]-o[j])
    lw = min(o[j], c[j]) - l[j]
    uw = h[j] - max(o[j], c[j])
    return body > 0 and lw >= ratio*body and lw > uw


def is_bearish_pinbar(o, h, l, c, j, ratio=2.0):
    body = abs(c[j]-o[j])
    lw = min(o[j], c[j]) - l[j]
    uw = h[j] - max(o[j], c[j])
    return body > 0 and uw >= ratio*body and uw > lw


def run_self_checks():
    print("Running self-checks (synthetic data)...")
    assert classify_oi_regime(10, 5000) == "LONG_BUILDUP"
    assert classify_oi_regime(-10, 5000) == "SHORT_BUILDUP"
    assert classify_oi_regime(10, -5000) == "SHORT_COVERING"
    assert classify_oi_regime(-10, -5000) == "LONG_UNWINDING"
    o = [100, 105]; c = [105, 100]
    assert is_bearish_engulfing(o, c, 1) is True
    print("All self-checks passed.\n")


# ============================== CONCEPT DETECTORS ============================
# Each returns a list of {"bar": i, "side": "LONG"/"SHORT"} for one day_df.

def detect_concepts_2_4(day_df):
    """EXACT reuse -- dlb.compute_latest_signal(), expanding window per bar,
    same pattern already used and proven in this session's OI-gated test."""
    signals = []
    n = len(day_df)
    for i in range(AGGR_MIN_WARMUP_BARS, n):
        sig = dlb.compute_latest_signal(day_df.iloc[:i+1])
        if sig is None:
            continue
        if sig["final_buy"]:
            signals.append({"bar": i, "side": "LONG"})
        elif sig["final_sell"]:
            signals.append({"bar": i, "side": "SHORT"})
    return signals


def detect_setup2_v1(day_df, min_sweep_pts=1.0):
    """EXACT reuse -- Reversal/Liquidity Sweep v1, from scalping_3setups_backtest.py
    built earlier this session."""
    o, h, l, c = day_df["open"].tolist(), day_df["high"].tolist(), day_df["low"].tolist(), day_df["close"].tolist()
    n = len(day_df)
    sh, sl = find_confirmed_swings(h, l, 3)
    ssh, ssl = sorted(sh.items()), sorted(sl.items())
    shp, slp = 0, 0
    last_sh, last_sl = None, None
    signals = []
    pend_bull, pend_bear = None, None
    for j in range(n):
        while shp < len(ssh) and ssh[shp][0] + 3 <= j:
            last_sh = ssh[shp][1]; shp += 1
        while slp < len(ssl) and ssl[slp][0] + 3 <= j:
            last_sl = ssl[slp][1]; slp += 1
        if pend_bull is not None:
            if c[j] > pend_bull:
                signals.append({"bar": j, "side": "LONG"})
            pend_bull = None
        if pend_bear is not None:
            if c[j] < pend_bear:
                signals.append({"bar": j, "side": "SHORT"})
            pend_bear = None
        if last_sl is not None and l[j] < last_sl - min_sweep_pts and c[j] > last_sl:
            pend_bull = c[j]
        if last_sh is not None and h[j] > last_sh + min_sweep_pts and c[j] < last_sh:
            pend_bear = c[j]
    return signals


def detect_scalping_setup1_trend_pullback(day_df):
    """EXACT reuse -- Trend Pullback Continuation, from scalping_3setups_backtest.py."""
    o, h, l, c = day_df["open"].tolist(), day_df["high"].tolist(), day_df["low"].tolist(), day_df["close"].tolist()
    trend_5m = day_df["trend_5m"].tolist() if "trend_5m" in day_df.columns else ["NONE"]*len(day_df)
    trend_15m = day_df["trend_15m"].tolist() if "trend_15m" in day_df.columns else ["NONE"]*len(day_df)
    n = len(day_df)
    sh, sl = find_confirmed_swings(h, l, 3)
    ssh, ssl = sorted(sh.items()), sorted(sl.items())
    shp, slp = 0, 0
    conf_h, conf_l = [], []
    signals = []
    up_leg = down_leg = None
    up_armed = down_armed = False
    up_pb = down_pb = None
    for j in range(n):
        while shp < len(ssh) and ssh[shp][0] + 3 <= j:
            conf_h.append(ssh[shp][1])
            if conf_l:
                up_leg = {"low": conf_l[-1], "high": ssh[shp][1]}; up_armed = True; up_pb = None
            shp += 1
        while slp < len(ssl) and ssl[slp][0] + 3 <= j:
            conf_l.append(ssl[slp][1])
            if conf_h:
                down_leg = {"high": conf_h[-1], "low": ssl[slp][1]}; down_armed = True; down_pb = None
            slp += 1
        ok_up = trend_5m[j] == "UP" and trend_15m[j] == "UP"
        ok_down = trend_5m[j] == "DOWN" and trend_15m[j] == "DOWN"
        if up_armed and up_leg and ok_up:
            rng = up_leg["high"] - up_leg["low"]
            if rng > 0:
                if l[j] < up_leg["low"]:
                    up_armed = False
                else:
                    if up_pb is None or l[j] < up_pb:
                        up_pb = l[j]
                    retrace = (up_leg["high"] - up_pb) / rng
                    if retrace <= 0.5 and c[j] < up_leg["high"]:
                        if is_bullish_engulfing(o, c, j) or is_bullish_pinbar(o, h, l, c, j):
                            signals.append({"bar": j, "side": "LONG"}); up_armed = False
        if down_armed and down_leg and ok_down:
            rng = down_leg["high"] - down_leg["low"]
            if rng > 0:
                if h[j] > down_leg["high"]:
                    down_armed = False
                else:
                    if down_pb is None or h[j] > down_pb:
                        down_pb = h[j]
                    retrace = (down_pb - down_leg["low"]) / rng
                    if retrace <= 0.5 and c[j] > down_leg["low"]:
                        if is_bearish_engulfing(o, c, j) or is_bearish_pinbar(o, h, l, c, j):
                            signals.append({"bar": j, "side": "SHORT"}); down_armed = False
    return signals


def detect_scalping_setup3_breakout_retest(day_df):
    """EXACT reuse -- Breakout and Retest, from scalping_3setups_backtest.py."""
    o, h, l, c = day_df["open"].tolist(), day_df["high"].tolist(), day_df["low"].tolist(), day_df["close"].tolist()
    n = len(day_df)
    atr = compute_atr(h, l, c, 14)
    signals = []
    pending = None
    for j in range(n):
        if j >= 8:
            wh, wl = h[j-8:j], l[j-8:j]
            ch_, cl_ = max(wh), min(wl)
            in_cons = (ch_ - cl_) <= 1.0 * atr[j]
            body = abs(c[j]-o[j])
            if in_cons and pending is None:
                if c[j] > ch_ and body > 0.3*atr[j]:
                    pending = {"level": ch_, "dir": "up", "bar": j}
                elif c[j] < cl_ and body > 0.3*atr[j]:
                    pending = {"level": cl_, "dir": "down", "bar": j}
        if pending is not None:
            if j - pending["bar"] > 20:
                pending = None
            else:
                if pending["dir"] == "up":
                    if l[j] <= pending["level"] and c[j] > pending["level"]:
                        signals.append({"bar": j, "side": "LONG"}); pending = None
                else:
                    if h[j] >= pending["level"] and c[j] < pending["level"]:
                        signals.append({"bar": j, "side": "SHORT"}); pending = None
    return signals


def detect_smc_reconstructed(day_df):
    """RECONSTRUCTED FROM SUMMARY (not exact original code) -- simplified SMC:
    break of a confirmed swing (BOS) in a direction, followed by a liquidity
    sweep back through the opposite recent swing that reverses within
    SWEEP_LOOKBACK bars, treated as a CHoCH-style reversal entry. Much
    simpler than the original OB/FVG premium/discount zone logic (which
    isn't reconstructable in full fidelity from the summary alone) -- this
    captures the core BOS+sweep idea, not the full original mechanism."""
    o, h, l, c = day_df["open"].tolist(), day_df["high"].tolist(), day_df["low"].tolist(), day_df["close"].tolist()
    n = len(day_df)
    sh, sl = find_confirmed_swings(h, l, 3)
    ssh, ssl = sorted(sh.items()), sorted(sl.items())
    shp, slp = 0, 0
    last_sh, last_sl = None, None
    signals = []
    for j in range(n):
        while shp < len(ssh) and ssh[shp][0] + 3 <= j:
            last_sh = ssh[shp][1]; shp += 1
        while slp < len(ssl) and ssl[slp][0] + 3 <= j:
            last_sl = ssl[slp][1]; slp += 1
        if last_sl is not None and l[j] < last_sl and c[j] > last_sl and c[j] > o[j]:
            signals.append({"bar": j, "side": "LONG"})
        elif last_sh is not None and h[j] > last_sh and c[j] < last_sh and c[j] < o[j]:
            signals.append({"bar": j, "side": "SHORT"})
    return signals


def detect_price_action_3step_reconstructed(day_df):
    """RECONSTRUCTED FROM SUMMARY -- zone (ATR-width, 2+ touches) + engulfing/
    pinbar trigger, matching the documented parameters (SWING_N=3,
    ZONE_WIDTH_MULT=0.5xATR, PINBAR_WICK_RATIO=2.0)."""
    o, h, l, c = day_df["open"].tolist(), day_df["high"].tolist(), day_df["low"].tolist(), day_df["close"].tolist()
    n = len(day_df)
    atr = compute_atr(h, l, c, 14)
    zones = []  # each: {"price": level, "touches": int, "is_support": bool} -- built lazily by proximity
    signals = []
    for j in range(n):
        width = 0.5 * atr[j]
        # find/touch a zone near current low (support) or high (resistance)
        for zdict, price_pt, is_support in ((zones, l[j], True), (zones, h[j], False)):
            for z in zdict:
                if z["is_support"] == is_support and abs(z["price"] - price_pt) <= width:
                    z["touches"] += 1
                    break
            else:
                zdict.append({"price": price_pt, "touches": 1, "is_support": is_support})
        for z in zones:
            if z["touches"] >= 2 and abs((l[j] if z["is_support"] else h[j]) - z["price"]) <= width:
                if z["is_support"] and (is_bullish_engulfing(o, c, j) or is_bullish_pinbar(o, h, l, c, j)):
                    signals.append({"bar": j, "side": "LONG"})
                elif not z["is_support"] and (is_bearish_engulfing(o, c, j) or is_bearish_pinbar(o, h, l, c, j)):
                    signals.append({"bar": j, "side": "SHORT"})
    return signals


def detect_setup2_v2_reconstructed(day_df, min_sweep=4.0, confirm_min=2.0, stop_buffer=1.0):
    """RECONSTRUCTED FROM SUMMARY -- Setup 2 v2's core fixes: raised sweep
    threshold + confirmation-candle minimum magnitude (structure-based stop
    is applied at the risk-manager stage uniformly in this file, so only
    the ENTRY trigger logic -- the part that's concept-specific -- is
    reconstructed here)."""
    o, h, l, c = day_df["open"].tolist(), day_df["high"].tolist(), day_df["low"].tolist(), day_df["close"].tolist()
    n = len(day_df)
    sh, sl = find_confirmed_swings(h, l, 3)
    ssh, ssl = sorted(sh.items()), sorted(sl.items())
    shp, slp = 0, 0
    last_sh, last_sl = None, None
    signals = []
    pend_bull, pend_bear, pend_bull_close, pend_bear_close = None, None, None, None
    for j in range(n):
        while shp < len(ssh) and ssh[shp][0] + 3 <= j:
            last_sh = ssh[shp][1]; shp += 1
        while slp < len(ssl) and ssl[slp][0] + 3 <= j:
            last_sl = ssl[slp][1]; slp += 1
        if pend_bull is not None:
            if c[j] - pend_bull_close >= confirm_min:
                signals.append({"bar": j, "side": "LONG"})
            pend_bull = None
        if pend_bear is not None:
            if pend_bear_close - c[j] >= confirm_min:
                signals.append({"bar": j, "side": "SHORT"})
            pend_bear = None
        if last_sl is not None and l[j] < last_sl - min_sweep and c[j] > last_sl:
            pend_bull, pend_bull_close = True, c[j]
        if last_sh is not None and h[j] > last_sh + min_sweep and c[j] < last_sh:
            pend_bear, pend_bear_close = True, c[j]
    return signals


def detect_strike_momentum_reconstructed(day_df, weekday):
    """RECONSTRUCTED, LOWEST CONFIDENCE -- only the risk parameters were in
    the summary (SM_INITIAL_STOP_PTS=10, SM_TRAIL_PTS=3, entry window
    9:15-9:45, weekdays Mon/Wed/Fri), the exact entry TRIGGER rule was not.
    Best-effort proxy used here: a clear directional move away from the
    day's opening price within the first 30 minutes, invalidated if price
    returns back within SM_OPEN_INVALIDATE_PTS(=5) of the open. Treat this
    one with real skepticism -- it is a guess at the original logic."""
    o, h, l, c = day_df["open"].tolist(), day_df["high"].tolist(), day_df["low"].tolist(), day_df["close"].tolist()
    idx = day_df.index
    n = len(day_df)
    if weekday not in (0, 2, 4) or n == 0:
        return []
    day_open = o[0]
    signals = []
    fired = False
    for j in range(n):
        t = idx[j].time()
        if not (dt.time(9, 15) <= t < dt.time(9, 45)):
            continue
        if fired:
            continue
        if c[j] - day_open > 8 and abs(c[j] - day_open) > 5:
            signals.append({"bar": j, "side": "LONG"}); fired = True
        elif day_open - c[j] > 8 and abs(c[j] - day_open) > 5:
            signals.append({"bar": j, "side": "SHORT"}); fired = True
    return signals


CONCEPTS = [
    ("Concepts_2_4", detect_concepts_2_4, "EXACT"),
    ("Setup2_v1_LiquiditySweep", detect_setup2_v1, "EXACT"),
    ("ScalpSetup1_TrendPullback", detect_scalping_setup1_trend_pullback, "EXACT"),
    ("ScalpSetup3_BreakoutRetest", detect_scalping_setup3_breakout_retest, "EXACT"),
    ("SMC_reconstructed", detect_smc_reconstructed, "RECONSTRUCTED"),
    ("PriceAction3Step_reconstructed", detect_price_action_3step_reconstructed, "RECONSTRUCTED"),
    ("Setup2_v2_reconstructed", detect_setup2_v2_reconstructed, "RECONSTRUCTED"),
]


# ============================== RISK MANAGER =================================

def simulate_with_signals(day_df, entry_signals, target_pts, trail_after_target, full_trail_from_start):
    closes, highs, lows = day_df["close"].tolist(), day_df["high"].tolist(), day_df["low"].tolist()
    idx = day_df.index
    n = len(day_df)
    signals_by_bar = {}
    for s in entry_signals:
        signals_by_bar.setdefault(s["bar"], s["side"])
    position = None
    trades = []
    for j in range(n):
        c, h, l = closes[j], highs[j], lows[j]
        if position is not None:
            position["mfe"] = max(position["mfe"], (h - position["entry"]) if position["side"] == "LONG" else (position["entry"] - l))
            position["mae"] = max(position["mae"], (position["entry"] - l) if position["side"] == "LONG" else (h - position["entry"]))
            exited, reason = False, None
            if position["side"] == "LONG":
                if position["trailing_active"]:
                    if c <= position["stop_level"]:
                        exited, reason = True, "TRAIL"
                    else:
                        position["stop_level"] = max(position["stop_level"], l - TRAIL_PTS)
                else:
                    if c <= position["stop"]:
                        exited, reason = True, "STOP"
                    elif target_pts is not None and c >= position["entry"] + target_pts:
                        if trail_after_target:
                            position["trailing_active"] = True; position["stop_level"] = l - TRAIL_PTS
                        else:
                            exited, reason = True, "TARGET"
            else:
                if position["trailing_active"]:
                    if c >= position["stop_level"]:
                        exited, reason = True, "TRAIL"
                    else:
                        position["stop_level"] = min(position["stop_level"], h + TRAIL_PTS)
                else:
                    if c >= position["stop"]:
                        exited, reason = True, "STOP"
                    elif target_pts is not None and c <= position["entry"] - target_pts:
                        if trail_after_target:
                            position["trailing_active"] = True; position["stop_level"] = h + TRAIL_PTS
                        else:
                            exited, reason = True, "TARGET"
            if not exited and (j - position["entry_bar"]) >= dlb.MAX_BARS_IN_TRADE:
                exited, reason = True, "TIME_EXIT"
            if exited:
                pts = (c - position["entry"]) if position["side"] == "LONG" else (position["entry"] - c)
                trades.append({"entry_time": position["entry_time"], "exit_time": idx[j], "side": position["side"],
                               "entry": position["entry"], "exit": c, "points": pts, "reason": reason,
                               "mfe": position["mfe"], "mae": position["mae"], "concept": position["concept"]})
                position = None
        if position is None and j in signals_by_bar:
            side = signals_by_bar[j]
            entry = c
            stop = entry - SL_POINTS if side == "LONG" else entry + SL_POINTS
            stop_level = None
            if full_trail_from_start:
                stop_level = (l - TRAIL_PTS) if side == "LONG" else (h + TRAIL_PTS)
            position = {"side": side, "entry": entry, "entry_time": idx[j], "stop": stop,
                        "trailing_active": full_trail_from_start, "stop_level": stop_level,
                        "entry_bar": j, "mfe": 0.0, "mae": 0.0, "concept": None}
    if position is not None:
        c = closes[-1]
        pts = (c - position["entry"]) if position["side"] == "LONG" else (position["entry"] - c)
        trades.append({"entry_time": position["entry_time"], "exit_time": idx[-1], "side": position["side"],
                       "entry": position["entry"], "exit": c, "points": pts, "reason": "EOD",
                       "mfe": position["mfe"], "mae": position["mae"], "concept": position["concept"]})
    return trades


def classify_30min_session(ts):
    t = ts.time()
    for label in SESSIONS:
        s_, e_ = label.split("-")
        sh_, sm_ = map(int, s_.split(":")); eh_, em_ = map(int, e_.split(":"))
        if dt.time(sh_, sm_) <= t < dt.time(eh_, em_):
            return label
    return "Outside usual session"


# ============================== REPORTING ====================================

def basic_report(trades, label, filename=None):
    if not trades:
        print(f"\n--- {label}: no trades. ---")
        return None
    tdf = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
    wins = tdf[tdf["points"] > 0]
    print(f"\n===== {label} =====")
    print(f"Total: {len(tdf)}  Win%: {100*len(wins)/len(tdf):.1f}  Net pts: {tdf['points'].sum():.2f}  "
          f"Exit reasons: {tdf['reason'].value_counts().to_dict()}")
    if filename:
        tdf.to_csv(filename, index=False)
    rupee_pnl = tdf["points"] * ASSUMED_DELTA * LOT_SIZE - BROKERAGE_PER_TRADE
    capital_curve = STARTING_CAPITAL + rupee_pnl.cumsum()
    print(f"  Money sim: Rs {STARTING_CAPITAL:,} -> Rs {capital_curve.iloc[-1]:,.2f}  "
          f"(max DD Rs {(capital_curve - capital_curve.cummax()).min():,.2f})")
    return {"label": label, "trades": len(tdf), "win_rate": 100*len(wins)/len(tdf),
            "net_points": tdf["points"].sum(), "final_capital": capital_curve.iloc[-1],
            "max_dd": (capital_curve - capital_curve.cummax()).min()}


def full_analytics_report(trades, label, filename):
    tdf = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
    if tdf.empty:
        print(f"\n{label}: no trades, skipping full analytics.")
        return
    tdf.to_csv(filename, index=False)
    n = len(tdf)
    wins = tdf[tdf["points"] > 0]
    losses = tdf[tdf["points"] <= 0]
    win_rate = len(wins) / n

    print(f"\n\n########## FULL ANALYTICS: {label} ({n} trades) ##########")

    print("\n-- Win/Loss Statistics --")
    print(f"Win rate: {100*win_rate:.1f}%  Avg win: {wins['points'].mean() if len(wins) else 0:.2f}  "
          f"Avg loss: {losses['points'].mean() if len(losses) else 0:.2f}  "
          f"Largest win: {tdf['points'].max():.2f}  Largest loss: {tdf['points'].min():.2f}")

    print("\n-- Expectancy --")
    avg_win = wins['points'].mean() if len(wins) else 0
    avg_loss = abs(losses['points'].mean()) if len(losses) else 0
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
    print(f"Expectancy per trade: {expectancy:.2f} points "
          f"({expectancy*ASSUMED_DELTA*LOT_SIZE - BROKERAGE_PER_TRADE:.2f} Rs after brokerage)")

    print("\n-- CE (LONG) vs PE (SHORT) --")
    for side, tag in [("LONG", "CE"), ("SHORT", "PE")]:
        g = tdf[tdf["side"] == side]
        if len(g):
            print(f"{tag}: {len(g)} trades, net {g['points'].sum():.2f} pts, "
                  f"win rate {100*(g['points']>0).mean():.1f}%")

    print("\n-- MFE / MAE Analysis --")
    print(f"Mean MFE (max favorable move seen during trade): {tdf['mfe'].mean():.2f} pts")
    print(f"Mean MAE (max adverse move seen during trade):   {tdf['mae'].mean():.2f} pts")
    print(f"Median MFE: {tdf['mfe'].median():.2f}   Median MAE: {tdf['mae'].median():.2f}")

    print("\n-- Daily (weekday) Analysis --")
    tdf["weekday"] = tdf["entry_time"].dt.day_name()
    for wd in WEEKDAY_ORDER:
        g = tdf[tdf["weekday"] == wd]
        if len(g):
            print(f"{wd}: {len(g)} trades, net {g['points'].sum():.2f} pts, win {100*(g['points']>0).mean():.1f}%")

    print("\n-- 30-Min Session Analysis --")
    tdf["session"] = tdf["entry_time"].apply(classify_30min_session)
    for s in SESSIONS + ["Outside usual session"]:
        g = tdf[tdf["session"] == s]
        if len(g):
            print(f"{s}: {len(g)} trades, net {g['points'].sum():.2f} pts, win {100*(g['points']>0).mean():.1f}%")

    print("\n-- Weekly / Monthly Analysis --")
    tdf["iso_week"] = tdf["entry_time"].dt.isocalendar().week
    tdf["month"] = tdf["entry_time"].dt.to_period("M").astype(str)
    for wk, g in tdf.groupby("iso_week"):
        print(f"Week {wk}: {len(g)} trades, net {g['points'].sum():.2f} pts")
    for mo, g in tdf.groupby("month"):
        print(f"Month {mo}: {len(g)} trades, net {g['points'].sum():.2f} pts")

    print("\n-- Drawdown & Equity Curve --")
    rupee_pnl = tdf["points"] * ASSUMED_DELTA * LOT_SIZE - BROKERAGE_PER_TRADE
    capital_curve = STARTING_CAPITAL + rupee_pnl.cumsum()
    running_max = capital_curve.cummax()
    dd = capital_curve - running_max
    max_dd = dd.min()
    underwater_trades = int((dd < 0).sum())
    print(f"Start Rs {STARTING_CAPITAL:,} -> End Rs {capital_curve.iloc[-1]:,.2f}  "
          f"(P&L Rs {capital_curve.iloc[-1]-STARTING_CAPITAL:,.2f})")
    print(f"Max drawdown: Rs {max_dd:,.2f}   Trades spent underwater: {underwater_trades}/{n}")
    print(f"Equity curve milestones -- min: Rs {capital_curve.min():,.2f}  max: Rs {capital_curve.max():,.2f}")

    print("\n-- Trend Analysis (by 15m HTF trend at entry) --")
    if "trend_15m_at_entry" in tdf.columns:
        for tr, g in tdf.groupby("trend_15m_at_entry"):
            print(f"trend_15m={tr}: {len(g)} trades, net {g['points'].sum():.2f} pts")
    else:
        print("(trend context not attached to this trade set)")

    print("\n-- Walk-Forward-STYLE Consistency Check (sequential weekly folds) --")
    print("(Not classic walk-forward -- no parameters are optimized in this project;")
    print(" this just extends the H1/H2 stability check to more, smaller folds.)")
    for wk, g in tdf.groupby("iso_week"):
        net_rs = (g["points"]*ASSUMED_DELTA*LOT_SIZE - BROKERAGE_PER_TRADE).sum()
        print(f"  Fold (week {wk}): {len(g)} trades, net Rs {net_rs:,.2f}")

    print("\n-- Monte Carlo Simulation (reshuffling the real trade sequence, "
          f"{MONTE_CARLO_ITERATIONS} iterations) --")
    pnl_list = rupee_pnl.tolist()
    final_results, maxdd_results = [], []
    rng = random.Random(42)
    for _ in range(MONTE_CARLO_ITERATIONS):
        shuffled = pnl_list[:]
        rng.shuffle(shuffled)
        curve = STARTING_CAPITAL + pd.Series(shuffled).cumsum()
        final_results.append(curve.iloc[-1])
        maxdd_results.append((curve - curve.cummax()).min())
    final_arr = pd.Series(final_results)
    dd_arr = pd.Series(maxdd_results)
    print(f"Final capital across {MONTE_CARLO_ITERATIONS} random reorderings: "
          f"mean Rs {final_arr.mean():,.2f}, 5th pct Rs {final_arr.quantile(0.05):,.2f}, "
          f"95th pct Rs {final_arr.quantile(0.95):,.2f}")
    print("(Final capital is the SAME across every reordering, by definition -- it's just the sum "
          "of the same trade P&Ls, and a sum doesn't change when you add the same numbers in a "
          "different order. Only the PATH -- and therefore drawdown -- actually varies below.)")
    print(f"Max drawdown across reorderings: mean Rs {dd_arr.mean():,.2f}, "
          f"worst-case (5th pct) Rs {dd_arr.quantile(0.05):,.2f}")
    print(f"(Actual observed result: end Rs {capital_curve.iloc[-1]:,.2f}, max DD Rs {max_dd:,.2f} -- "
          f"compare against the distribution above to see if the real path was lucky/unlucky.)")

    print(f"\nFull trade log saved to {filename}")


# ============================== MAIN =========================================

def main():
    run_self_checks()
    print("=== MASTER OI-GATED COMBINED BACKTEST -- all concepts, 1-month window ===\n")

    dlb.ALLOWED_WEEKDAYS = {0, 1, 2, 3, 4}

    print("Finding current NIFTY futures contract...")
    fut_id, fut_symbol, fut_expiry = find_current_nifty_future()
    print(f"Using {fut_symbol} (id={fut_id}, expiry={fut_expiry})\n")

    print("Fetching NIFTY futures OI (1 month)...")
    oi_df = fetch_futures_oi_chunked(fut_id, FROM_DATE, TO_DATE)
    if oi_df.empty:
        print("No OI data -- stopping.")
        return
    oi_start, oi_end = oi_df.index.min(), oi_df.index.max()
    print(f"OI window: {oi_start} to {oi_end}\n")

    print("Fetching Nifty index price + trend context...")
    warmup_start = (oi_start - dt.timedelta(days=dlb.HTF_CONTEXT_DAYS)).strftime("%Y-%m-%d")
    price_df = fetch_index_chunked(warmup_start, oi_end.strftime("%Y-%m-%d"))
    price_df = dlb.attach_htf_trend(price_df)

    df = price_df.join(oi_df[["oi"]], how="left")
    df["oi"] = df["oi"].ffill()
    oi_change = df["oi"].diff(OI_CHANGE_LOOKBACK_MIN)
    price_change = df["close"].diff(OI_CHANGE_LOOKBACK_MIN)
    df["oi_regime"] = [classify_oi_regime(pc, oc) for pc, oc in zip(price_change, oi_change)]

    test_df = df[df.index.date >= oi_start.date()]
    days_covered = sorted(set(test_df.index.date))
    print(f"Test window: {len(test_df)} candles, {len(days_covered)} days: "
          f"{days_covered[0]} to {days_covered[-1]}\n")

    print("Detecting signals for all concepts (this is the slow part)...\n")
    concept_signals = {}  # name -> {day: [signals]}
    for name, fn, fidelity in CONCEPTS:
        print(f"  Running {name} ({fidelity})...")
        per_day = {}
        for day in days_covered:
            day_df = df[df.index.date == day]
            if len(day_df) < 40:
                continue
            per_day[day] = fn(day_df)
        concept_signals[name] = per_day
        total = sum(len(v) for v in per_day.values())
        print(f"    {total} raw signals.")

    print(f"\n  Running StrikeMomentum_reconstructed (RECONSTRUCTED, LOWEST CONFIDENCE)...")
    sm_per_day = {}
    for day in days_covered:
        day_df = df[df.index.date == day]
        if len(day_df) < 40:
            continue
        sm_per_day[day] = detect_strike_momentum_reconstructed(day_df, day.weekday())
    concept_signals["StrikeMomentum_reconstructed"] = sm_per_day
    print(f"    {sum(len(v) for v in sm_per_day.values())} raw signals.")

    def gate_signals(raw_signals, day_df):
        gated = []
        for s in raw_signals:
            regime = day_df["oi_regime"].iloc[s["bar"]]
            if (s["side"] == "LONG" and regime == "LONG_BUILDUP") or \
               (s["side"] == "SHORT" and regime == "SHORT_BUILDUP"):
                gated.append(s)
        return gated

    print("\n\n===================== PART 1: EACH CONCEPT INDIVIDUALLY (OI-gated, Fixed 1:2) =====================")
    target_pts_baseline = SL_POINTS * 2
    individual_results = []
    pooled_gated_trades = []
    for name in concept_signals:
        all_trades = []
        for day, raw in concept_signals[name].items():
            day_df = df[df.index.date == day]
            gated = gate_signals(raw, day_df)
            if not gated:
                continue
            trades = simulate_with_signals(day_df, gated, target_pts_baseline, False, False)
            for t in trades:
                t["concept"] = name
            all_trades.extend(trades)
            pooled_gated_trades.extend(trades)
        res = basic_report(all_trades, f"{name} (OI-gated, Fixed_1_2)", f"master_{name}_trades.csv")
        if res:
            individual_results.append(res)

    print("\n----- Individual concept comparison table -----")
    print(f"{'Concept':<32}{'Trades':>8}{'Win%':>7}{'NetPts':>9}{'End Rs':>11}{'MaxDD Rs':>11}")
    for r in individual_results:
        print(f"{r['label']:<32}{r['trades']:>8}{r['win_rate']:>6.1f}%{r['net_points']:>9.2f}"
              f"{r['final_capital']:>11,.2f}{r['max_dd']:>11,.2f}")

    print("\n\n===================== PART 2: ALL CONCEPTS POOLED, 8 RISK VARIANTS =====================")
    all_raw_by_day = {}
    for day in days_covered:
        combined = []
        for name in concept_signals:
            combined.extend(concept_signals[name].get(day, []))
        all_raw_by_day[day] = combined

    pooled_results = []
    baseline_trades_for_analytics = None
    for variant_name, target_pts, trail_after, full_trail in RISK_VARIANTS:
        all_trades = []
        for day in days_covered:
            day_df = df[df.index.date == day]
            gated = gate_signals(all_raw_by_day.get(day, []), day_df)
            if not gated:
                continue
            all_trades.extend(simulate_with_signals(day_df, gated, target_pts, trail_after, full_trail))
        res = basic_report(all_trades, f"POOLED_ALL | {variant_name}", f"master_pooled_{variant_name}_trades.csv")
        if res:
            pooled_results.append(res)
        if variant_name == BASELINE_VARIANT:
            baseline_trades_for_analytics = all_trades

    print("\n----- Pooled portfolio, all 8 risk variants -----")
    print(f"{'Variant':<28}{'Trades':>8}{'Win%':>7}{'NetPts':>9}{'End Rs':>11}{'MaxDD Rs':>11}")
    for r in pooled_results:
        print(f"{r['label']:<28}{r['trades']:>8}{r['win_rate']:>6.1f}%{r['net_points']:>9.2f}"
              f"{r['final_capital']:>11,.2f}{r['max_dd']:>11,.2f}")

    if baseline_trades_for_analytics:
        full_analytics_report(baseline_trades_for_analytics,
                               f"POOLED_ALL | {BASELINE_VARIANT}",
                               "master_pooled_baseline_full_trade_log.csv")
    else:
        print("\nNo trades in the baseline variant -- skipping full analytics section.")


if __name__ == "__main__":
    main()
