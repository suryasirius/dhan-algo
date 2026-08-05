"""
Combined Portfolio Backtest: Velez 8pt Short + Concepts 2 & 4
=================================================================
Runs your two PROVEN-profitable strategies side by side on the same
2-year window and the same shared starting capital, as if both were
traded from one account, and reports both the combined result and
each strategy's individual contribution.

VELEZ 8pt SHORT (GBI + ELEPHANT_PLUS, 8pt initial stop / 4pt trailing):
reproduced verbatim from velez_final_8pt.py. That file is LOCKED per
this project's convention (never edited directly) -- this script does
NOT import or modify it, it re-implements the exact same logic in a
new standalone file, same as every other Velez variant in this project.

CONCEPTS 2 & 4: reused via the SAME calling convention as every other
Concepts 2&4 backtest in this project -- dlb.compute_latest_signal(),
dlb.attach_htf_trend(), and dlb.SL_POINTS / dlb.TP_POINTS /
dlb.MAX_BARS_IN_TRADE / dlb.ALLOWED_WEEKDAYS read directly off your
dhan_live_bot.py. This script does NOT reimplement or guess at concept
detection -- that logic lives entirely in your own dhan_live_bot.py and
is used exactly as your live bot would use it. Nothing in
dhan_live_bot.py is modified or needs to be understood beyond calling
these functions, same as always.

IMPORTANT -- the two strategies keep their own, ALREADY-PROVEN P&L
conventions rather than being forced onto one shared formula:
  - Velez converts index points to Rs 1:1 (points x lot_size), matching
    the exact convention velez_final_8pt.py was validated with
    (Rs 44,344.75 profit over 2yrs).
  - Concepts 2&4 converts index points to Rs via the project's
    ASSUMED_DELTA=0.6 options-proxy (points x 0.6 x lot_size), matching
    the exact convention its own 6-month/2-year results were computed
    with (Rs 18,075 / ~Rs 20,863).
Harmonizing these would silently change both strategies away from
their already-validated numbers, so each keeps its own formula here.

CAPITAL & LOT SIZE CHANGES FROM THE ORIGINAL FILES (flagged deliberately):
  - Shared STARTING_CAPITAL = Rs 25,000 (Velez's own proven-adequate
    level; Concepts 2&4 already showed a small drawdown at Rs 10,000,
    so Rs 25,000 comfortably covers both running together).
  - Both strategies use the date-aware lot_size_for_date() lookup. The
    original Concepts 2&4 6-month script hardcoded LOT_SIZE=65 (only
    correct for recent months) -- fixed here so the full 2-year run is
    historically correct throughout.

Both strategies run fully independently (each manages its own signals,
entries, and exits) -- their trade logs are simply merged onto one
chronological equity curve afterward, so you can see combined
drawdown, whether the two return streams are correlated or diversify
each other, and the combined Monte Carlo wipeout risk.
"""

import datetime as dt
import math
import random
import pandas as pd
import numpy as np

import dhan_live_bot as dlb

FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")   # 2 years
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

STARTING_CAPITAL = 25_000
BROKERAGE_PER_TRADE = 40.0
STRIKE_STEP = 50
EOD_SQUAREOFF = dt.time(15, 20)
MONTE_CARLO_ITERS = 5000
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# ---------------------------- shared helpers ----------------------------

def lot_size_for_date(d):
    if d < dt.date(2024, 11, 20):
        return 25
    elif d < dt.date(2026, 1, 1):
        return 75
    else:
        return 65


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


def resample_2min_per_day(df, bar_rule="2min"):
    out = []
    for day in sorted(set(df.index.date)):
        day_df = df[df.index.date == day]
        r = day_df.resample(bar_rule, origin=day_df.index[0]).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        out.append(r)
    return pd.concat(out).sort_index()


# ============================ VELEZ 8pt SHORT ============================
# Reproduced verbatim from velez_final_8pt.py (LOCKED, not modified/imported).

SMA_FAST = 20
SMA_SLOW = 200
POP_SLOPE_LOOKBACK = 5
AVG_RANGE_LOOKBACK = 20

SMALL_BAR_MULT = 0.6
ELEPHANT_MULT = 2.0
ELEPHANT_PLUS_RETAIN = 0.8
NEAR_20_PCT = 0.0015

VELEZ_INITIAL_SL_PTS = 8
VELEZ_TRAIL_PTS = 4


def expiry_weekday_for_date(d):
    return 3 if d < dt.date(2025, 9, 1) else 1


def next_expiry(d):
    target_wd = expiry_weekday_for_date(d)
    days_ahead = (target_wd - d.weekday()) % 7
    return d + dt.timedelta(days=days_ahead)


def itm_put_strike(spot):
    return math.ceil(spot / STRIKE_STEP) * STRIKE_STEP


def compute_velez_indicators(df):
    df = df.copy()
    df["sma20"] = df["close"].rolling(SMA_FAST, min_periods=SMA_FAST).mean()
    df["sma200"] = df["close"].rolling(SMA_SLOW, min_periods=SMA_SLOW).mean()
    df["range"] = df["high"] - df["low"]
    df["avg_range"] = df["range"].shift(1).rolling(AVG_RANGE_LOOKBACK, min_periods=AVG_RANGE_LOOKBACK).mean()
    return df


def pop_bearish(price, sma20, sma20_prev, sma200):
    if pd.isna(sma20) or pd.isna(sma200) or pd.isna(sma20_prev):
        return False
    return price < sma20 and sma20 < sma200 and sma20 < sma20_prev


def is_green(o, c):
    return c > o


def is_red(o, c):
    return c < o


def is_small(rng, avgr):
    return bool(avgr) and avgr > 0 and rng < SMALL_BAR_MULT * avgr


def is_elephant_plus_bear(o, h, l, c, avgr):
    rng = h - l
    if not (bool(avgr) and avgr > 0 and (rng * ELEPHANT_PLUS_RETAIN) > ELEPHANT_MULT * avgr):
        return False
    return is_red(o, c)


def find_small_green_run(highs, lows, opens, closes, avg_ranges, end_idx):
    i = end_idx - 1
    if i < 0:
        return None
    if not (is_green(opens[i], closes[i]) and is_small(highs[i] - lows[i], avg_ranges[i])):
        return None
    combined_high, combined_low = highs[i], lows[i]
    start_idx = i
    j = i - 1
    while j >= 0 and is_green(opens[j], closes[j]) and is_small(highs[j] - lows[j], avg_ranges[j]):
        combined_high = max(combined_high, highs[j])
        combined_low = min(combined_low, lows[j])
        start_idx = j
        j -= 1
    if start_idx - 1 < 0:
        return None
    if not is_red(opens[start_idx - 1], closes[start_idx - 1]):
        return None
    return start_idx, combined_high, combined_low


def detect_velez_signals_for_day(day_df):
    opens = day_df["open"].tolist()
    highs = day_df["high"].tolist()
    lows = day_df["low"].tolist()
    closes = day_df["close"].tolist()
    sma20 = day_df["sma20"].tolist()
    sma200 = day_df["sma200"].tolist()
    avg_ranges = day_df["avg_range"].tolist()
    n = len(day_df)

    signals = []
    if n < POP_SLOPE_LOOKBACK + 3:
        return signals

    for j in range(POP_SLOPE_LOOKBACK, n - 1):
        if pd.isna(sma20[j]) or pd.isna(sma200[j]) or pd.isna(sma20[j - POP_SLOPE_LOOKBACK]):
            continue
        if not pop_bearish(closes[j], sma20[j], sma20[j - POP_SLOPE_LOOKBACK], sma200[j]):
            continue

        near20 = abs(closes[j] - sma20[j]) / closes[j] < NEAR_20_PCT

        if near20:
            run = find_small_green_run(highs, lows, opens, closes, avg_ranges, j)
            if run:
                _, comb_high, comb_low = run
                if closes[j] < comb_low and is_red(opens[j], closes[j]):
                    signals.append({"bar": j, "side": "SHORT", "event": "GBI"})
                    continue

        if j - 1 >= 0:
            eo, eh, el, ec = opens[j - 1], highs[j - 1], lows[j - 1], closes[j - 1]
            eavgr = avg_ranges[j - 1]
            if is_elephant_plus_bear(eo, eh, el, ec, eavgr) and closes[j] < el:
                signals.append({"bar": j, "side": "SHORT", "event": "ELEPHANT_PLUS"})

    seen_bars = set()
    deduped = []
    for s in signals:
        if s["bar"] in seen_bars:
            continue
        seen_bars.add(s["bar"])
        deduped.append(s)
    return deduped


def simulate_velez_trade(day_df, sig, day):
    closes = day_df["close"].tolist()
    highs = day_df["high"].tolist()
    idx = day_df.index
    n = len(day_df)
    j0 = sig["bar"]
    entry_price = closes[j0]

    lot = lot_size_for_date(day)
    expiry = next_expiry(day)
    strike = itm_put_strike(entry_price)

    stop = entry_price + VELEZ_INITIAL_SL_PTS
    trailing_active = False

    exit_j, exit_price, reason = n - 1, closes[-1], "EOD"
    for j in range(j0 + 1, n):
        t = idx[j]
        h, c = highs[j], closes[j]
        if h >= stop:
            exit_j, exit_price, reason = j, stop, ("TRAIL_SL" if trailing_active else "INITIAL_SL")
            break
        if c < entry_price:
            trailing_active = True
        if trailing_active:
            stop = min(stop, h + VELEZ_TRAIL_PTS)
        if t.time() >= EOD_SQUAREOFF:
            exit_j, exit_price, reason = j, c, "EOD"
            break

    points_gained = entry_price - exit_price
    rupee_pnl = points_gained * lot - BROKERAGE_PER_TRADE   # Velez's own 1:1 convention

    return {
        "date": day, "event": sig["event"], "side": "PE", "strike": strike, "expiry": expiry,
        "lot_size": lot, "entry_time": idx[j0], "exit_time": idx[exit_j],
        "points": points_gained, "reason": reason, "rupee_pnl": rupee_pnl,
        "strategy": "Velez_8pt",
    }


# =========================== CONCEPTS 2 & 4 ===========================
# Reconstructed from this project's own concepts_2_4_6month_standalone.py
# (recovered via decompilation after the source file was lost to an
# outputs-folder reset -- the trade-management logic below is verified
# to match that file's bytecode exactly; concept detection itself is
# NOT reimplemented, it's called live via dlb.compute_latest_signal()).

WARMUP_BARS = 30
CONCEPTS_ASSUMED_DELTA = 0.6


def simulate_concepts_day(day_df, lot):
    """Expanding-window calls to dlb.compute_latest_signal(), then SL/TP/
    time-exit management using dlb.SL_POINTS / dlb.TP_POINTS /
    dlb.MAX_BARS_IN_TRADE -- exactly the calling convention every other
    Concepts 2&4 backtest in this project uses."""
    closes = day_df["close"].tolist()
    idx = day_df.index
    n = len(day_df)
    trades = []
    position = None

    for i in range(WARMUP_BARS, n):
        c = closes[i]

        if position is not None:
            exited, reason = False, None
            if position["side"] == "LONG":
                if c <= position["entry"] - dlb.SL_POINTS:
                    exited, reason = True, "STOP"
                elif c >= position["entry"] + dlb.TP_POINTS:
                    exited, reason = True, "TARGET"
            else:
                if c >= position["entry"] + dlb.SL_POINTS:
                    exited, reason = True, "STOP"
                elif c <= position["entry"] - dlb.TP_POINTS:
                    exited, reason = True, "TARGET"
            if not exited and (i - position["entry_bar"]) >= dlb.MAX_BARS_IN_TRADE:
                exited, reason = True, "TIME_EXIT"

            if exited:
                pts = (c - position["entry"]) if position["side"] == "LONG" else (position["entry"] - c)
                rupee_pnl = pts * CONCEPTS_ASSUMED_DELTA * lot - BROKERAGE_PER_TRADE
                trades.append({
                    "entry_time": position["entry_time"], "exit_time": idx[i],
                    "side": position["side"], "points": pts, "reason": reason,
                    "concept_id": position["concept_id"], "rupee_pnl": rupee_pnl,
                    "strategy": "Concepts_2_4",
                })
                position = None

        if position is None:
            sig = dlb.compute_latest_signal(day_df.iloc[: i + 1])
            if sig is None:
                continue
            if sig.get("final_buy"):
                position = {"side": "LONG", "entry": c, "entry_time": idx[i], "entry_bar": i,
                            "concept_id": sig.get("concept_id")}
                continue
            if sig.get("final_sell"):
                position = {"side": "SHORT", "entry": c, "entry_time": idx[i], "entry_bar": i,
                            "concept_id": sig.get("concept_id")}
                continue

    if position is not None:
        c = closes[-1]
        pts = (c - position["entry"]) if position["side"] == "LONG" else (position["entry"] - c)
        rupee_pnl = pts * CONCEPTS_ASSUMED_DELTA * lot - BROKERAGE_PER_TRADE
        trades.append({
            "entry_time": position["entry_time"], "exit_time": idx[-1],
            "side": position["side"], "points": pts, "reason": "EOD",
            "concept_id": position["concept_id"], "rupee_pnl": rupee_pnl,
            "strategy": "Concepts_2_4",
        })
    return trades


# ------------------------------ self-checks ------------------------------

class _StubSignalSource:
    """Fake dlb.compute_latest_signal/attach_htf_trend/SL_POINTS/TP_POINTS/
    MAX_BARS_IN_TRADE, used ONLY to self-check the trade-management wrapper
    above (simulate_concepts_day). This does NOT attempt to reproduce real
    concept-detection logic -- that lives in your actual dhan_live_bot.py
    and is out of scope for a self-check; this only verifies the STOP/
    TARGET/TIME_EXIT/EOD/Rs-conversion code this script itself adds."""
    def __init__(self, fire_at, side, concept_id=2):
        self.fire_at = fire_at
        self.side = side
        self.concept_id = concept_id
        self.SL_POINTS = 5
        self.TP_POINTS = 10
        self.MAX_BARS_IN_TRADE = 5
        self.ALLOWED_WEEKDAYS = {1, 2}

    def compute_latest_signal(self, window_df):
        i = len(window_df) - 1
        if i == self.fire_at:
            return {"final_buy": self.side == "LONG", "final_sell": self.side == "SHORT",
                    "concept_id": self.concept_id}
        return None

    def attach_htf_trend(self, df):
        return df


def run_self_checks():
    print("Running self-checks (synthetic data)...")

    # ---- Velez trailing-stop math (identical hand-trace to velez_final_8pt.py) ----
    idx = pd.date_range("2025-01-06 09:15", periods=6, freq="2min")
    closes_t = [100, 103, 99, 92, 97, 104]
    highs_t = [100, 104, 100.5, 93, 98, 105]
    lows_t = [100, 102, 98.5, 91, 96, 103]
    opens_t = [100, 100, 103, 99, 92, 97]
    day_df = pd.DataFrame({"open": opens_t, "high": highs_t, "low": lows_t, "close": closes_t}, index=idx)
    trade = simulate_velez_trade(day_df, {"bar": 0, "side": "SHORT", "event": "TEST"}, dt.date(2025, 1, 6))
    assert trade["reason"] == "TRAIL_SL", f"expected TRAIL_SL, got {trade['reason']}"
    assert abs(trade["points"] - 3.0) < 0.01
    print("  simulate_velez_trade trailing-stop math check ok")

    # ---- Concepts 2&4 wrapper: TARGET hit (LONG) ----
    global dlb
    real_dlb = dlb
    stub = _StubSignalSource(fire_at=WARMUP_BARS, side="LONG", concept_id=2)
    dlb = stub
    idx2 = pd.date_range("2025-03-10 09:15", periods=WARMUP_BARS + 15, freq="1min")
    # Note: the real Concepts 2&4 exit logic (matched from the decompiled
    # bytecode) does NOT clamp the fill to the exact SL/TP level -- it
    # exits at whatever close crossed the threshold, so points can overshoot
    # the 5/10 line slightly. Land the rally exactly on 110 here so the
    # check has an unambiguous expected value.
    closes2 = [100.0] * (WARMUP_BARS + 1) + [101, 103, 106, 109, 110, 108, 105, 104, 103, 102, 101, 100, 99, 98]
    day_df2 = pd.DataFrame({"open": [100.0] + closes2[:-1], "high": [c + 0.2 for c in closes2],
                             "low": [c - 0.2 for c in closes2], "close": closes2}, index=idx2)
    trades2 = simulate_concepts_day(day_df2, lot=75)
    assert len(trades2) == 1 and trades2[0]["reason"] == "TARGET", f"expected 1 TARGET trade, got {trades2}"
    assert abs(trades2[0]["points"] - 10.0) < 0.01
    expected_rs = 10.0 * CONCEPTS_ASSUMED_DELTA * 75 - BROKERAGE_PER_TRADE
    assert abs(trades2[0]["rupee_pnl"] - expected_rs) < 0.01
    print("  simulate_concepts_day LONG TARGET-hit + Rs-conversion check ok")

    # ---- Concepts 2&4 wrapper: STOP hit (SHORT) ----
    stub2 = _StubSignalSource(fire_at=WARMUP_BARS, side="SHORT", concept_id=4)
    dlb = stub2
    closes3 = [100.0] * (WARMUP_BARS + 1) + [101, 103, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94]
    day_df3 = pd.DataFrame({"open": [100.0] + closes3[:-1], "high": [c + 0.2 for c in closes3],
                             "low": [c - 0.2 for c in closes3], "close": closes3}, index=idx2)
    trades3 = simulate_concepts_day(day_df3, lot=75)
    assert len(trades3) == 1 and trades3[0]["reason"] == "STOP", f"expected 1 STOP trade, got {trades3}"
    assert abs(trades3[0]["points"] - (-5.0)) < 0.01
    print("  simulate_concepts_day SHORT STOP-hit check ok")

    # ---- Concepts 2&4 wrapper: TIME_EXIT (never reaches SL or TP within MAX_BARS_IN_TRADE) ----
    stub3 = _StubSignalSource(fire_at=WARMUP_BARS, side="LONG", concept_id=2)
    dlb = stub3
    closes4 = [100.0] * (WARMUP_BARS + 1) + [100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 90, 90, 90, 90, 90, 90, 90, 90]
    day_df4 = pd.DataFrame({"open": [100.0] + closes4[:-1], "high": [c + 0.2 for c in closes4],
                             "low": [c - 0.2 for c in closes4], "close": closes4}, index=idx2)
    trades4 = simulate_concepts_day(day_df4, lot=75)
    assert len(trades4) == 1 and trades4[0]["reason"] == "TIME_EXIT", f"expected TIME_EXIT, got {trades4}"
    print("  simulate_concepts_day TIME_EXIT (MAX_BARS_IN_TRADE) check ok")

    dlb = real_dlb  # restore the real module reference before any real run

    # ---- combined equity merge sanity check ----
    a = pd.DataFrame({"entry_time": pd.to_datetime(["2025-01-01 10:00", "2025-01-03 10:00"]),
                       "rupee_pnl": [100.0, -50.0], "strategy": "Velez_8pt"})
    b = pd.DataFrame({"entry_time": pd.to_datetime(["2025-01-02 10:00", "2025-01-04 10:00"]),
                       "rupee_pnl": [200.0, 300.0], "strategy": "Concepts_2_4"})
    combined = pd.concat([a, b]).sort_values("entry_time").reset_index(drop=True)
    combined["running_equity"] = 1000 + combined["rupee_pnl"].cumsum()
    assert list(combined["strategy"]) == ["Velez_8pt", "Concepts_2_4", "Velez_8pt", "Concepts_2_4"]
    assert abs(combined["running_equity"].iloc[-1] - (1000 + 100 - 50 + 200 + 300)) < 0.01
    print("  combined chronological merge + equity-curve check ok")

    print("All self-checks passed.\n")


def main():
    run_self_checks()
    print("=== Combined Portfolio: Velez 8pt Short + Concepts 2 & 4 ===\n")
    print(f"Shared starting capital: Rs {STARTING_CAPITAL:,}   Brokerage: Rs {BROKERAGE_PER_TRADE:.2f}/trade\n")
    print(f"ALLOWED_WEEKDAYS currently set to: {dlb.ALLOWED_WEEKDAYS} "
          f"(should be {{1, 2}} = Tue/Wed if nothing's changed in dhan_live_bot.py)\n")

    print(f"Fetching Nifty index 1-min data ({FROM_DATE} to {TO_DATE})...")
    df1m = fetch_index_chunked(FROM_DATE, TO_DATE)
    if df1m.empty:
        print("No data returned.")
        return
    days_covered = sorted(set(df1m.index.date))
    print(f"Got {len(df1m)} 1-min bars across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}\n")

    # ---------------- Concepts 2 & 4 (runs on raw 1-min data) ----------------
    df1m_htf = dlb.attach_htf_trend(df1m)
    concepts_trades = []
    for day in days_covered:
        day_df = df1m_htf[df1m_htf.index.date == day]
        if len(day_df) < WARMUP_BARS + 1:
            continue
        lot = lot_size_for_date(day)
        concepts_trades.extend(simulate_concepts_day(day_df, lot))
    concepts_tdf = pd.DataFrame(concepts_trades)
    if not concepts_tdf.empty:
        concepts_tdf = concepts_tdf.sort_values("entry_time").reset_index(drop=True)

    # ---------------- Velez 8pt Short (runs on 2-min resampled data) ----------------
    df2m = resample_2min_per_day(df1m)
    df2m = compute_velez_indicators(df2m)
    velez_trades = []
    for day in days_covered:
        day_df = df2m[df2m.index.date == day].copy()
        signals = detect_velez_signals_for_day(day_df)
        open_trade_bar = -1
        for sig in signals:
            if sig["bar"] <= open_trade_bar:
                continue
            t = simulate_velez_trade(day_df, sig, day)
            velez_trades.append(t)
            open_trade_bar = day_df.index.get_loc(t["exit_time"])
    velez_tdf = pd.DataFrame(velez_trades)
    if not velez_tdf.empty:
        velez_tdf = velez_tdf.sort_values("entry_time").reset_index(drop=True)

    if concepts_tdf.empty and velez_tdf.empty:
        print("No trades generated by either strategy -- check ALLOWED_WEEKDAYS / DISABLED_CONCEPTS "
              "in dhan_live_bot.py, and confirm the Velez signal logic still fires on this window.")
        return

    def stats_block(g, label):
        if g.empty:
            print(f"{label}: no trades")
            return
        wins = g[g["points"] > 0]
        losses = g[g["points"] <= 0]
        win_rate = len(wins) / len(g)
        pf = (wins["points"].sum() / abs(losses["points"].sum())
              if losses["points"].sum() != 0 else float("inf"))
        print(f"{label}: n={len(g)}  win%={100*win_rate:.1f}  PF={pf:.2f}  "
              f"net_pts={g['points'].sum():.2f}  net_Rs={g['rupee_pnl'].sum():,.2f}")

    print("--- Individual strategy results ---")
    stats_block(concepts_tdf, "Concepts_2_4")
    stats_block(velez_tdf, "Velez_8pt")
    print()

    # ---------------- combined portfolio ----------------
    cols = ["entry_time", "exit_time", "strategy", "points", "reason", "rupee_pnl"]
    combined = pd.concat([
        concepts_tdf[cols] if not concepts_tdf.empty else pd.DataFrame(columns=cols),
        velez_tdf[cols] if not velez_tdf.empty else pd.DataFrame(columns=cols),
    ]).sort_values("entry_time").reset_index(drop=True)
    combined["running_equity"] = STARTING_CAPITAL + combined["rupee_pnl"].cumsum()
    combined.to_csv("velez_plus_concepts24_combined_trades.csv", index=False)

    print("--- Combined portfolio ---")
    stats_block(combined, "Combined")
    equity = combined["running_equity"]
    peak = equity.cummax()
    max_dd = (equity - peak).min()
    print(f"Start Rs {STARTING_CAPITAL:,} -> End Rs {equity.iloc[-1]:,.2f}   Max DD Rs {max_dd:,.2f}")
    print("WARNING: wipes out at some point." if (equity <= 0).any() else "Never wipes out.")
    print()

    # ---------------- diversification check: daily P&L correlation ----------------
    print("--- Diversification: daily P&L correlation between the two strategies ---")
    if not concepts_tdf.empty and not velez_tdf.empty:
        c_daily = concepts_tdf.assign(date=concepts_tdf["entry_time"].dt.date).groupby("date")["rupee_pnl"].sum()
        v_daily = velez_tdf.assign(date=velez_tdf["entry_time"].dt.date).groupby("date")["rupee_pnl"].sum()
        all_dates = sorted(set(days_covered))
        c_series = pd.Series(0.0, index=all_dates)
        v_series = pd.Series(0.0, index=all_dates)
        c_series.update(c_daily)
        v_series.update(v_daily)
        corr = np.corrcoef(c_series.values, v_series.values)[0, 1]
        print(f"Correlation of daily P&L (Concepts_2_4 vs Velez_8pt): {corr:.3f}")
        if corr < 0.2:
            print("  -> low correlation: running both together genuinely diversifies risk.")
        else:
            print("  -> meaningfully correlated: the two strategies tend to win/lose on similar days.")
    else:
        print("  One strategy generated no trades -- correlation not meaningful.")
    print()

    # ---------------- stability check ----------------
    mid = len(combined) // 2
    h1 = combined["rupee_pnl"].iloc[:mid].sum()
    h2 = combined["rupee_pnl"].iloc[mid:].sum()
    print("--- Stability check (combined) ---")
    print(f"First half: n={mid} net Rs {h1:,.2f}   |   Second half: n={len(combined)-mid} net Rs {h2:,.2f}\n")

    # ---------------- Monte Carlo (combined trade-order reshuffle) ----------------
    print("--- Monte Carlo (combined trade-order reshuffle) ---")
    pnl_list = combined["rupee_pnl"].tolist()
    final_equities, min_equities = [], []
    for _ in range(MONTE_CARLO_ITERS):
        shuffled = pnl_list.copy()
        random.shuffle(shuffled)
        curve = STARTING_CAPITAL + np.cumsum(shuffled)
        final_equities.append(curve[-1])
        min_equities.append(curve.min())
    fe = np.array(final_equities)
    me = np.array(min_equities)
    print(f"Final equity -> 5th: Rs {np.percentile(fe,5):,.2f}  50th: Rs {np.percentile(fe,50):,.2f}  "
          f"95th: Rs {np.percentile(fe,95):,.2f}")
    print(f"Probability equity ever goes <= 0: {100*(me<=0).mean():.2f}%")
    print(f"Probability equity falls below 50% of starting capital: {100*(fe < STARTING_CAPITAL*0.5).mean():.2f}%\n")

    print(f"Combined trade log saved: velez_plus_concepts24_combined_trades.csv ({len(combined)} rows)")


if __name__ == "__main__":
    main()
