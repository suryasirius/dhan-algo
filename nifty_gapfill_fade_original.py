"""
ORIGINAL STRATEGY (my own idea, not from any provided doc): Nifty Overnight
Gap-Fill Fade
=================================================================
Why this idea specifically: every doc-derived strategy tested this session
(the AMT model and its companion refinements) depends on volume-spike or
footprint-style "aggression" data that has turned out to be either
unavailable or too fragile to fire on real Nifty data (see
nifty_amt_diagnose_zero_trades.py -- the aggression gate passes only 0.7%
of eligible bars). I deliberately picked something that sidesteps that
whole problem class entirely, and that tests a genuinely different
mechanism from every momentum/breakout pattern already tried in this
project (Velez 8pt, Opening-Drive Momentum, Crash Play, the AMT model):

  HYPOTHESIS: Nifty's overnight gap (today's opening price vs YESTERDAY's
  closing price, driven mostly by overnight global cues -- SGX/GIFT Nifty,
  US markets) has some tendency to partially or fully "fill" (retrace back
  toward yesterday's close) during the session when the gap is large
  enough to be meaningful. This is a well-known, testable claim in many
  markets -- this script tests it directly on 2 years of real Nifty data
  rather than assuming it transfers.

RULE: if |gap| >= GAP_THRESHOLD_PTS, fade it. Gap UP -> SHORT (buy the ITM
PUT), betting on a pullback toward yesterday's close. Gap DOWN -> LONG
(buy the ITM CALL), betting on a bounce toward yesterday's close. Target =
yesterday's close (a full-fill test, the simplest and most literal version
of the hypothesis -- a partial-fill variant is a natural follow-up if this
looks real). Stop = a fixed number of premium points, same convention as
every other strategy tested in this project. EOD square-off. No trailing
on this first pass -- the whole point is testing the raw fill hypothesis
without extra machinery obscuring whether it's real.

WHY THERE'S NO LOOK-AHEAD ISSUE HERE (unlike every other strategy in this
project): the signal itself -- yesterday's close vs today's gap -- is
fully knowable BEFORE the market even opens. There's no "same-bar-close-
generated-the-signal" problem to create, because the signal doesn't come
from any bar at all. Entry is still delayed past the first SKIP_MINUTES of
the session, but that's a discretionary "avoid the opening liquidity-test
chop" choice (same lesson already learned from the AMT docs this
session), not a correctness fix.

Conservative intrabar ordering: identical convention to every other script
here -- within any single bar, STOP is checked before TARGET.

Same proxy premium model, lot-size schedule, brokerage, and Rs25,000
starting capital used everywhere else in this project, for comparability.
"""

import datetime as dt
import math
import pandas as pd
import numpy as np

import dhan_live_bot as dlb

FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85
BAR_RULE = "2min"

STRIKE_STEP = 50
EOD_SQUAREOFF = dt.time(15, 20)
SKIP_MINUTES = 10   # discretionary opening-chop avoidance, not a look-ahead fix (see docstring)

GAP_THRESHOLD_PTS = 25.0   # a priori pick (~0.1% of Nifty's current ~24-25k level), NOT tuned on this data
STOP_PTS = 12.0            # a priori pick, same reasoning -- a sweep is a natural follow-up, not done on the first look

ASSUMED_DELTA = 0.6
ASSUMED_TIME_VALUE = 40.0
BROKERAGE_PER_TRADE = 40.0
STARTING_CAPITAL = 25_000


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
        last_err, df = None, None
        for attempt in range(2):
            try:
                resp = dlb.dhan.intraday_minute_data(
                    security_id=dlb.UNDERLYING_SECURITY_ID, exchange_segment=dlb.UNDERLYING_EXCHANGE_SEGMENT,
                    instrument_type=dlb.UNDERLYING_INSTRUMENT_TYPE,
                    from_date=cur.strftime("%Y-%m-%d"), to_date=chunk_end.strftime("%Y-%m-%d"),
                )
                if not isinstance(resp, dict):
                    raise RuntimeError(f"non-dict response: {str(resp)[:300]}")
                data = resp.get("data", resp)
                if not isinstance(data, dict) or "open" not in data:
                    raise RuntimeError(f"response missing OHLC keys: {str(resp)[:300]}")
                df = pd.DataFrame({
                    "open": data["open"], "high": data["high"], "low": data["low"], "close": data["close"],
                    "timestamp": data["timestamp"],
                })
                df["time"] = pd.to_datetime(df["timestamp"], unit="s") + pd.Timedelta(hours=5, minutes=30)
                df = df.set_index("time").sort_index()
                break
            except Exception as e:
                last_err = e
                if attempt == 0:
                    import time as _t
                    _t.sleep(2)
        if df is None:
            raise RuntimeError(f"fetch failed for chunk {cur} to {chunk_end}: {last_err}")
        chunks.append(df)
        cur = chunk_end + dt.timedelta(days=1)
    full = pd.concat(chunks)
    return full[~full.index.duplicated(keep="first")].sort_index()


def resample_2min_per_day(df):
    out = []
    for day in sorted(set(df.index.date)):
        day_df = df[df.index.date == day]
        r = day_df.resample(BAR_RULE, origin=day_df.index[0]).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
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


def premium_at_index_level(index_level, day_open_spot, strike, side):
    if side == "CE":
        intrinsic0 = max(day_open_spot - strike, 0.0)
        move = index_level - day_open_spot
    else:
        intrinsic0 = max(strike - day_open_spot, 0.0)
        move = day_open_spot - index_level
    return max(intrinsic0 + ASSUMED_TIME_VALUE + move * ASSUMED_DELTA, 0.0)


def detect_gap_signal(prev_close, day_open, threshold=GAP_THRESHOLD_PTS):
    gap = day_open - prev_close
    if gap >= threshold:
        return "SHORT", gap
    if gap <= -threshold:
        return "LONG", gap
    return None, gap


def simulate_trade(premium_df, entry_idx, entry_price, stop_level, target_level):
    n = len(premium_df)
    for j in range(entry_idx, n):
        bar = premium_df.iloc[j]
        t = premium_df.index[j].time()
        if bar["low"] <= stop_level:
            return stop_level, "STOP", premium_df.index[j]
        if bar["high"] >= target_level:
            return target_level, "TARGET", premium_df.index[j]
        if t >= EOD_SQUAREOFF:
            return bar["close"], "EOD", premium_df.index[j]
    last = premium_df.iloc[-1]
    return last["close"], "EOD", premium_df.index[-1]


def run_self_checks():
    print("Running self-checks (synthetic data)...")

    d, gap = detect_gap_signal(prev_close=22000, day_open=22030)
    assert d == "SHORT" and abs(gap - 30) < 1e-6
    d, gap = detect_gap_signal(prev_close=22000, day_open=21965)
    assert d == "LONG" and abs(gap - (-35)) < 1e-6
    d, gap = detect_gap_signal(prev_close=22000, day_open=22010)
    assert d is None and abs(gap - 10) < 1e-6
    print("  detect_gap_signal (SHORT/LONG/no-trade) check ok")

    p = premium_at_index_level(21970, 22000, itm_put_strike(22000), side="PE")
    # strike=ceil(22000/50)*50=22000, intrinsic0=0; move=22000-21970=30 -> premium=0+40+30*0.6=58.0
    assert abs(p - 58.0) < 1e-6, f"expected 58.0, got {p}"
    print("  premium_at_index_level check ok")

    idx_t = pd.date_range("2025-01-06 09:15", periods=6, freq="2min")
    # clean STOP case
    prem_stop = pd.DataFrame({
        "open": [50, 48, 44, 40, 40, 40], "high": [50, 49, 45, 41, 41, 41],
        "low": [50, 47, 43, 39, 39, 39], "close": [50, 48, 44, 40, 40, 40],
    }, index=idx_t)
    exit_p, reason, _ = simulate_trade(prem_stop, 0, entry_price=50, stop_level=42, target_level=90)
    assert reason == "STOP" and abs(exit_p - 42.0) < 1e-6
    print("  simulate_trade clean STOP-hit check ok")

    # clean TARGET case
    prem_target = pd.DataFrame({
        "open": [50, 55, 60, 65, 70, 70], "high": [50, 56, 61, 66, 92, 92],
        "low": [50, 54, 59, 64, 69, 69], "close": [50, 55, 60, 65, 90, 90],
    }, index=idx_t)
    exit_p2, reason2, _ = simulate_trade(prem_target, 0, entry_price=50, stop_level=30, target_level=90)
    assert reason2 == "TARGET" and abs(exit_p2 - 90.0) < 1e-6
    print("  simulate_trade clean TARGET-hit check ok")

    # ambiguous same-bar case -- must resolve to STOP (conservative), even though the bar's
    # high would also separately reach the target
    prem_amb = pd.DataFrame({
        "open": [50, 50, 50, 50, 50, 50], "high": [50, 52, 95, 60, 60, 60],
        "low": [50, 48, 40, 58, 58, 58], "close": [50, 48, 60, 60, 60, 60],
    }, index=idx_t)
    exit_p3, reason3, _ = simulate_trade(prem_amb, 0, entry_price=50, stop_level=42, target_level=90)
    assert reason3 == "STOP" and abs(exit_p3 - 42.0) < 1e-6, f"expected conservative STOP@42, got {exit_p3},{reason3}"
    print("  simulate_trade conservative stop-before-target ordering check ok")

    # --- multi-day previous-close tracking, hand-traced ---
    idx_day1 = pd.date_range("2025-06-02 09:15", periods=5, freq="2min")
    idx_day2 = pd.date_range("2025-06-03 09:15", periods=5, freq="2min")
    day1 = pd.DataFrame({"open": [22000] * 5, "high": [22005] * 5, "low": [21995] * 5,
                          "close": [22000, 22010, 22015, 22012, 22018]}, index=idx_day1)
    day2_open = 22018 + 30   # gap up by 30pts vs day1's actual last close (22018)
    day2 = pd.DataFrame({"open": [day2_open] * 5, "high": [day2_open + 5] * 5, "low": [day2_open - 5] * 5,
                          "close": [day2_open] * 5}, index=idx_day2)
    prev_close = day1["close"].iloc[-1]
    direction, gap = detect_gap_signal(prev_close, day2["open"].iloc[0])
    assert abs(prev_close - 22018) < 1e-6 and direction == "SHORT" and abs(gap - 30) < 1e-6
    print("  multi-day previous-close tracking + gap computation check ok")

    print("All self-checks passed.\n")


def compute_metrics(trades_df, starting_capital=STARTING_CAPITAL):
    if trades_df.empty:
        return {"n": 0, "win_pct": 0.0, "pf": 0.0, "net_rs": 0.0, "max_dd_rs": 0.0, "sqn": 0.0, "cagr_pct": 0.0}
    pnl = trades_df["rupee_pnl"].to_numpy()
    n = len(pnl)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    win_pct = 100.0 * len(wins) / n
    gross_win, gross_loss = wins.sum(), abs(losses.sum())
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    net_rs = pnl.sum()
    equity = starting_capital + np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    max_dd_rs = (peak - equity).max()
    sqn = (pnl.mean() / pnl.std() * math.sqrt(n)) if n > 1 and pnl.std() > 0 else 0.0
    dates = pd.to_datetime(trades_df["date"])
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    cagr_pct = 0.0
    if years > 0 and (starting_capital + net_rs) > 0:
        cagr_pct = 100.0 * (((starting_capital + net_rs) / starting_capital) ** (1.0 / years) - 1.0)
    return {"n": n, "win_pct": round(win_pct, 1), "pf": round(pf, 2) if pf != float("inf") else pf,
            "net_rs": round(net_rs, 2), "max_dd_rs": round(max_dd_rs, 2), "sqn": round(sqn, 2),
            "cagr_pct": round(cagr_pct, 2)}


def main():
    run_self_checks()
    print("=== ORIGINAL STRATEGY: Nifty Overnight Gap-Fill Fade ===\n")
    print(f"Gap threshold: {GAP_THRESHOLD_PTS}pts   Stop: {STOP_PTS}pts (premium)   "
          f"Target: full fill to yesterday's close   Starting capital: Rs {STARTING_CAPITAL:,}\n")

    print(f"Fetching Nifty index data ({FROM_DATE} to {TO_DATE})...")
    try:
        raw = fetch_index_chunked(FROM_DATE, TO_DATE)
    except Exception as e:
        print(f"\nCould not fetch real data: {e}\n")
        print("All self-checks above still passed -- this failure is about reaching the Dhan API from "
              "wherever this just ran. If you're seeing this on your own machine with a working Dhan "
              "connection, check your network/ACCESS_TOKEN and retry.")
        return
    df2m = resample_2min_per_day(raw)
    days = sorted(set(df2m.index.date))
    total_days = len(days)
    print(f"Got {len(df2m)} 2-min bars across {total_days} trading day(s): {days[0]} to {days[-1]}\n")

    trades = []
    n_days_with_gap = 0
    prev_close = None
    for day in days:
        day_df = df2m[df2m.index.date == day]
        if len(day_df) < 5:
            prev_close = day_df["close"].iloc[-1] if not day_df.empty else prev_close
            continue
        day_open_spot = day_df["open"].iloc[0]

        if prev_close is not None:
            direction, gap = detect_gap_signal(prev_close, day_open_spot)
            if direction is not None:
                n_days_with_gap += 1
                exec_time_cutoff = (day_df.index[0] + dt.timedelta(minutes=SKIP_MINUTES)).time()
                after_skip = day_df[day_df.index.time >= exec_time_cutoff]
                if not after_skip.empty:
                    exec_idx = day_df.index.get_loc(after_skip.index[0])
                    side = "CE" if direction == "LONG" else "PE"
                    strike = itm_call_strike(day_open_spot) if side == "CE" else itm_put_strike(day_open_spot)
                    premium_df = build_premium_series(day_df, strike, day_open_spot, side)
                    entry_price = premium_df["open"].iloc[exec_idx]
                    stop_level = entry_price - STOP_PTS
                    target_level = premium_at_index_level(prev_close, day_open_spot, strike, side)

                    if target_level > entry_price and stop_level > 0:
                        exit_price, reason, exit_time = simulate_trade(premium_df, exec_idx, entry_price, stop_level, target_level)
                        lot = lot_size_for_date(day)
                        pnl_pts = exit_price - entry_price
                        rupee_pnl = pnl_pts * lot - BROKERAGE_PER_TRADE
                        rr = (target_level - entry_price) / STOP_PTS
                        trades.append({
                            "date": day, "direction": direction, "side": side, "strike": strike, "gap_pts": round(gap, 2),
                            "entry_time": day_df.index[exec_idx], "exit_time": exit_time,
                            "entry_price": round(entry_price, 2), "exit_price": round(exit_price, 2),
                            "exit_reason": reason, "pnl_pts": round(pnl_pts, 2), "rupee_pnl": round(rupee_pnl, 2),
                            "rr_planned": round(rr, 2),
                        })
        prev_close = day_df["close"].iloc[-1]

    print(f"Days with a qualifying gap (>= {GAP_THRESHOLD_PTS}pts): {n_days_with_gap} / {total_days} "
          f"({100*n_days_with_gap/total_days:.1f}%)\n")

    if not trades:
        print("No trades generated.")
        return

    tdf = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
    tdf["running_equity"] = STARTING_CAPITAL + tdf["rupee_pnl"].cumsum()
    tdf.to_csv("nifty_gapfill_fade_trades.csv", index=False)

    m = compute_metrics(tdf)
    print("--- Overall ---")
    print(f"n={m['n']}  win%={m['win_pct']}  PF={m['pf']}  net_Rs={m['net_rs']:,.2f}  "
          f"maxDD_Rs={m['max_dd_rs']:,.2f}  SQN={m['sqn']}  CAGR%={m['cagr_pct']}\n")

    print("--- By direction ---")
    for d, g in tdf.groupby("direction"):
        mm = compute_metrics(g)
        print(f"  [{d}] n={mm['n']} win%={mm['win_pct']} PF={mm['pf']} net_Rs={mm['net_rs']:,.2f}")

    print("\n--- By exit reason ---")
    for reason, g in tdf.groupby("exit_reason"):
        print(f"  [{reason}] n={len(g)} net_Rs={g['rupee_pnl'].sum():,.2f}")

    dates = pd.to_datetime(tdf["date"])
    mid = dates.min() + (dates.max() - dates.min()) / 2
    fh, sh = tdf[dates <= mid], tdf[dates > mid]
    m1, m2 = compute_metrics(fh), compute_metrics(sh)
    print(f"\n--- Stability check ---\nFirst half: n={m1['n']} net_Rs={m1['net_rs']:,.2f} PF={m1['pf']} SQN={m1['sqn']}  |  "
          f"Second half: n={m2['n']} net_Rs={m2['net_rs']:,.2f} PF={m2['pf']} SQN={m2['sqn']}")

    if m['pf'] != float("inf") and (m['pf'] > 2.5 or m['sqn'] > 5):
        print("\n*** RED FLAG: PF>2.5 or SQN>5 -- treat as an extraordinary claim requiring extra scrutiny "
              "before trusting it, per this project's standing calibration (Velez 8pt benchmark: PF~1.5-1.7, "
              "SQN~2.5-3). ***")

    print(f"\nSaved: nifty_gapfill_fade_trades.csv ({len(tdf)} rows)")


if __name__ == "__main__":
    main()
