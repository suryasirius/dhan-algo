"""
Compression Spike (Concept 2 ALONE) -- 8pt stop / 5pt target on the OPTION
PREMIUM, or exit after 3 minutes -- replaces the fixed 1:2 R:R
=================================================================
Per your instruction: remove the old fixed 1:2 risk:reward (5pt SL / 10pt
TP measured on raw Nifty index points -- dhan_live_bot.py's current live
setting). Replace with:
  - Stop:   -8 points on the OPTION PREMIUM itself (not the index)
  - Target: +5 points on the OPTION PREMIUM itself (not the index)
  - Time exit: if neither hits within 3 minutes (3 one-minute bars) of
    entry, close the trade at whatever the premium is at that point

This is a real change in modeling, not just new numbers: every previous
concepts_2_4_* script in this project (including
concept2_compression_spike_optimize.py) measured SL/TP in raw Nifty INDEX
points, scaled by a flat assumed delta only at the very end to convert to
rupees. This script instead builds the actual proxy premium series (same
model used by Velez/Opening-Drive Momentum/Crash Play elsewhere in this
project: intrinsic-at-open + assumed time value, moves scaled by delta,
floored at 0) and applies the stop/target DIRECTLY to that premium series,
because you asked for "-8 points... in the option chart" -- i.e. on the
premium, not the underlying. Flagged explicitly since it changes what the
numbers mean versus every earlier concept-2/4 result in the playbook.

Concept-2 (compression spike) always fires on the SELL side (per
dhan_live_bot.py's own concept definitions), so this always buys the ITM
PUT -- matches the project's "always buy, never write options" convention.

Entry timing: kept identical to the existing concepts_2_4_* family and to
how dhan_live_bot.py itself actually trades live -- the position opens at
the CLOSE of the same bar that produced the signal (dlb.compute_latest_signal
returning final_sell on bar i means the live bot would act on that
signal immediately at that live price). This is different from the
1-bar-execution-lag convention used in Velez/Opening-Drive Momentum
(which fixed a real look-ahead bug there) -- it's kept here because it
matches this specific bot's actual live execution behavior, not because
it was overlooked; noted so the two conventions aren't confused with each
other.

Conservative ordering, unchanged from every other script here: within any
given bar, the STOP is checked before the TARGET.

Reuses dlb.compute_latest_signal() -- the exact real, unedited live-bot
signal logic -- via the same parallelized, HTF-context-window-correct
signal collection already proven in concept2_compression_spike_optimize.py,
restricted to concept_id==2 and the already-validated Tuesday/Wednesday
weekday filter.
"""

import datetime as dt
import math
import os
import traceback
import multiprocessing as mp
import pandas as pd
import numpy as np

import dhan_live_bot as dlb

FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

WARMUP_BARS = 30
TARGET_CONCEPT_ID = 2
TARGET_WEEKDAYS = {1, 2}   # Tuesday, Wednesday -- already-validated subset

STOP_PTS = 8.0     # on the PREMIUM
TARGET_PTS = 5.0   # on the PREMIUM
MAX_HOLD_MINUTES = 3

STRIKE_STEP = 50
STARTING_CAPITAL = 10_000
ASSUMED_DELTA = 0.6
ASSUMED_TIME_VALUE = 40.0
LOT_SIZE = 65
BROKERAGE_PER_TRADE = 40.0


def itm_put_strike(spot):
    return math.ceil(spot / STRIKE_STEP) * STRIKE_STEP


def build_pe_premium_series(day_df, strike, day_open_spot):
    idx_open = day_df["open"].to_numpy()
    idx_high = day_df["high"].to_numpy()
    idx_low = day_df["low"].to_numpy()
    idx_close = day_df["close"].to_numpy()
    intrinsic0 = max(strike - day_open_spot, 0.0)
    base = intrinsic0 + ASSUMED_TIME_VALUE
    p_open = np.maximum(base + (day_open_spot - idx_open) * ASSUMED_DELTA, 0.0)
    p_high = np.maximum(base + (day_open_spot - idx_low) * ASSUMED_DELTA, 0.0)    # index LOW is best for a PE
    p_low = np.maximum(base + (day_open_spot - idx_high) * ASSUMED_DELTA, 0.0)    # index HIGH is worst for a PE
    p_close = np.maximum(base + (day_open_spot - idx_close) * ASSUMED_DELTA, 0.0)
    return pd.DataFrame({"open": p_open, "high": p_high, "low": p_low, "close": p_close}, index=day_df.index)


def extract_signal(sig):
    if sig.get("final_buy"):
        return "LONG", sig.get("concept_id")
    if sig.get("final_sell"):
        return "SHORT", sig.get("concept_id")
    return None


def get_htf_context_days():
    cd = getattr(dlb, "HTF_CONTEXT_DAYS", None)
    return cd if cd is not None else 5


def build_context_window_days(all_trading_days_sorted, target_day, context_days):
    idx_of_target = all_trading_days_sorted.index(target_day)
    start_idx = max(0, idx_of_target - context_days + 1)
    return all_trading_days_sorted[start_idx:idx_of_target + 1]


def simulate_premium_trade(premium_df, entry_bar, stop_pts=STOP_PTS, target_pts=TARGET_PTS,
                            max_hold_minutes=MAX_HOLD_MINUTES):
    """Entry fills at the signal bar's own CLOSE (see docstring -- matches
    this bot's real live-execution behavior). Conservative ordering: STOP
    checked before TARGET within any bar. If neither hits within
    max_hold_minutes bars (1-min bars, so this literally is minutes), exit
    at that bar's close."""
    closes = premium_df["close"].tolist()
    highs = premium_df["high"].tolist()
    lows = premium_df["low"].tolist()
    idx = premium_df.index
    n = len(premium_df)
    entry_price = closes[entry_bar]
    stop_level = entry_price - stop_pts
    target_level = entry_price + target_pts

    last_bar = min(entry_bar + max_hold_minutes, n - 1)
    for j in range(entry_bar + 1, last_bar + 1):
        if lows[j] <= stop_level:
            return {"entry_time": idx[entry_bar], "exit_time": idx[j], "entry_price": entry_price,
                    "exit_price": stop_level, "exit_reason": "STOP", "points": stop_level - entry_price}
        if highs[j] >= target_level:
            return {"entry_time": idx[entry_bar], "exit_time": idx[j], "entry_price": entry_price,
                    "exit_price": target_level, "exit_reason": "TARGET", "points": target_level - entry_price}
        if j == last_bar:
            return {"entry_time": idx[entry_bar], "exit_time": idx[j], "entry_price": entry_price,
                    "exit_price": closes[j], "exit_reason": "TIME_3MIN", "points": closes[j] - entry_price}
    # entry bar was the last bar available in the day -- exit immediately at entry price (no bars left)
    return {"entry_time": idx[entry_bar], "exit_time": idx[entry_bar], "entry_price": entry_price,
            "exit_price": entry_price, "exit_reason": "NO_BARS_LEFT", "points": 0.0}


def run_self_checks():
    print("Running self-checks (synthetic data)...")
    assert extract_signal({"final_buy": True, "final_sell": False, "concept_id": 2}) == ("LONG", 2)
    assert extract_signal({"final_buy": False, "final_sell": True, "concept_id": 2}) == ("SHORT", 2)
    print("  extract_signal check ok")

    days = [dt.date(2025, 1, d) for d in range(1, 11)]
    w = build_context_window_days(days, dt.date(2025, 1, 10), 5)
    assert w == [dt.date(2025, 1, 6), dt.date(2025, 1, 7), dt.date(2025, 1, 8),
                 dt.date(2025, 1, 9), dt.date(2025, 1, 10)]
    print("  build_context_window_days check ok")

    p = build_pe_premium_series(
        pd.DataFrame({"open": [22000], "high": [22005], "low": [21995], "close": [22000]},
                     index=pd.date_range("2025-01-06 09:15", periods=1, freq="1min")),
        strike=itm_put_strike(22000), day_open_spot=22000,
    )
    # strike=22000, intrinsic0=0, base=40; close move=22000-22000=0 -> premium=40
    assert abs(p["close"].iloc[0] - 40.0) < 1e-6
    print("  build_pe_premium_series check ok")

    idx_t = pd.date_range("2025-01-06 09:15", periods=6, freq="1min")

    # clean STOP case: premium bleeds down past entry-8 within 3 bars
    prem_stop = pd.DataFrame({
        "open": [50, 48, 44, 40, 40, 40], "high": [50, 49, 45, 41, 41, 41],
        "low": [50, 47, 43, 39, 39, 39], "close": [50, 48, 44, 40, 40, 40],
    }, index=idx_t)
    t = simulate_premium_trade(prem_stop, 0)
    assert t["exit_reason"] == "STOP" and abs(t["exit_price"] - 42.0) < 1e-6
    print("  simulate_premium_trade clean STOP check ok")

    # clean TARGET case: premium jumps up past entry+5 within 3 bars
    prem_target = pd.DataFrame({
        "open": [50, 53, 56, 60, 60, 60], "high": [50, 54, 57, 61, 61, 61],
        "low": [50, 52, 55, 59, 59, 59], "close": [50, 53, 56, 60, 60, 60],
    }, index=idx_t)
    t2 = simulate_premium_trade(prem_target, 0)
    assert t2["exit_reason"] == "TARGET" and abs(t2["exit_price"] - 55.0) < 1e-6
    print("  simulate_premium_trade clean TARGET check ok")

    # 3-minute timeout: neither stop nor target hit within 3 bars -- exit at that bar's close
    prem_time = pd.DataFrame({
        "open": [50, 51, 52, 53, 53, 53], "high": [50, 52, 53, 54, 54, 54],
        "low": [50, 50, 51, 52, 52, 52], "close": [50, 51, 52, 53, 53, 53],
    }, index=idx_t)
    t3 = simulate_premium_trade(prem_time, 0)
    assert t3["exit_reason"] == "TIME_3MIN" and abs(t3["exit_price"] - 53.0) < 1e-6, f"got {t3}"
    print("  simulate_premium_trade 3-minute timeout check ok")

    # conservative ordering: same bar breaches BOTH stop and target -- must resolve STOP
    prem_amb = pd.DataFrame({
        "open": [50, 50, 50, 50, 50, 50], "high": [50, 51, 70, 60, 60, 60],
        "low": [50, 49, 40, 58, 58, 58], "close": [50, 49, 55, 55, 55, 55],
    }, index=idx_t)
    t4 = simulate_premium_trade(prem_amb, 0)
    assert t4["exit_reason"] == "STOP" and abs(t4["exit_price"] - 42.0) < 1e-6, f"expected conservative STOP@42, got {t4}"
    print("  simulate_premium_trade conservative stop-before-target ordering check ok")

    print("All self-checks passed.\n")


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
            "timestamp": data["timestamp"],
        })
        df["time"] = pd.to_datetime(df["timestamp"], unit="s") + pd.Timedelta(hours=5, minutes=30)
        chunks.append(df.set_index("time").sort_index())
        cur = chunk_end + dt.timedelta(days=1)
    full = pd.concat(chunks)
    return full[~full.index.duplicated(keep="first")].sort_index()


def _scan_one_day(args):
    target_day, window_df = args
    today_positions = [i for i, d in enumerate(window_df.index.date) if d == target_day]
    if not today_positions:
        return [], None
    first_bar_today, last_bar_today = today_positions[0], today_positions[-1]
    if (last_bar_today - first_bar_today) <= WARMUP_BARS + 2:
        return [], None

    found, first_error, open_bar = [], None, -1
    for i in today_positions:
        if i - first_bar_today < WARMUP_BARS or i <= open_bar:
            continue
        try:
            sig = dlb.compute_latest_signal(window_df.iloc[:i + 1])
        except Exception:
            if first_error is None:
                first_error = traceback.format_exc()
            continue
        if sig is None:
            continue
        result = extract_signal(sig)
        if result is None:
            continue
        side, concept = result
        found.append((i, side, concept))
        open_bar = i + MAX_HOLD_MINUTES   # a trade now only ties up 3 minutes, not dlb.MAX_BARS_IN_TRADE
    return found, first_error


def collect_concept2_signals(df, days_covered):
    context_days = get_htf_context_days()
    target_days = [d for d in days_covered if d.weekday() in TARGET_WEEKDAYS]
    print(f"Scanning Tuesday/Wednesday only: {len(target_days)} of {len(days_covered)} days.")
    print(f"Each day gets a {context_days}-trading-day rolling context window before scanning.\n")

    jobs = []
    for target_day in target_days:
        window_days = build_context_window_days(days_covered, target_day, context_days)
        window_df = df[(df.index.date >= window_days[0]) & (df.index.date <= window_days[-1])]
        jobs.append((target_day, window_df))

    n_workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"Scanning {len(target_days)} days across {n_workers} worker processes...")
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_scan_one_day, jobs)

    all_signals, errors = [], []
    for (target_day, window_df), (found, err) in zip(jobs, results):
        day_df = window_df[window_df.index.date == target_day]
        for (i_window, side, concept) in found:
            if concept != TARGET_CONCEPT_ID:
                continue
            i_day = i_window - window_df.index.get_loc(day_df.index[0])
            all_signals.append((target_day, day_df, i_day, side))
        if err:
            errors.append((target_day, err))

    if errors:
        print(f"\n{len(errors)} of {len(target_days)} days hit an error. First error (from {errors[0][0]}):\n"
              f"{errors[0][1]}\nPaste this back if you see it.\n")

    return all_signals


def compute_metrics(tdf, starting_capital=STARTING_CAPITAL):
    if tdf.empty:
        return {"n": 0, "win_pct": 0.0, "pf": 0.0, "net_rs": 0.0, "max_dd_rs": 0.0, "sqn": 0.0, "cagr_pct": 0.0}
    wins, losses = tdf[tdf["points"] > 0], tdf[tdf["points"] <= 0]
    win_pct = 100.0 * len(wins) / len(tdf)
    pf = (wins["points"].sum() / abs(losses["points"].sum())) if losses["points"].sum() != 0 else float("inf")
    equity = starting_capital + tdf["rupee_pnl"].cumsum()
    peak = equity.cummax()
    max_dd_rs = (peak - equity).max()
    r_multiple = tdf["rupee_pnl"] / (STOP_PTS * LOT_SIZE)
    sqn = (r_multiple.mean() / r_multiple.std()) * math.sqrt(len(tdf)) if r_multiple.std() else 0.0
    dates = pd.to_datetime(tdf["date"])
    years = (dates.max() - dates.min()).days / 365.25
    net_rs = tdf["rupee_pnl"].sum()
    cagr_pct = 0.0
    if years > 0 and (starting_capital + net_rs) > 0:
        cagr_pct = 100.0 * (((starting_capital + net_rs) / starting_capital) ** (1.0 / years) - 1.0)
    return {"n": len(tdf), "win_pct": round(win_pct, 1), "pf": round(pf, 2) if pf != float("inf") else pf,
            "net_rs": round(net_rs, 2), "max_dd_rs": round(max_dd_rs, 2), "sqn": round(sqn, 2),
            "cagr_pct": round(cagr_pct, 2)}


def main():
    run_self_checks()
    print("=== Compression Spike (Concept 2 ALONE) -- 8pt STOP / 5pt TARGET on the PREMIUM, or 3-min exit ===\n")
    print(f"Stop: -{STOP_PTS}pts (premium)   Target: +{TARGET_PTS}pts (premium)   Max hold: {MAX_HOLD_MINUTES} minutes\n")

    print("Fetching Nifty index 1-min data (2 years)...")
    df = fetch_index_chunked(FROM_DATE, TO_DATE)
    days_covered = sorted(set(df.index.date))
    print(f"Got {len(df)} candles across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}\n")

    signals = collect_concept2_signals(df, days_covered)
    if not signals:
        print("No concept-2 signals collected on Tue/Wed. If an error was printed above, paste it back.")
        return
    print(f"Total concept-2 (compression spike) signals on Tue/Wed: {len(signals)}\n")

    trades = []
    for target_day, day_df, entry_bar, side in signals:
        day_open_spot = day_df["open"].iloc[0]
        strike = itm_put_strike(day_open_spot)   # concept 2 is always a SELL signal -> always buys the PUT
        premium_df = build_pe_premium_series(day_df, strike, day_open_spot)
        t = simulate_premium_trade(premium_df, entry_bar)
        lot = LOT_SIZE
        rupee_pnl = t["points"] * lot - BROKERAGE_PER_TRADE
        trades.append({
            "date": target_day, "strike": strike, "entry_time": t["entry_time"], "exit_time": t["exit_time"],
            "entry_price": round(t["entry_price"], 2), "exit_price": round(t["exit_price"], 2),
            "exit_reason": t["exit_reason"], "points": round(t["points"], 2), "rupee_pnl": round(rupee_pnl, 2),
        })

    tdf = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
    m = compute_metrics(tdf)
    print("--- Overall ---")
    print(f"n={m['n']}  win%={m['win_pct']}  PF={m['pf']}  net_Rs={m['net_rs']:,.2f}  "
          f"maxDD_Rs={m['max_dd_rs']:,.2f}  SQN={m['sqn']}  CAGR%={m['cagr_pct']}\n")

    print("--- By exit reason ---")
    for reason, g in tdf.groupby("exit_reason"):
        print(f"  [{reason}] n={len(g)} ({100*len(g)/len(tdf):.1f}%) net_Rs={g['rupee_pnl'].sum():,.2f} "
              f"avg_pts={g['points'].mean():.2f}")

    dates = pd.to_datetime(tdf["date"])
    mid = dates.min() + (dates.max() - dates.min()) / 2
    fh, sh = tdf[dates <= mid], tdf[dates > mid]
    m1, m2 = compute_metrics(fh), compute_metrics(sh)
    print(f"\n--- Stability check ---\nFirst half:  n={m1['n']} net_Rs={m1['net_rs']:,.2f} PF={m1['pf']} SQN={m1['sqn']}")
    print(f"Second half: n={m2['n']} net_Rs={m2['net_rs']:,.2f} PF={m2['pf']} SQN={m2['sqn']}")

    if m["pf"] != float("inf") and (m["pf"] > 2.5 or m["sqn"] > 5):
        print("\n*** RED FLAG: PF>2.5 or SQN>5 -- treat as an extraordinary claim requiring extra scrutiny "
              "before trusting it, per this project's standing calibration. ***")

    tdf.to_csv("concept2_compression_spike_8sl_5tp_3min_trades.csv", index=False)
    print(f"\nSaved: concept2_compression_spike_8sl_5tp_3min_trades.csv ({len(tdf)} rows)")
    print(f"\nFor reference, the OLD fixed-1:2-RR result (index points, not premium) was: "
          f"n=170+134=304 (Tue+Wed), net +454.20 index points combined. Compare this new premium-based "
          f"result against that directly, keeping in mind they're measured in different units (index points "
          f"then vs premium points now) -- net_Rs and PF are the fair comparison, not raw points.")


if __name__ == "__main__":
    main()
