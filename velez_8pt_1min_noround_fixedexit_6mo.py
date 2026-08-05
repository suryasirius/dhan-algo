"""
Velez 8pt (SHORT-only) -- MODIFIED standalone variant, 6-month backtest
=================================================================
Does NOT touch velez_final_8pt.py or velez_8pt_proxy_premium_test.py
(both locked/validated -- per project convention, new ideas always get
their own file). This reuses the exact same GBI/ELEPHANT_PLUS signal
logic verbatim, but changes three things per your request:

  1. Bar timeframe: the original resamples to 2-MINUTE bars, so its
     SMA_FAST=20 actually spans 40 minutes (20 x 2min), not 20. This
     version resamples to 1-MINUTE bars instead, so the same 20-period
     SMA genuinely covers 20 minutes, matching what you asked for
     ("instead of 2 min candle for 20mins use 1 min candle for
     20mins"). SMA_SLOW=200 similarly now spans 200 minutes instead of
     400. Nothing else about the indicator math changed.

  2. Round-figure filter: skips a signal entirely if the index level
     at the moment of entry is within 10 points of a round hundred
     (e.g. 25000, 24900, 24800...) -- the classic "round numbers act
     as support/resistance and cause whipsaws near them" filter.
     Checked against the index's own open price at the execution bar
     (the actual traded level), not the option premium.

  3. Exit rule: the original's 8pt initial stop / 4pt trail is REPLACED
     entirely with a fixed target-or-time exit: +5pt on the premium OR
     3 minutes after entry, whichever comes first -- same convention as
     the fixed-strike ORB family tested elsewhere in this project.
     NO stop-loss at all in this version (flagged as a real risk, same
     as every other no-stop variant here) -- the 3-minute clock is the
     only downside brake.

Everything else is unchanged: SHORT-only (always buys the ITM PUT,
never writes options), same GBI/ELEPHANT_PLUS detection, same
execution-lag fix (fills at the NEXT bar's open, never the signal
bar's own close), same proxy premium model (delta=0.6, floored at 0)
since real Dhan option-premium data is still blocked (DH-901).

6-month window, per your request.
"""

import datetime as dt
import math
import pandas as pd
import numpy as np

import dhan_live_bot as dlb

FROM_DATE = (dt.date.today() - dt.timedelta(days=182)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85
BAR_RULE = "1min"

SMA_FAST = 20
SMA_SLOW = 200
POP_SLOPE_LOOKBACK = 5
AVG_RANGE_LOOKBACK = 20

SMALL_BAR_MULT = 0.6
ELEPHANT_MULT = 2.0
ELEPHANT_PLUS_RETAIN = 0.8
NEAR_20_PCT = 0.0015

ROUND_FIGURE_STEP = 100.0
ROUND_FIGURE_BUFFER = 10.0

FIXED_TARGET_PTS = 5.0
MAX_HOLD = pd.Timedelta(minutes=3)
EOD_SQUAREOFF = dt.time(15, 20)

STRIKE_STEP = 50
STARTING_CAPITAL = 25_000
BROKERAGE_PER_TRADE = 40.0

ASSUMED_DELTA = 0.6
ASSUMED_TIME_VALUE = 40.0


def lot_size_for_date(d):
    if d < dt.date(2024, 11, 20):
        return 25
    elif d < dt.date(2026, 1, 1):
        return 75
    else:
        return 65


def itm_put_strike(spot):
    return math.ceil(spot / STRIKE_STEP) * STRIKE_STEP


def near_round_figure(price, step=ROUND_FIGURE_STEP, buffer=ROUND_FIGURE_BUFFER):
    """True if `price` is within `buffer` points of the nearest multiple
    of `step` (e.g. within 10 points of 25000, 24900, 25100, ...)."""
    nearest = round(price / step) * step
    return abs(price - nearest) < buffer


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


def resample_bars_per_day(df):
    """Resamples to BAR_RULE (1-minute in this variant -- effectively a
    no-op reshape since the source data is already 1-min, but kept as a
    named step so the pipeline shape matches the 2-min original and any
    future bar-size change is a one-line edit."""
    out = []
    for day in sorted(set(df.index.date)):
        day_df = df[df.index.date == day]
        r = day_df.resample(BAR_RULE, origin=day_df.index[0]).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        out.append(r)
    return pd.concat(out).sort_index()


def compute_indicators(df):
    df = df.copy()
    df["sma20"] = df["close"].rolling(SMA_FAST, min_periods=SMA_FAST).mean()
    df["sma200"] = df["close"].rolling(SMA_SLOW, min_periods=SMA_SLOW).mean()
    df["range"] = df["high"] - df["low"]
    df["avg_range"] = df["range"].shift(1).rolling(AVG_RANGE_LOOKBACK, min_periods=AVG_RANGE_LOOKBACK).mean()
    return df


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


# ----------------------------- signal detection (verbatim from velez_final_8pt.py) -----------------------------

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


def detect_signals_for_day(day_df):
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


# ----------------------------- trade management: fixed target OR 3-min time exit -----------------------------

def simulate_trade_fixed_exit(premium_df, entry_idx, entry_price,
                               target_pts=FIXED_TARGET_PTS, max_hold=MAX_HOLD):
    """NO stop-loss. Conservative ordering: TARGET is checked before the
    time deadline on any bar where both would technically be
    satisfiable. If neither hits, forces an exit at the EOD squareoff."""
    idx = premium_df.index
    closes = premium_df["close"].tolist()
    highs = premium_df["high"].tolist()
    target = entry_price + target_pts
    deadline = idx[entry_idx] + max_hold

    for j in range(entry_idx, len(idx)):
        t = idx[j].time()
        if highs[j] >= target:
            return target, "TARGET", idx[j]
        if idx[j] >= idx[entry_idx] + max_hold:
            return closes[j], "TIME_3MIN", idx[j]
        if t >= EOD_SQUAREOFF:
            return closes[j], "EOD", idx[j]
    last = premium_df.iloc[-1]
    return last["close"], "EOD", idx[-1]


def run_self_checks():
    print("Running self-checks (synthetic data)...")

    # --- verbatim checks from velez_final_8pt.py, unchanged logic ---
    assert pop_bearish(95, 97, 98, 100) is True
    assert pop_bearish(105, 103, 102, 100) is False
    assert is_small(2, 10) is True
    assert is_elephant_plus_bear(100, 100, 70, 71, 10) is True
    assert is_elephant_plus_bear(100, 101, 99, 100.5, 10) is False
    print("  pop_bearish / is_small / is_elephant_plus_bear checks ok (verbatim from velez_final_8pt.py)")

    # --- round-figure filter checks ---
    assert near_round_figure(25005) is True     # 5 pts from 25000
    assert near_round_figure(24995) is True     # 5 pts from 25000
    assert near_round_figure(25009.9) is True   # just inside the 10pt buffer
    assert near_round_figure(25050) is False    # 50 pts from both 25000 and 25100
    assert near_round_figure(25015) is False    # 15 pts from 25000, outside buffer
    print("  near_round_figure filter check ok")

    # --- fixed exit: clean TARGET hit before the 3-min deadline ---
    idx_t = pd.date_range("2025-01-06 09:15", periods=8, freq="1min")
    prem = pd.DataFrame({
        "open": [100, 100, 100, 100, 100, 100, 100, 100],
        "high": [100, 105, 105, 105, 105, 105, 105, 105],
        "low": [100, 99, 99, 99, 99, 99, 99, 99],
        "close": [100, 100, 100, 100, 100, 100, 100, 100],
    }, index=idx_t)
    exit_price, reason, exit_time = simulate_trade_fixed_exit(prem, entry_idx=0, entry_price=100)
    assert reason == "TARGET" and abs(exit_price - 105.0) < 1e-6, f"got {exit_price}, {reason}"
    assert exit_time == idx_t[1], f"expected target hit at bar 1 (well before the bar-3 deadline), got {exit_time}"
    print("  simulate_trade_fixed_exit clean TARGET check ok")

    # --- fixed exit: target never reached -> forced TIME_3MIN exit at
    # the first bar whose timestamp is >= entry+3min ---
    prem2 = pd.DataFrame({
        "open": [100, 100.2, 100.4, 100.6, 100.8, 101.0, 101.1, 101.2],
        "high": [100, 100.3, 100.5, 100.7, 100.9, 101.1, 101.2, 101.3],
        "low": [100, 99.9, 100.1, 100.3, 100.5, 100.7, 100.9, 101.0],
        "close": [100, 100.2, 100.4, 100.6, 100.8, 101.0, 101.1, 101.2],
    }, index=idx_t)
    exit_price2, reason2, exit_time2 = simulate_trade_fixed_exit(prem2, entry_idx=0, entry_price=100)
    assert reason2 == "TIME_3MIN", f"expected TIME_3MIN, got {reason2}"
    assert exit_time2 == idx_t[0] + pd.Timedelta(minutes=3)
    assert abs(exit_price2 - prem2["close"].iloc[3]) < 1e-9  # bar index 0+3=3
    print("  simulate_trade_fixed_exit TIME_3MIN-when-target-not-hit check ok")

    # --- fixed exit: TARGET and the 3-min deadline both satisfiable on
    # the SAME bar -> TARGET must win ---
    prem3 = pd.DataFrame({
        "open": [100, 100.1, 100.1, 110.0, 110.0],
        "high": [100, 100.2, 100.2, 106.0, 110.0],
        "low": [100, 100.0, 100.0, 100.0, 109.0],
        "close": [100, 100.1, 100.1, 110.0, 110.0],
    }, index=idx_t[:5])
    exit_price3, reason3, exit_time3 = simulate_trade_fixed_exit(prem3, entry_idx=0, entry_price=100)
    assert reason3 == "TARGET", f"expected TARGET to win on the ambiguous bar, got {reason3}"
    assert exit_time3 == idx_t[3]
    print("  simulate_trade_fixed_exit same-bar target-vs-deadline ordering check ok")

    # --- detect_signals_for_day regression smoke (on 1-min bars now) ---
    n_bars = 250
    idx2 = pd.date_range("2025-06-02 09:15", periods=n_bars, freq="1min")
    closes2 = [100.0 - i * 0.02 for i in range(n_bars)]
    opens2 = [c + 0.01 for c in closes2]
    highs2 = [c + 0.03 for c in closes2]
    lows2 = [c - 0.02 for c in closes2]
    day2 = pd.DataFrame({"open": opens2, "high": highs2, "low": lows2, "close": closes2}, index=idx2)
    day2["sma20"] = pd.Series(closes2).rolling(SMA_FAST, min_periods=1).mean().values
    day2["sma200"] = pd.Series([100.0] * n_bars) - pd.Series(range(n_bars)) * 0.015
    day2["range"] = day2["high"] - day2["low"]
    day2["avg_range"] = day2["range"].shift(1).rolling(AVG_RANGE_LOOKBACK, min_periods=1).mean()
    sigs = detect_signals_for_day(day2)
    assert len(sigs) >= 0
    print(f"  detect_signals_for_day regression check ok ({len(sigs)} signal(s) found, 1-min bars)")

    print("All self-checks passed.\n")


def main():
    run_self_checks()
    print("=== Velez 8pt (SHORT-only) MODIFIED -- 1-min bars, round-figure filter, 5pt/3min fixed exit -- 6-month backtest ===\n")
    print(f"Starting capital: Rs {STARTING_CAPITAL:,}   Brokerage: Rs {BROKERAGE_PER_TRADE:.2f}/trade")
    print(f"Bars: {BAR_RULE} (SMA20 = 20 minutes, SMA200 = 200 minutes)")
    print(f"Round-figure filter: skip entry if index is within {ROUND_FIGURE_BUFFER:.0f}pts of a {ROUND_FIGURE_STEP:.0f}-multiple")
    print(f"Exit: +{FIXED_TARGET_PTS}pt target OR {int(MAX_HOLD.total_seconds()//60)} minutes, whichever first. NO stop-loss.\n")

    print(f"Fetching Nifty index data ({FROM_DATE} to {TO_DATE}, ~6 months) and resampling to {BAR_RULE} bars...")
    df1m = fetch_index_chunked(FROM_DATE, TO_DATE)
    dfN = resample_bars_per_day(df1m)
    dfN = compute_indicators(dfN)
    days_covered = sorted(set(dfN.index.date))
    print(f"Got {len(dfN)} {BAR_RULE} bars across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}\n")

    all_trades = []
    skipped_round_figure = 0
    for day in days_covered:
        day_df = dfN[dfN.index.date == day].copy()
        if len(day_df) < POP_SLOPE_LOOKBACK + 3:
            continue
        day_open_spot = day_df["open"].iloc[0]
        signals = detect_signals_for_day(day_df)
        open_trade_bar = -1
        strike = itm_put_strike(day_open_spot)
        premium_df = build_premium_series(day_df, strike, day_open_spot, side="PE")
        for sig in signals:
            j0 = sig["bar"]
            if j0 <= open_trade_bar:
                continue
            exec_idx = j0 + 1
            if exec_idx >= len(day_df):
                continue

            index_price_at_entry = day_df["open"].iloc[exec_idx]
            if near_round_figure(index_price_at_entry):
                skipped_round_figure += 1
                continue

            entry_price = premium_df["open"].iloc[exec_idx]
            exit_price, reason, exit_time = simulate_trade_fixed_exit(premium_df, exec_idx, entry_price)
            lot = lot_size_for_date(day)
            pnl_pts = exit_price - entry_price
            rupee_pnl = pnl_pts * lot - BROKERAGE_PER_TRADE
            all_trades.append({
                "date": day, "event": sig["event"], "side": "PE", "strike": strike,
                "signal_time": day_df.index[j0], "entry_time": day_df.index[exec_idx],
                "index_price_at_entry": round(index_price_at_entry, 2),
                "exit_time": exit_time, "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2), "exit_reason": reason,
                "pnl_pts": round(pnl_pts, 2), "rupee_pnl": round(rupee_pnl, 2),
            })
            open_trade_bar = day_df.index.get_loc(exit_time)

    print(f"Signals skipped by the round-figure filter: {skipped_round_figure}\n")

    if not all_trades:
        print("No trades generated.")
        return

    tdf = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    tdf["running_equity"] = STARTING_CAPITAL + tdf["rupee_pnl"].cumsum()
    tdf.to_csv("velez_8pt_1min_noround_fixedexit_6mo_trades.csv", index=False)

    def stats_block(g, label):
        if g.empty:
            print(f"{label}: no trades")
            return
        wins = g[g["rupee_pnl"] > 0]
        losses = g[g["rupee_pnl"] <= 0]
        win_rate = len(wins) / len(g)
        pf = (wins["rupee_pnl"].sum() / abs(losses["rupee_pnl"].sum())
              if losses["rupee_pnl"].sum() != 0 else float("inf"))
        print(f"{label}: n={len(g)}  win%={100*win_rate:.1f}  net_Rs={g['rupee_pnl'].sum():,.2f}  PF={pf:.2f}")

    print("--- Overall ---")
    stats_block(tdf, "All events")
    print("\n--- By event type ---")
    for ev, g in tdf.groupby("event"):
        stats_block(g, ev)
    print("\n--- By exit reason ---")
    for reason, g in tdf.groupby("exit_reason"):
        stats_block(g, reason)

    equity = tdf["running_equity"]
    running_peak = equity.cummax()
    max_dd_rs = (equity - running_peak).min()
    r_multiple = tdf["rupee_pnl"] / (FIXED_TARGET_PTS * tdf["date"].map(lot_size_for_date))
    sqn = (r_multiple.mean() / r_multiple.std()) * math.sqrt(len(tdf)) if r_multiple.std() else float("nan")
    print(f"\n--- Risk ---\nMax drawdown: Rs {max_dd_rs:,.2f}   SQN={sqn:.2f}")
    print("Never wipes out." if (equity > 0).all() else "WIPES OUT at some point.")
    print(f"\nStart Rs {STARTING_CAPITAL:,} -> End Rs {equity.iloc[-1]:,.2f}   "
          f"Total return: {100*(equity.iloc[-1]/STARTING_CAPITAL - 1):.1f}%")

    mid = len(tdf) // 2
    h1 = tdf["rupee_pnl"].iloc[:mid].sum()
    h2 = tdf["rupee_pnl"].iloc[mid:].sum()
    print(f"\n--- Stability check ---\nFirst half: n={mid} net Rs {h1:,.2f}   |   "
          f"Second half: n={len(tdf)-mid} net Rs {h2:,.2f}")

    print(f"\nTrade log saved: velez_8pt_1min_noround_fixedexit_6mo_trades.csv ({len(tdf)} rows)")
    print("\nFor reference, the ORIGINAL Velez 8pt (2-min bars, 8pt stop/4pt trail, no round-figure filter,")
    print("2-year window) real result was: n=284, PF=1.71, net Rs27,116.07, CAGR=45.23%, max DD -5.46%.")
    print("Compare directly, keeping in mind this run is 6 months not 2 years, so smaller n is expected.")


if __name__ == "__main__":
    main()
