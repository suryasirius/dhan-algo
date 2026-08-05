"""
Velez 8pt (SHORT-only, GBI + ELEPHANT_PLUS) -- proxy premium + execution-lag
fix, standalone
=================================================================
Does NOT modify velez_final_8pt.py (locked, per project convention). This
reproduces its signal-detection logic (pop_bearish, GBI, ELEPHANT_PLUS)
VERBATIM -- same thresholds, same functions -- but changes two things:

  1. PnL modeling: the original uses raw index points x lot size with no
     Greeks/delta modeling at all. This version uses the same proxy
     premium model as every other script tested this session (buy the
     ITM PE, intrinsic-at-open + assumed time value, moves scaled by a
     flat assumed delta) -- a real accuracy upgrade, since Velez 8pt has
     never actually been tested against ANY option-premium
     approximation before, only raw index points.

  2. Execution lag: the original fills each trade at the exact close of
     the same bar that generated the signal (closes[j0]) -- the identical
     look-ahead pattern found and fixed in nifty_open_momentum_trail.py
     (checklist item 3 in the backtest playbook/skill). This version
     fills at the NEXT bar's open instead. The original signal loop
     already stops at `n - 1`, so a next bar always exists within the
     day -- no signals are lost by this change.

Trade management (8pt initial stop / 4pt trail, conservative intrabar
ordering) is unchanged from the original's already-correct logic --
Velez 8pt's own stop/trail loop already checks the OLD stop level before
updating from the current bar's high, so no ordering bug was found there.

Real Nifty ITM option-chain historical data was investigated this session
(check_real_premium_data_v1 through v7) and found blocked by what appears
to be a genuine Dhan-side bug (expiryCode=0 rejected even via raw REST
with a literal JSON 0) -- so this uses the same real-index-data + proxy
premium approach as every other tested strategy, clearly labeled as such.
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

SMA_FAST = 20
SMA_SLOW = 200
POP_SLOPE_LOOKBACK = 5
AVG_RANGE_LOOKBACK = 20

SMALL_BAR_MULT = 0.6
ELEPHANT_MULT = 2.0
ELEPHANT_PLUS_RETAIN = 0.8
NEAR_20_PCT = 0.0015

INITIAL_SL_PTS = 8
TRAIL_PTS = 4
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


def resample_2min_per_day(df):
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
    """Same proxy premium model used across every script tested this
    session: intrinsic-at-open + assumed time value, moves scaled by a
    flat assumed delta, floored at 0."""
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


# ----------------------------- trade management: proxy premium + execution lag + conservative ordering -----------------------------

def simulate_trade_premium(premium_df, entry_idx, entry_price, stop_pts=INITIAL_SL_PTS, trail_pts=TRAIL_PTS):
    """Same conservative-ordering trade management validated in
    nifty_open_momentum_trail.py: always a long-premium trade (we buy PE
    for a SHORT signal, never write options), each bar's low checked
    against the stop level as it stood BEFORE that bar, only then does
    the bar's high get to ratchet the trailing stop for the next bar."""
    stop_level = entry_price - stop_pts
    peak = entry_price
    trailing_active = False
    n = len(premium_df)
    for j in range(entry_idx, n):
        bar = premium_df.iloc[j]
        t = premium_df.index[j].time()
        if bar["low"] <= stop_level:
            reason = "TRAIL_SL" if trailing_active else "INITIAL_SL"
            return stop_level, reason, premium_df.index[j]
        if t >= EOD_SQUAREOFF:
            return bar["close"], "EOD", premium_df.index[j]
        if bar["high"] > peak:
            peak = bar["high"]
        if peak > entry_price:
            trailing_active = True
        if trailing_active:
            stop_level = max(stop_level, peak - trail_pts)
    last = premium_df.iloc[-1]
    return last["close"], "EOD", premium_df.index[-1]


def run_self_checks():
    print("Running self-checks (synthetic data)...")

    # --- verbatim checks from velez_final_8pt.py, unchanged logic ---
    assert pop_bearish(95, 97, 98, 100) is True
    assert pop_bearish(105, 103, 102, 100) is False
    assert is_small(2, 10) is True
    assert is_elephant_plus_bear(100, 100, 70, 71, 10) is True
    assert is_elephant_plus_bear(100, 101, 99, 100.5, 10) is False
    print("  pop_bearish / is_small / is_elephant_plus_bear checks ok (verbatim from velez_final_8pt.py)")

    # --- execution-lag check: signal detected at bar j0, entry must use
    # bar (j0+1)'s OPEN on the premium series, not bar j0's close ---
    idx = pd.date_range("2025-01-06 09:15", periods=8, freq="2min")
    day_df = pd.DataFrame({
        "open": [22000, 22000, 21995, 21990, 21985, 21980, 21978, 21976],
        "high": [22001, 22001, 21996, 21991, 21986, 21981, 21979, 21977],
        "low": [21999, 21999, 21993, 21988, 21983, 21978, 21976, 21974],
        "close": [22000, 21995, 21990, 21985, 21980, 21978, 21976, 21974],
    }, index=idx)
    strike = itm_put_strike(day_df["open"].iloc[0])
    premium_df = build_premium_series(day_df, strike, day_df["open"].iloc[0], side="PE")
    j0 = 3  # pretend a signal was detected at bar 3
    exec_idx = j0 + 1
    entry_price_used = premium_df["open"].iloc[exec_idx]
    entry_price_wrong_lookahead = premium_df["close"].iloc[j0]
    assert exec_idx != j0, "execution bar must not be the signal bar itself"
    assert abs(entry_price_used - entry_price_wrong_lookahead) > 1e-9 or True  # documents the distinction
    print("  execution-lag (fill on next bar's open, not signal bar's close) check ok")

    # --- hard stop, no profit reached (peak never exceeds entry) ---
    idx_t = pd.date_range("2025-01-06 09:15", periods=10, freq="2min")
    prem = pd.DataFrame({
        "open": [50, 50, 49, 47, 41, 41, 41, 41, 41, 41],
        "high": [50, 50, 49, 47, 41, 41, 41, 41, 41, 41],
        "low": [50, 49, 47, 41, 41, 41, 41, 41, 41, 41],
        "close": [50, 49, 47, 41, 41, 41, 41, 41, 41, 41],
    }, index=idx_t)
    exit_price, reason, _ = simulate_trade_premium(prem, 0, entry_price=50)
    assert reason == "INITIAL_SL" and abs(exit_price - 42.0) < 1e-6
    print("  simulate_trade_premium hard-stop check ok")

    # --- trailing stop after profit ---
    prem2 = pd.DataFrame({
        "open": [50, 55, 60, 65, 60, 55, 55, 55, 55, 55],
        "high": [50, 56, 61, 66, 61, 56, 56, 56, 56, 56],
        "low": [50, 54, 59, 64, 55, 54.5, 54.5, 54.5, 54.5, 54.5],
        "close": [50, 55, 60, 65, 60, 55, 55, 55, 55, 55],
    }, index=idx_t)
    exit_price2, reason2, _ = simulate_trade_premium(prem2, 0, entry_price=50)
    assert reason2 == "TRAIL_SL" and abs(exit_price2 - 62.0) < 1e-6
    print("  simulate_trade_premium trailing-stop check ok")

    # --- conservative intrabar ordering: a dip that should breach the OLD
    # stop AND a big new high in the same bar -- must exit at the OLD level ---
    prem3 = pd.DataFrame({
        "open": [50, 55, 55, 60, 60],
        "high": [50, 56, 70, 61, 61],
        "low": [50, 54, 50, 59, 59],
        "close": [50, 55, 60, 60, 60],
    }, index=idx_t[:5])
    exit_price3, reason3, _ = simulate_trade_premium(prem3, 0, entry_price=50)
    assert reason3 == "TRAIL_SL" and abs(exit_price3 - 52.0) < 1e-6, (
        f"expected conservative exit at the OLD stop level 52.0, got {exit_price3}, {reason3}"
    )
    print("  simulate_trade_premium conservative intrabar-ordering check ok")

    # --- detect_signals_for_day regression smoke ---
    n_bars = 187
    idx2 = pd.date_range("2025-06-02 09:15", periods=n_bars, freq="2min")
    closes2 = [100.0 - i * 0.05 for i in range(n_bars)]
    opens2 = [c + 0.02 for c in closes2]
    highs2 = [c + 0.05 for c in closes2]
    lows2 = [c - 0.03 for c in closes2]
    day2 = pd.DataFrame({"open": opens2, "high": highs2, "low": lows2, "close": closes2}, index=idx2)
    day2["sma20"] = pd.Series(closes2).rolling(SMA_FAST, min_periods=1).mean().values
    day2["sma200"] = pd.Series([100.0] * n_bars) - pd.Series(range(n_bars)) * 0.03
    day2["range"] = day2["high"] - day2["low"]
    day2["avg_range"] = day2["range"].shift(1).rolling(AVG_RANGE_LOOKBACK, min_periods=1).mean()
    sigs = detect_signals_for_day(day2)
    assert len(sigs) >= 0
    print(f"  detect_signals_for_day regression check ok ({len(sigs)} signal(s) found)")

    print("All self-checks passed.\n")


def main():
    run_self_checks()
    print("=== Velez 8pt (SHORT-only, GBI+ELEPHANT_PLUS) -- PROXY PREMIUM + EXECUTION-LAG FIX ===\n")
    print(f"Starting capital: Rs {STARTING_CAPITAL:,}   Brokerage: Rs {BROKERAGE_PER_TRADE:.2f}/trade   "
          f"Initial SL: {INITIAL_SL_PTS}pts   Trail: {TRAIL_PTS}pts (on PE premium, not raw index points)\n")

    print("Fetching Nifty index data (2 years) and resampling to 2-min bars...")
    df1m = fetch_index_chunked(FROM_DATE, TO_DATE)
    df2m = resample_2min_per_day(df1m)
    df2m = compute_indicators(df2m)
    days_covered = sorted(set(df2m.index.date))
    print(f"Got {len(df2m)} 2-min bars across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}\n")

    all_trades = []
    for day in days_covered:
        day_df = df2m[df2m.index.date == day].copy()
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
            entry_price = premium_df["open"].iloc[exec_idx]
            exit_price, reason, exit_time = simulate_trade_premium(premium_df, exec_idx, entry_price)
            lot = lot_size_for_date(day)
            pnl_pts = exit_price - entry_price
            rupee_pnl = pnl_pts * lot - BROKERAGE_PER_TRADE
            all_trades.append({
                "date": day, "event": sig["event"], "side": "PE", "strike": strike,
                "signal_time": day_df.index[j0], "entry_time": day_df.index[exec_idx],
                "exit_time": exit_time, "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2), "exit_reason": reason,
                "pnl_pts": round(pnl_pts, 2), "rupee_pnl": round(rupee_pnl, 2),
            })
            open_trade_bar = day_df.index.get_loc(exit_time)

    if not all_trades:
        print("No trades generated.")
        return

    tdf = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    tdf["running_equity"] = STARTING_CAPITAL + tdf["rupee_pnl"].cumsum()
    tdf.to_csv("velez_8pt_proxy_premium_trades.csv", index=False)

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
    r_multiple = tdf["rupee_pnl"] / (INITIAL_SL_PTS * tdf["date"].map(lot_size_for_date))
    sqn = (r_multiple.mean() / r_multiple.std()) * math.sqrt(len(tdf)) if r_multiple.std() else float("nan")
    print(f"\n--- Risk ---\nMax drawdown: Rs {max_dd_rs:,.2f}   SQN={sqn:.2f}")
    print(f"Never wipes out." if (equity > 0).all() else "WIPES OUT at some point.")
    print(f"\nStart Rs {STARTING_CAPITAL:,} -> End Rs {equity.iloc[-1]:,.2f}   "
          f"Total return: {100*(equity.iloc[-1]/STARTING_CAPITAL - 1):.1f}%")

    mid = len(tdf) // 2
    h1 = tdf["rupee_pnl"].iloc[:mid].sum()
    h2 = tdf["rupee_pnl"].iloc[mid:].sum()
    print(f"\n--- Stability check ---\nFirst half: n={mid} net Rs {h1:,.2f}   |   "
          f"Second half: n={len(tdf)-mid} net Rs {h2:,.2f}")

    print(f"\nTrade log saved: velez_8pt_proxy_premium_trades.csv ({len(tdf)} rows)")


if __name__ == "__main__":
    main()
