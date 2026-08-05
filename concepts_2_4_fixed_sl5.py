"""
Concepts 2 & 4 -- Fixed -5pt Stop-Loss variant, standalone
=================================================================
STANDALONE. Uses the REAL dlb.compute_latest_signal() from the live bot
for signal detection (same fidelity convention as every other script in
this project) via the expanding-window-per-day pattern: at each bar of
an allowed weekday, call compute_latest_signal(day_df.iloc[:i+1]) and
act only when it returns final_buy=True or final_sell=True.

IMPORTANT: this uses SINGLE-DAY-ONLY slicing (no multi-day rolling
context window). That matches how your original working 6-month test
(86 signals) was run, and matches every other script's convention in
this project. An earlier "5-day context window" theory was tried in a
separate diagnostic thread and produced 0 signals on real data -- this
script deliberately does NOT use that approach.

Weekday filter: reads dlb.ALLOWED_WEEKDAYS if the live bot module
exposes it. Note this is a set of Python weekday() INTS (Monday=0 ...
e.g. {1, 2} for Tue/Wed), not weekday name strings -- confirmed from a
real run, matching every historical Concepts 2&4 trade log in this
project (410/410 trades in your uploaded 2-year log were Tue/Wed only).

TRADE MANAGEMENT (this variant): FIXED stop-loss of -5 points on the
raw index move (as requested), no trailing. Target defaults to +10
points -- a 1:2 risk:reward, chosen because it matches the exact
TARGET floor (10.00 pts) already observed in your historical trade
log, i.e. this is the tightest target your system has actually used.
Edit TARGET_PTS below if you want to test a different target instead.
A MAX_HOLD_MINUTES cap (15 min) closes out any trade that hits neither
level, matching the "TIME" exit reason seen in your historical data
(whose trades ran 1-16 minutes).

P&L convention: matches every prior Concepts 2&4 trade log in this
project -- raw index points x lot size (date-correct lookup) minus
Rs 40/trade brokerage. No options-premium/delta modeling (that
convention is specific to ORB/Velez; Concepts 2&4 has never used it
in this project).
"""

import datetime as dt
import pandas as pd
import numpy as np

import dhan_live_bot as dlb

FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

FIXED_STOP_PTS = 5.0
TARGET_PTS = 10.0
MAX_HOLD_MINUTES = 15

STARTING_CAPITAL = 10_000
BROKERAGE_PER_TRADE = 40.0
MONTE_CARLO_ITERS = 5000

DEFAULT_ALLOWED_WEEKDAYS = ["Tuesday", "Wednesday"]


def lot_size_for_date(d):
    if d < dt.date(2024, 11, 20):
        return 25
    elif d < dt.date(2026, 1, 1):
        return 75
    else:
        return 65


def get_allowed_weekdays():
    """Returns a set of Python weekday() ints (Monday=0 ... Sunday=6).
    dlb.ALLOWED_WEEKDAYS is itself already int-keyed (e.g. {1, 2} for
    Tue/Wed) -- normalize the string-name fallback to the same int form
    so day-filtering only ever has to compare ints to ints."""
    raw = getattr(dlb, "ALLOWED_WEEKDAYS", None)
    if raw is None:
        name_to_int = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                       "Friday": 4, "Saturday": 5, "Sunday": 6}
        return {name_to_int[n] for n in DEFAULT_ALLOWED_WEEKDAYS}
    return set(raw)


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


def extract_signal(sig):
    """Returns (side, concept_id) or None. sig is the dict returned by
    dlb.compute_latest_signal() -- always a dict, real signal only when
    final_buy or final_sell is True."""
    if sig is None:
        return None
    if sig.get("final_buy"):
        return ("LONG", sig.get("concept_id"))
    if sig.get("final_sell"):
        return ("SHORT", sig.get("concept_id"))
    return None


def scan_day_for_signals(day_df):
    """Single-day-only expanding window scan (Theory A -- matches the
    original working convention). Returns a list of (bar_index, side,
    concept_id) tuples."""
    signals = []
    n = len(day_df)
    for i in range(n):
        sig = dlb.compute_latest_signal(day_df.iloc[:i + 1])
        extracted = extract_signal(sig)
        if extracted is not None:
            side, concept_id = extracted
            signals.append((i, side, concept_id))
    return signals


def simulate_trade(day_df, entry_idx, side, day, concept_id):
    idx = day_df.index
    closes = day_df["close"].tolist()
    highs = day_df["high"].tolist()
    lows = day_df["low"].tolist()
    entry_price = closes[entry_idx]
    lot = lot_size_for_date(day)

    if side == "SHORT":
        stop = entry_price + FIXED_STOP_PTS
        target = entry_price - TARGET_PTS
    else:
        stop = entry_price - FIXED_STOP_PTS
        target = entry_price + TARGET_PTS

    deadline = idx[entry_idx] + pd.Timedelta(minutes=MAX_HOLD_MINUTES)
    exit_i, exit_price, reason = len(idx) - 1, closes[-1], "EOD"

    for i in range(entry_idx + 1, len(idx)):
        t = idx[i]
        if side == "SHORT":
            if highs[i] >= stop:
                exit_i, exit_price, reason = i, stop, "STOP"
                break
            if lows[i] <= target:
                exit_i, exit_price, reason = i, target, "TARGET"
                break
        else:
            if lows[i] <= stop:
                exit_i, exit_price, reason = i, stop, "STOP"
                break
            if highs[i] >= target:
                exit_i, exit_price, reason = i, target, "TARGET"
                break
        if t >= deadline:
            exit_i, exit_price, reason = i, closes[i], "TIME"
            break

    if side == "SHORT":
        points = entry_price - exit_price
    else:
        points = exit_price - entry_price

    rupee_pnl = points * lot - BROKERAGE_PER_TRADE
    return {
        "date": day, "side": side, "concept_id": concept_id, "lot_size": lot,
        "entry_time": idx[entry_idx], "exit_time": idx[exit_i],
        "entry_price": entry_price, "exit_price": exit_price,
        "exit_reason": reason, "points_gained": points, "rupee_pnl": rupee_pnl,
    }


def run_self_checks():
    print("Running self-checks (synthetic data)...")

    # SHORT winner: price declines steadily past the target
    rng = pd.date_range("2026-01-06 09:15", periods=30, freq="1min")  # Tuesday
    close = 25000 - np.arange(30) * 0.8
    day_df = pd.DataFrame({
        "open": np.roll(close, 1), "high": close + 0.3, "low": close - 0.3,
        "close": close, "volume": 100,
    }, index=rng)
    day_df.iloc[0, day_df.columns.get_loc("open")] = close[0]
    t = simulate_trade(day_df, entry_idx=0, side="SHORT", day=dt.date(2026, 1, 6), concept_id=2)
    assert t["exit_reason"] == "TARGET", f"expected TARGET, got {t['exit_reason']}"
    assert abs(t["points_gained"] - TARGET_PTS) < 0.01
    print("  simulate_trade SHORT target-hit check ok")

    # SHORT loser: price rallies past the fixed 5pt stop
    close2 = 25000 + np.arange(30) * 0.8
    day_df2 = pd.DataFrame({
        "open": np.roll(close2, 1), "high": close2 + 0.3, "low": close2 - 0.3,
        "close": close2, "volume": 100,
    }, index=rng)
    day_df2.iloc[0, day_df2.columns.get_loc("open")] = close2[0]
    t2 = simulate_trade(day_df2, entry_idx=0, side="SHORT", day=dt.date(2026, 1, 6), concept_id=2)
    assert t2["exit_reason"] == "STOP", f"expected STOP, got {t2['exit_reason']}"
    assert abs(t2["points_gained"] + FIXED_STOP_PTS) < 0.01, "SL loss should be bounded to exactly -5pts"
    print("  simulate_trade SHORT stop-loss check ok")

    # Flat/going-nowhere trade should time out at MAX_HOLD_MINUTES
    close3 = np.full(30, 25000.0)
    day_df3 = pd.DataFrame({
        "open": close3, "high": close3 + 0.1, "low": close3 - 0.1,
        "close": close3, "volume": 100,
    }, index=rng)
    t3 = simulate_trade(day_df3, entry_idx=0, side="SHORT", day=dt.date(2026, 1, 6), concept_id=2)
    assert t3["exit_reason"] == "TIME", f"expected TIME, got {t3['exit_reason']}"
    print("  simulate_trade TIME-exit check ok")

    # extract_signal shape handling
    assert extract_signal({"final_buy": False, "final_sell": True, "concept_id": 4}) == ("SHORT", 4)
    assert extract_signal({"final_buy": True, "final_sell": False, "concept_id": 2}) == ("LONG", 2)
    assert extract_signal({"final_buy": False, "final_sell": False, "concept_id": None}) is None
    print("  extract_signal shape check ok")

    print("All self-checks passed.\n")


def stats_block(g, label):
    if g.empty:
        print(f"{label}: no trades")
        return
    wins = g[g["points_gained"] > 0]
    losses = g[g["points_gained"] <= 0]
    win_rate = len(wins) / len(g)
    pf = (wins["points_gained"].sum() / abs(losses["points_gained"].sum())
          if losses["points_gained"].sum() != 0 else float("inf"))
    print(f"{label}: n={len(g)}  win%={100*win_rate:.1f}  net_pts={g['points_gained'].sum():.2f}  "
          f"PF={pf:.2f}  net_Rs={g['rupee_pnl'].sum():,.2f}")


def main():
    run_self_checks()
    print("=== Concepts 2 & 4 -- FIXED -5pt stop / +10pt target / 15-min max hold ===\n")
    print(f"Stop: -{FIXED_STOP_PTS}pts  Target: +{TARGET_PTS}pts  Max hold: {MAX_HOLD_MINUTES} min  "
          f"Rs {BROKERAGE_PER_TRADE:.0f}/trade brokerage\n")

    allowed_weekdays = get_allowed_weekdays()
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    print(f"Allowed weekdays: {sorted(weekday_names[w] for w in allowed_weekdays)}\n")

    print("Fetching Nifty index 1-min data...")
    df1m = fetch_index_chunked(FROM_DATE, TO_DATE)
    all_days = sorted(set(df1m.index.date))
    trading_days = [d for d in all_days if d.weekday() in allowed_weekdays]
    print(f"Got {len(df1m)} 1-min bars across {len(all_days)} trading day(s) total, "
          f"{len(trading_days)} of them on allowed weekdays "
          f"({all_days[0]} to {all_days[-1]})\n")

    trades = []
    for d in trading_days:
        day_df = df1m[df1m.index.date == d]
        if len(day_df) < 5:
            continue
        signals = scan_day_for_signals(day_df)
        open_bar = -1
        for bar_i, side, concept_id in signals:
            if bar_i <= open_bar:
                continue
            t = simulate_trade(day_df, bar_i, side, d, concept_id)
            trades.append(t)
            open_bar = day_df.index.get_loc(t["exit_time"])

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        print("No trades generated. If this is unexpected, this likely means the real "
              "dhan_live_bot.compute_latest_signal() found no qualifying setups on the "
              "days scanned -- check that dlb is importing correctly and ALLOWED_WEEKDAYS "
              "matches what you expect.")
        return

    tdf = tdf.sort_values("entry_time").reset_index(drop=True)
    tdf["running_equity"] = STARTING_CAPITAL + tdf["rupee_pnl"].cumsum()
    tdf.to_csv("concepts_2_4_fixed_sl5_trades.csv", index=False)

    print(f"Total trades: {len(tdf)}\n")
    stats_block(tdf, "Overall")
    stats_block(tdf[tdf.side == "SHORT"], "SHORT only")
    stats_block(tdf[tdf.side == "LONG"], "LONG only")
    print()
    for reason, g in tdf.groupby("exit_reason"):
        stats_block(g, reason)
    print()
    for cid, g in tdf.groupby("concept_id"):
        stats_block(g, f"Concept {cid}")

    equity = tdf["running_equity"]
    running_peak = equity.cummax()
    max_dd = (equity - running_peak).min()
    print(f"\nMax drawdown: Rs {max_dd:,.2f}   {'WIPES OUT' if (equity<=0).any() else 'never wipes out'}")
    print(f"Start Rs {STARTING_CAPITAL:,} -> End Rs {equity.iloc[-1]:,.2f}   "
          f"Return: {100*(equity.iloc[-1]/STARTING_CAPITAL-1):.1f}%")

    mid = len(tdf) // 2
    h1 = tdf["rupee_pnl"].iloc[:mid].sum()
    h2 = tdf["rupee_pnl"].iloc[mid:].sum()
    print(f"Stability: first half n={mid} Rs{h1:,.2f}  |  second half n={len(tdf)-mid} Rs{h2:,.2f}")

    r = tdf["points_gained"] / FIXED_STOP_PTS
    sqn = (r.mean() / r.std()) * (len(r) ** 0.5) if r.std() > 0 else float("nan")
    print(f"SQN: {sqn:.2f}")

    pnls = tdf["rupee_pnl"].to_numpy()
    rng_mc = np.random.RandomState(42)
    touches_zero = below_half = 0
    for _ in range(MONTE_CARLO_ITERS):
        shuffled = rng_mc.permutation(pnls)
        eq = STARTING_CAPITAL + np.cumsum(shuffled)
        if (eq <= 0).any():
            touches_zero += 1
        if (eq <= STARTING_CAPITAL * 0.5).any():
            below_half += 1
    print(f"Monte Carlo ({MONTE_CARLO_ITERS} reshuffles): "
          f"{100*touches_zero/MONTE_CARLO_ITERS:.2f}% of paths touch <=0, "
          f"{100*below_half/MONTE_CARLO_ITERS:.2f}% fall below 50% of starting capital")

    print(f"\nTrade log saved: concepts_2_4_fixed_sl5_trades.csv ({len(tdf)} rows)")


if __name__ == "__main__":
    main()
