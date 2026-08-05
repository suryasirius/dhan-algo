"""
Oliver Velez "Crash Play" -- short-side fade of a failed bounce after a
crash bar, using the half/thirds zone system. Standalone, 2-year backtest.
=================================================================
REVISION 2 -- three changes made after reviewing the first real run's
results, per your instruction to fix what needs fixing:

  1. BUG FIX (zone-odds bonus validation only, does NOT affect the trade
     grids below): the first version classified the "outcome zone" as
     the zone AT THE MOMENT price resolved to a new low or a recovery --
     which is tautological (a new low is trivially near Zone 4, a
     recovery must pass through Zone 1 on the way past crash_open), and
     produced a meaningless 0%/100% split with Zone 2/3 empty. Fixed to
     track the DEEPEST point the bounce actually reached (its peak
     close, i.e. the smallest zone number / furthest retracement) BEFORE
     it turns back down, then check what happened AFTER that peak. This
     is what Section 3 actually describes ("pullback holds within Zone
     X" = the zone reached before reversing).
  2. STARTING CAPITAL raised from Rs 10,000 to Rs 25,000. The first run's
     best combo had a max drawdown (-Rs 11,528) larger than the Rs
     10,000 starting capital -- the strategy wasn't broken, the capital
     was inadequate, same lesson as Velez needing Rs 25,000.
  3. SWEEP EXTENDED per the observed trend (every stop size did better
     with a tighter trail; trail2 beat trail4/trail6 across the board):
       INITIAL_SL_VALUES: {3,5,8,10} -> {1,2,3,5,8,10} (added 1,2)
       TRAIL_VALUES: {2,4,6} -> {1,2,4,6} (added 1)
     TARGET_VALUES kept at {5,10,15} for comparison, even though Grid A
     (fixed target) was a clear loser across the board last time --
     keeping it lets you confirm that's still true at the new stop
     sizes and capital level, not just assume it.

Everything else is unchanged from the first version -- same crash-bar
detection, same zone system, same CONFIRM/AGGRESSIVE entry variants, same
PE-only (long premium) trade direction. See below for the full original
scope notes.

ORIGINAL SCOPE DECISIONS (still apply):
  - Only Section 5/6 (the Crash Play itself) is implemented. Section 1's
    4-characteristic "market top" framework is NOT used as a pre-filter
    here -- leg-boundary detection is explicitly flagged by the source
    material itself as subjective.
  - Crash bar: a down-closing 2-min bar whose range >= CRASH_MULT (2.5,
    the midpoint of the doc's suggested 2-3x) times the average range of
    the prior 20 bars.
  - Zone system applied to the MICRO move only (the crash bar plus the
    continued decline immediately after it, until the bounce begins) --
    not a separate macro/full-decline version. Zones measured from the
    post-crash low (0%) up to the crash bar's own open (100%):
    Zone 4 = 0-33%, Zone 3 = 33-50%, Zone 2 = 50-67%, Zone 1 = 67-100%.
    "Below the halfway point" (Zone 3/4) is the trade zone per Section 5.
  - CONFIRM = wait for a color-change bar (green-to-red flip breaking
    the prior bar's low) while price is in Zone 3/4.
    AGGRESSIVE = enter the moment price first reaches Zone 3/4 during
    the bounce, no confirmation bar required.
  - Stop/target/trailing are FIXED POINT sweeps on the option premium
    (not the doc's qualitative "stop above the bounce high" / "2
    zone-levels down" rules), same convention as every other sweep in
    this project.
  - Trade direction: always buys the PUT (long premium) -- matches this
    project's "always buy, never write" convention. Premium is simulated
    the same way as every other script here (delta=0.6, +40pt time
    value, floored at 0).
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

AVG_RANGE_LOOKBACK = 20
CRASH_MULT = 2.5
MAX_LOOKAHEAD_BARS = 60   # cap how far forward we look for the bounce/outcome, per day

ASSUMED_DELTA = 0.6
ASSUMED_TIME_VALUE = 40.0

INITIAL_SL_VALUES = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]   # extended: added 1, 2
TARGET_VALUES = [5.0, 10.0, 15.0]     # Grid A
TRAIL_VALUES = [1.0, 2.0, 4.0, 6.0]   # Grid B, extended: added 1

STARTING_CAPITAL = 25_000   # raised from 10,000
BROKERAGE_PER_TRADE = 40.0


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


def resample_2min_per_day(df, bar_rule=BAR_RULE):
    out = []
    for day in sorted(set(df.index.date)):
        day_df = df[df.index.date == day]
        r = day_df.resample(bar_rule, origin=day_df.index[0]).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        out.append(r)
    return pd.concat(out).sort_index()


def is_red(o, c):
    return c < o


def is_green(o, c):
    return c > o


def build_pe_premium_series(day_df, strike, day_open_spot):
    """PE premium rises as the index falls -- same proxy convention as
    every other script in this project."""
    idx_open = day_df["open"].to_numpy()
    idx_high = day_df["high"].to_numpy()
    idx_low = day_df["low"].to_numpy()
    idx_close = day_df["close"].to_numpy()

    intrinsic0 = max(strike - day_open_spot, 0.0)
    base = intrinsic0 + ASSUMED_TIME_VALUE

    p_open = np.maximum(base + (day_open_spot - idx_open) * ASSUMED_DELTA, 0.0)
    p_close = np.maximum(base + (day_open_spot - idx_close) * ASSUMED_DELTA, 0.0)
    p_high = np.maximum(base + (day_open_spot - idx_low) * ASSUMED_DELTA, 0.0)    # index LOW is best for PE
    p_low = np.maximum(base + (day_open_spot - idx_high) * ASSUMED_DELTA, 0.0)    # index HIGH is worst for PE

    return pd.DataFrame({"open": p_open, "high": p_high, "low": p_low, "close": p_close},
                         index=day_df.index)


def zone_of(level, crash_low, crash_open):
    """Returns 1 (top third, closest to crash_open) .. 4 (bottom third,
    closest to crash_low), or None if level is outside [crash_low, crash_open]."""
    rng = crash_open - crash_low
    if rng <= 0:
        return None
    pct = (level - crash_low) / rng   # 0 at the low, 1 at the crash open
    if pct < 0 or pct > 1.0 + 1e-9:
        return None
    if pct >= 0.67:
        return 1
    elif pct >= 0.50:
        return 2
    elif pct >= 0.33:
        return 3
    else:
        return 4


def find_crash_signals_for_day(day_df):
    """Returns a list of dicts, one per crash-bar event found this day:
    {crash_bar_idx, crash_open, crash_low, confirm_entry_idx, aggressive_entry_idx,
     zone_confirm, zone_aggr, outcome, outcome_zone}.

    outcome/outcome_zone (FIXED in revision 2): outcome_zone is the zone
    of the bounce's PEAK close (the deepest retracement it reached before
    turning back down), tracked across the whole lookahead window;
    outcome is whichever happens FIRST after that peak -- a new low
    (below crash_low, i.e. continuation) or a recovery past crash_open."""
    opens = day_df["open"].tolist()
    highs = day_df["high"].tolist()
    lows = day_df["low"].tolist()
    closes = day_df["close"].tolist()
    rng = (day_df["high"] - day_df["low"]).tolist()
    avg_range = pd.Series(rng).shift(1).rolling(AVG_RANGE_LOOKBACK, min_periods=AVG_RANGE_LOOKBACK).mean().tolist()
    n = len(day_df)

    events = []
    j = AVG_RANGE_LOOKBACK
    while j < n - 2:
        if avg_range[j] and rng[j] >= CRASH_MULT * avg_range[j] and is_red(opens[j], closes[j]):
            crash_open = opens[j]
            crash_low = lows[j]
            # extend the decline while price keeps making lower lows
            k = j
            while k + 1 < n and lows[k + 1] <= crash_low + 1e-9:
                k += 1
                crash_low = min(crash_low, lows[k])
            # now watch the bounce from k+1 onward for entries / outcomes
            confirm_idx, aggr_idx = None, None
            zone_confirm, zone_aggr = None, None
            bounce_start = k + 1
            limit = min(n, bounce_start + MAX_LOOKAHEAD_BARS)

            for i in range(bounce_start, limit):
                c = closes[i]
                z = zone_of(c, crash_low, crash_open)
                if aggr_idx is None and z in (3, 4):
                    aggr_idx, zone_aggr = i, z
                if confirm_idx is None and z in (3, 4) and i > bounce_start:
                    if is_red(opens[i], closes[i]) and is_green(opens[i - 1], closes[i - 1]) and lows[i] < lows[i - 1]:
                        confirm_idx, zone_confirm = i, z

            # --- zone-odds bonus validation: find the bounce's peak (deepest
            # retracement reached, i.e. highest close / smallest zone number)
            # across the whole window, then check what happens AFTER that peak ---
            peak_i, peak_close = None, None
            for i in range(bounce_start, limit):
                c = closes[i]
                if peak_close is None or c > peak_close:
                    peak_close, peak_i = c, i
            outcome, outcome_zone = None, None
            if peak_i is not None:
                outcome_zone = zone_of(peak_close, crash_low, crash_open)
                if outcome_zone is None:
                    outcome_zone = 1 if peak_close > crash_open else 4
                for i in range(peak_i + 1, limit):
                    if closes[i] < crash_low:
                        outcome = "NEW_LOW"
                        break
                    if closes[i] > crash_open:
                        outcome = "RECOVERED"
                        break
                if outcome is None and peak_close > crash_open:
                    outcome = "RECOVERED"   # the peak itself already exceeded crash_open

            events.append({
                "crash_bar_idx": j, "crash_open": crash_open, "crash_low": crash_low,
                "confirm_entry_idx": confirm_idx, "zone_confirm": zone_confirm,
                "aggressive_entry_idx": aggr_idx, "zone_aggr": zone_aggr,
                "outcome": outcome, "outcome_zone": outcome_zone,
            })
            j = k + MAX_LOOKAHEAD_BARS  # skip past this event's window before scanning for another
        j += 1
    return events


def simulate_trade_fixed(prem_df, entry_idx, initial_sl, target, lot):
    closes = prem_df["close"].tolist()
    highs = prem_df["high"].tolist()
    lows = prem_df["low"].tolist()
    idx = prem_df.index
    n = len(prem_df)
    entry_premium = closes[entry_idx]
    stop = entry_premium - initial_sl
    tgt = entry_premium + target

    exit_i, exit_premium, reason = n - 1, closes[-1], "EOD"
    for i in range(entry_idx + 1, n):
        t = idx[i]
        if lows[i] <= stop:
            exit_i, exit_premium, reason = i, stop, "STOP"
            break
        if highs[i] >= tgt:
            exit_i, exit_premium, reason = i, tgt, "TARGET"
            break
        if t.time() >= EOD_SQUAREOFF:
            exit_i, exit_premium, reason = i, closes[i], "EOD"
            break

    points = exit_premium - entry_premium
    rupee_pnl = points * lot - BROKERAGE_PER_TRADE
    return {"entry_time": idx[entry_idx], "exit_time": idx[exit_i], "entry_premium": entry_premium,
            "exit_premium": exit_premium, "exit_reason": reason, "points_gained": points,
            "rupee_pnl": rupee_pnl}


def simulate_trade_trailing(prem_df, entry_idx, initial_sl, trail_pts, lot):
    closes = prem_df["close"].tolist()
    highs = prem_df["high"].tolist()
    lows = prem_df["low"].tolist()
    idx = prem_df.index
    n = len(prem_df)
    entry_premium = closes[entry_idx]
    stop = entry_premium - initial_sl
    trailing_active = False

    exit_i, exit_premium, reason = n - 1, closes[-1], "EOD"
    for i in range(entry_idx + 1, n):
        t = idx[i]
        if lows[i] <= stop:
            exit_i, exit_premium, reason = i, stop, ("TRAIL_SL" if trailing_active else "INITIAL_SL")
            break
        if closes[i] > entry_premium:
            trailing_active = True
        if trailing_active:
            stop = max(stop, lows[i] - trail_pts)
        if t.time() >= EOD_SQUAREOFF:
            exit_i, exit_premium, reason = i, closes[i], "EOD"
            break

    points = exit_premium - entry_premium
    rupee_pnl = points * lot - BROKERAGE_PER_TRADE
    return {"entry_time": idx[entry_idx], "exit_time": idx[exit_i], "entry_premium": entry_premium,
            "exit_premium": exit_premium, "exit_reason": reason, "points_gained": points,
            "rupee_pnl": rupee_pnl}


def run_self_checks():
    print("Running self-checks (synthetic data)...")

    # ---- zone_of boundaries ----
    assert zone_of(100, 0, 100) == 1     # exactly at crash_open -> top zone
    assert zone_of(67, 0, 100) == 1
    assert zone_of(66, 0, 100) == 2
    assert zone_of(50, 0, 100) == 2
    assert zone_of(49, 0, 100) == 3
    assert zone_of(33, 0, 100) == 3
    assert zone_of(32, 0, 100) == 4
    assert zone_of(0, 0, 100) == 4
    assert zone_of(-1, 0, 100) is None
    assert zone_of(101, 0, 100) is None
    print("  zone_of boundary check ok")

    # ---- crash bar + bounce + confirm/aggressive entry detection ----
    idx = pd.date_range("2026-01-06 09:15", periods=25, freq="2min")
    opens_ = [100.0] * AVG_RANGE_LOOKBACK
    closes_ = [100.0] * AVG_RANGE_LOOKBACK
    highs_ = [100.1] * AVG_RANGE_LOOKBACK
    lows_ = [99.9] * AVG_RANGE_LOOKBACK
    opens_.append(100.0); closes_.append(80.0); highs_.append(100.0); lows_.append(80.0)
    opens_.append(80.0); closes_.append(78.0); highs_.append(80.0); lows_.append(77.5)
    opens_.append(78.0); closes_.append(83.0); highs_.append(83.0); lows_.append(78.5)   # bar22: zone4
    opens_.append(83.0); closes_.append(87.0); highs_.append(87.0); lows_.append(83.0)    # bar23: zone3 (bounce peak)
    opens_.append(87.0); closes_.append(85.0); highs_.append(87.0); lows_.append(82.0)    # bar24: color-change confirm
    day_df = pd.DataFrame({"open": opens_, "high": highs_, "low": lows_, "close": closes_}, index=idx)
    events = find_crash_signals_for_day(day_df)
    assert len(events) == 1, f"expected 1 crash event, got {len(events)}"
    ev = events[0]
    assert ev["crash_bar_idx"] == AVG_RANGE_LOOKBACK
    assert abs(ev["crash_open"] - 100.0) < 0.01
    assert abs(ev["crash_low"] - 77.5) < 0.01
    assert ev["aggressive_entry_idx"] == 22, f"expected aggressive entry at bar22, got {ev['aggressive_entry_idx']}"
    assert ev["zone_aggr"] == 4
    assert ev["confirm_entry_idx"] == 24, f"expected confirm entry at bar24, got {ev['confirm_entry_idx']}"
    assert ev["zone_confirm"] == 3
    print("  find_crash_signals_for_day crash-bar + confirm/aggressive-entry detection ok")

    # ---- zone-odds outcome fix check: bounce peaks at bar23 (close=87, zone3),
    # then bar24 dips to 85 (still >crash_low, no resolution yet), and this
    # short synthetic day ends without a clean new-low or recovery -- outcome
    # should be None (inconclusive), NOT tautologically tied to zone4/zone1
    assert ev["outcome_zone"] == 3, f"expected outcome_zone=3 (the bounce's peak at bar23), got {ev['outcome_zone']}"
    assert ev["outcome"] is None, f"expected inconclusive outcome (data ends before resolution), got {ev['outcome']}"
    print("  zone-odds outcome-zone-is-the-peak (not the resolution moment) check ok")

    # A second, longer synthetic day: bounce peaks in zone2, then price goes
    # on to make a genuine new low afterward -> outcome should be NEW_LOW,
    # tagged with the PEAK's zone (2), not zone4 (which the old buggy version
    # would have produced).
    idx2 = pd.date_range("2026-01-07 09:15", periods=30, freq="2min")
    o2 = [100.0] * AVG_RANGE_LOOKBACK
    c2 = [100.0] * AVG_RANGE_LOOKBACK
    h2 = [100.1] * AVG_RANGE_LOOKBACK
    l2 = [99.9] * AVG_RANGE_LOOKBACK
    o2.append(100.0); c2.append(80.0); h2.append(100.0); l2.append(80.0)     # bar20: crash
    o2.append(80.0); c2.append(78.0); h2.append(80.0); l2.append(77.5)        # bar21: decline continues
    o2.append(78.0); c2.append(90.0); h2.append(90.0); l2.append(78.5)        # bar22: strong bounce -> zone_of((90-77.5)/22.5=55.6%)=2
    o2.append(90.0); c2.append(85.0); h2.append(90.0); l2.append(84.5)        # bar23: pulls back a bit
    o2.append(85.0); c2.append(70.0); h2.append(85.5); l2.append(70.0)        # bar24: drops to a genuine new low (<77.5)
    for _ in range(5):
        o2.append(70.0); c2.append(70.0); h2.append(70.1); l2.append(69.9)
    day_df2 = pd.DataFrame({"open": o2, "high": h2, "low": l2, "close": c2}, index=idx2)
    events2 = find_crash_signals_for_day(day_df2)
    assert len(events2) == 1
    ev2 = events2[0]
    assert ev2["outcome_zone"] == 2, f"expected the bounce's peak zone (2, from bar22's close=90), got {ev2['outcome_zone']}"
    assert ev2["outcome"] == "NEW_LOW", f"expected NEW_LOW after the peak, got {ev2['outcome']}"
    print("  zone-odds NEW_LOW-tagged-with-peak-zone (not resolution zone) check ok")

    # ---- simulate_trade_fixed / trailing reuse (same proven logic as before) ----
    idx3 = pd.date_range("2026-02-02 09:15", periods=8, freq="1min")
    closes_t = [50, 52, 55, 58, 60, 59, 58, 57]
    highs_t = [c + 0.5 for c in closes_t]
    lows_t = [c - 0.5 for c in closes_t]
    prem_t = pd.DataFrame({"open": [50] + closes_t[:-1], "high": highs_t, "low": lows_t, "close": closes_t}, index=idx3)
    t1 = simulate_trade_fixed(prem_t, entry_idx=0, initial_sl=5.0, target=8.0, lot=75)
    assert t1["exit_reason"] == "TARGET" and abs(t1["points_gained"] - 8.0) < 0.01
    print("  simulate_trade_fixed TARGET-hit check ok")

    closes_tr = [50, 52, 54, 53, 52, 51, 50, 49]
    highs_tr = [c + 0.3 for c in closes_tr]
    lows_tr = [c - 0.3 for c in closes_tr]
    prem_tr = pd.DataFrame({"open": [50] + closes_tr[:-1], "high": highs_tr, "low": lows_tr, "close": closes_tr}, index=idx3)
    t3 = simulate_trade_trailing(prem_tr, entry_idx=0, initial_sl=5.0, trail_pts=2.0, lot=75)
    assert t3["exit_reason"] == "TRAIL_SL", f"expected TRAIL_SL, got {t3['exit_reason']}"
    print("  simulate_trade_trailing trailing-stop math check ok")

    print("All self-checks passed.\n")


def stats_for(g):
    if g.empty:
        return dict(n=0, win_pct=float("nan"), pf=float("nan"), net_pts=0.0, net_rs=0.0,
                    max_dd=0.0, final_equity=STARTING_CAPITAL)
    wins = g[g["points_gained"] > 0]
    losses = g[g["points_gained"] <= 0]
    win_rate = len(wins) / len(g)
    pf = (wins["points_gained"].sum() / abs(losses["points_gained"].sum())
          if losses["points_gained"].sum() != 0 else float("inf"))
    gg = g.sort_values("entry_time")
    equity = STARTING_CAPITAL + gg["rupee_pnl"].cumsum()
    peak = equity.cummax()
    max_dd = (equity - peak).min()
    return dict(n=len(g), win_pct=100 * win_rate, pf=pf, net_pts=g["points_gained"].sum(),
                net_rs=g["rupee_pnl"].sum(), max_dd=max_dd, final_equity=equity.iloc[-1])


def print_grid(results, label, keys_fn):
    print(f"{'Combo':<16} {'n':>5} {'win%':>7} {'PF':>7} {'net_pts':>10} {'net_Rs':>12} "
          f"{'max_DD':>12} {'End_Rs':>12}")
    for key in keys_fn():
        r = results[key]
        print(f"{key:<16} {r['n']:>5} {r['win_pct']:>6.1f}% {r['pf']:>7.2f} {r['net_pts']:>10.2f} "
              f"{r['net_rs']:>12,.2f} {r['max_dd']:>12,.2f} {r['final_equity']:>12,.2f}")
    print()


def run_variant(entries, label):
    """entries: list of (day, prem_df, entry_idx, lot). Runs Grid A and
    Grid B for this entry variant (CONFIRM or AGGRESSIVE)."""
    print(f"=== {label} entries: {len(entries)} trades available ===\n")
    if not entries:
        print("  No trades for this variant.\n")
        return

    print(f"--- {label}: Grid A (fixed stop x fixed target) ---")
    resultsA = {}
    for sl in INITIAL_SL_VALUES:
        for tgt in TARGET_VALUES:
            key = f"sl{sl:g}_tp{tgt:g}"
            trades = [simulate_trade_fixed(prem_df, entry_idx, sl, tgt, lot)
                      for _, prem_df, entry_idx, lot in entries]
            tdf = pd.DataFrame(trades)
            resultsA[key] = stats_for(tdf)
            tdf.to_csv(f"nifty_crash_play_{label}_gridA_{key}_trades.csv", index=False)
    print_grid(resultsA, label, lambda: [f"sl{sl:g}_tp{tgt:g}" for sl in INITIAL_SL_VALUES for tgt in TARGET_VALUES])

    print(f"--- {label}: Grid B (initial stop x trailing-after-profit) ---")
    resultsB = {}
    for sl in INITIAL_SL_VALUES:
        for trail in TRAIL_VALUES:
            key = f"sl{sl:g}_trail{trail:g}"
            trades = [simulate_trade_trailing(prem_df, entry_idx, sl, trail, lot)
                      for _, prem_df, entry_idx, lot in entries]
            tdf = pd.DataFrame(trades)
            resultsB[key] = stats_for(tdf)
            tdf.to_csv(f"nifty_crash_play_{label}_gridB_{key}_trades.csv", index=False)
    print_grid(resultsB, label, lambda: [f"sl{sl:g}_trail{trail:g}" for sl in INITIAL_SL_VALUES for trail in TRAIL_VALUES])


def main():
    run_self_checks()
    print("=== Oliver Velez Crash Play REV2 -- CONFIRM vs AGGRESSIVE, extended stop/trail sweep ===\n")
    print(f"Starting capital: Rs {STARTING_CAPITAL:,} (raised from Rs 10,000 -- see docstring)")
    print("Crash bar threshold: bar range >= 2.5x the average range of the prior 20 bars (2-min bars).")
    print("Zones measured from the post-crash low (0%) to the crash bar's open (100%): ")
    print("  Zone4=0-33%  Zone3=33-50%  Zone2=50-67%  Zone1=67-100%. Trade zone = Zone 3/4.\n")
    print("Section 1's 4-characteristic market-top framework is NOT used as a filter here -- scope")
    print("cut, see script docstring. Always buys the PUT (long premium), never writes options.\n")

    print(f"Fetching Nifty index 1-min data ({FROM_DATE} to {TO_DATE})...")
    df1m = fetch_index_chunked(FROM_DATE, TO_DATE)
    df2m = resample_2min_per_day(df1m)
    days = sorted(set(df2m.index.date))
    print(f"Got {len(df2m)} 2-min bars across {len(days)} trading day(s): {days[0]} to {days[-1]}\n")

    confirm_entries = []
    aggressive_entries = []
    zone_outcomes = []   # for the bonus zone-odds validation

    for d in days:
        day_df = df2m[df2m.index.date == d]
        if len(day_df) < AVG_RANGE_LOOKBACK + 5:
            continue
        day_open_spot = day_df["open"].iloc[0]
        strike = itm_put_strike(day_open_spot)
        prem_df = build_pe_premium_series(day_df, strike, day_open_spot)
        lot = lot_size_for_date(d)

        events = find_crash_signals_for_day(day_df)
        for ev in events:
            if ev["outcome"] is not None:
                zone_outcomes.append({"date": d, "zone": ev["outcome_zone"], "outcome": ev["outcome"]})
            if ev["confirm_entry_idx"] is not None:
                confirm_entries.append((d, prem_df, ev["confirm_entry_idx"], lot))
            if ev["aggressive_entry_idx"] is not None:
                aggressive_entries.append((d, prem_df, ev["aggressive_entry_idx"], lot))

    print(f"Crash-bar events with a CONFIRM entry: {len(confirm_entries)}")
    print(f"Crash-bar events with an AGGRESSIVE entry: {len(aggressive_entries)}\n")

    run_variant(confirm_entries, "CONFIRM")
    run_variant(aggressive_entries, "AGGRESSIVE")

    # ---------------- bonus: zone-odds validation (Section 3), FIXED ----------------
    print("--- BONUS: zone-odds validation (Section 3's stated 80/65/35/15% claim), FIXED ---")
    print("(zone = the bounce's PEAK retracement before it turned back down; outcome = what")
    print(" happened first afterward -- a new low (continuation) or a recovery past crash_open)\n")
    zdf = pd.DataFrame(zone_outcomes)
    if zdf.empty:
        print("No zone outcomes recorded.")
    else:
        print(f"Total resolved bounces: {len(zdf)} (inconclusive/unresolved ones are excluded)\n")
        for z in [1, 2, 3, 4]:
            g = zdf[zdf["zone"] == z]
            if g.empty:
                print(f"  Zone {z}: no observations")
                continue
            pct_new_low = 100 * (g["outcome"] == "NEW_LOW").mean()
            print(f"  Zone {z}: n={len(g)}  % continued to a new low={pct_new_low:.1f}%  "
                  f"(doc's stated mirror odds: Zone1~20% / Zone2~35% / Zone3~65% / Zone4~85% continuation)")

    print("\nTrade logs saved: nifty_crash_play_<CONFIRM|AGGRESSIVE>_grid<A|B>_<combo>_trades.csv")


if __name__ == "__main__":
    main()
