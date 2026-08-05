"""
Open-and-High / Open-and-Low (O=H / O=L) Retest Strategy -- APPROXIMATION
=================================================================
Standalone, 2-year backtest, sweeping initial stop x fixed target (Grid A)
and initial stop x trailing-after-profit (Grid B), separately.

THIS IS AN APPROXIMATION -- read before trusting the numbers.
The source material (OI Pulse strategy explainer) depends on a
proprietary live options-chain/OI screening tool that surfaces, per
strike: open, high, low, an O=H/O=L badge, a "probability" score, and
a hit timestamp. That tool and its probability model are not available
here. What this script does instead, clearly documented as a stand-in
per the source doc's own suggested fallback (Section 3, "Backtest
rule"):

  - O=H / O=L is detected directly from SIMULATED option premium OHLC
    (same delta=0.6 / +40pt time-value proxy used throughout this
    project), not from the OI Pulse tool. "O=H" = the option's own
    premium has never traded above its opening print (cummax high ==
    open); the same detection is applied independently to the CE and
    PE premium series for the day's ATM strike (this sidesteps a real
    ambiguity in the source doc about how the O=L/"puts" mirror is
    supposed to work directionally -- treating both sides identically
    in each side's OWN premium space is the most internally consistent
    reading and keeps this a "buy low in the pullback, sell high at
    the retest" long-only setup on both sides, matching this project's
    convention of never writing/selling options).
  - The "probability" filter is DROPPED ENTIRELY -- it cannot be
    reconstructed without the proprietary tool's model. This is an
    intentional, flagged approximation, not an oversight.
  - Cross-confirmation (matching O=H on the CE side + O=L on the PE
    side) is OPTIONAL, not required, per the source doc's own "stronger
    candidate" (not mandatory) framing. Every trade logs whether the
    opposite side was also active at entry, so you can inspect the
    confirmed-vs-unconfirmed split yourself in the trade log CSV.
  - A minimum pullback of MIN_PULLBACK_PTS (2.0 points on the option
    premium) is required before a setup counts as "active" -- the
    source doc doesn't specify a magnitude, this is a reasonable
    threshold chosen to avoid trivial noise-level "dips" of a fraction
    of a point.
  - Entry fires on the first uptick (CE) / downtick (PE) after that
    minimum pullback, before the 11:00 AM cutoff, while the O=H/O=L
    ceiling/floor still hasn't been re-touched -- this is my own
    concrete trigger definition standing in for the undocumented exact
    mechanical trigger in the source material.
  - Strike: ATM (nearest strike to the day's opening spot), per the
    source doc's own suggested starting parameter ("ATM to 3-4 strikes
    ITM" -- ATM is the simplest, most literal reading of that range).

Stop/target/trailing values are NOT specified in the source material
at all (flagged there as a real gap) -- per your instruction, this
runs TWO separate sweeps:
  Grid A (fixed stop x fixed target): initial stop in {3,5,8,10} pts x
    fixed target in {5,10,15} pts = 12 combinations.
  Grid B (initial stop x trailing-after-profit): initial stop in
    {3,5,8,10} pts x trailing distance in {2,4,6} pts (activates once
    the trade is in profit, only ever tightens) = 12 combinations, NO
    fixed target -- exits on the trailing stop or EOD only.

PROXY PREMIUM MODE / exit-priority convention: same delta=0.6, +40pt
time-value, zero-floored premium simulation as the rest of this
project. Within a bar, STOP is checked before TARGET (conservative
assumption: adverse moves are assumed to resolve first intrabar),
matching the priority convention already used in the Velez scripts.
"""

import datetime as dt
import math
import pandas as pd
import numpy as np

import dhan_live_bot as dlb

FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")   # 2 years
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

STRIKE_STEP = 50
ENTRY_CUTOFF = dt.time(11, 0)
MIN_PULLBACK_PTS = 2.0
EOD_SQUAREOFF = dt.time(15, 20)

ASSUMED_DELTA = 0.6
ASSUMED_TIME_VALUE = 40.0

INITIAL_SL_VALUES = [3.0, 5.0, 8.0, 10.0]
TARGET_VALUES = [5.0, 10.0, 15.0]       # Grid A
TRAIL_VALUES = [2.0, 4.0, 6.0]          # Grid B

STARTING_CAPITAL = 10_000
BROKERAGE_PER_TRADE = 40.0
MONTE_CARLO_ITERS = 5000


def lot_size_for_date(d):
    if d < dt.date(2024, 11, 20):
        return 25
    elif d < dt.date(2026, 1, 1):
        return 75
    else:
        return 65


def nearest_strike(spot):
    return round(spot / STRIKE_STEP) * STRIKE_STEP


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


def build_premium_series(day_df, strike, day_open_spot, side):
    """side='CE' premium rises with the index; side='PE' premium rises as
    the index falls -- same proxy convention as every other script here."""
    idx_open = day_df["open"].to_numpy()
    idx_high = day_df["high"].to_numpy()
    idx_low = day_df["low"].to_numpy()
    idx_close = day_df["close"].to_numpy()

    if side == "CE":
        intrinsic0 = max(day_open_spot - strike, 0.0)
        move_open, move_high, move_low, move_close = (
            idx_open - day_open_spot, idx_high - day_open_spot,
            idx_low - day_open_spot, idx_close - day_open_spot,
        )
    else:
        intrinsic0 = max(strike - day_open_spot, 0.0)
        move_open, move_high, move_low, move_close = (
            day_open_spot - idx_open, day_open_spot - idx_high,
            day_open_spot - idx_low, day_open_spot - idx_close,
        )
    base = intrinsic0 + ASSUMED_TIME_VALUE

    p_open = np.maximum(base + move_open * ASSUMED_DELTA, 0.0)
    # NOTE: for PE, index high (bad for PE) maps to premium LOW and vice versa
    if side == "CE":
        p_high = np.maximum(base + move_high * ASSUMED_DELTA, 0.0)
        p_low = np.maximum(base + move_low * ASSUMED_DELTA, 0.0)
    else:
        p_high = np.maximum(base + move_low * ASSUMED_DELTA, 0.0)   # index LOW is best for PE
        p_low = np.maximum(base + move_high * ASSUMED_DELTA, 0.0)   # index HIGH is worst for PE
    p_close = np.maximum(base + move_close * ASSUMED_DELTA, 0.0)

    return pd.DataFrame({"open": p_open, "high": p_high, "low": p_low, "close": p_close},
                         index=day_df.index)


def find_oh_signal(prem_df, cutoff=ENTRY_CUTOFF, min_pullback=MIN_PULLBACK_PTS):
    """Detects an 'O=H' (open==day-high-so-far) setup, a pullback of at
    least min_pullback points below that ceiling, then an uptick before
    the cutoff time -- our own concrete trigger standing in for the
    undocumented exact mechanism in the source material.
    Returns (entry_idx, day_open_premium) or (None, day_open_premium)."""
    highs = prem_df["high"].tolist()
    closes = prem_df["close"].tolist()
    times = prem_df.index
    n = len(prem_df)
    day_open = closes[0] if prem_df["open"].iloc[0] != closes[0] else prem_df["open"].iloc[0]
    day_open = prem_df["open"].iloc[0]

    running_high = highs[0]
    for i in range(1, n):
        if times[i].time() >= cutoff:
            break
        # still "O=H" as of the PRIOR bars (ceiling never breached before this bar)
        still_oh = running_high <= day_open + 1e-9
        if still_oh and closes[i] <= day_open - min_pullback and closes[i] > closes[i - 1]:
            return i, day_open
        running_high = max(running_high, highs[i])
    return None, day_open


def find_ol_signal(prem_df, cutoff=ENTRY_CUTOFF, min_pullback=MIN_PULLBACK_PTS):
    """Mirror of find_oh_signal for a floor (open==day-low-so-far) pattern:
    a rally of at least min_pullback points above the floor, then a
    downtick before the cutoff. Used only for the informational
    cross-confirmation flag (see module docstring for why both CE and PE
    trades here use the O=H/uptick logic for their own entry)."""
    lows = prem_df["low"].tolist()
    closes = prem_df["close"].tolist()
    times = prem_df.index
    n = len(prem_df)
    day_open = prem_df["open"].iloc[0]

    running_low = lows[0]
    for i in range(1, n):
        if times[i].time() >= cutoff:
            break
        still_ol = running_low >= day_open - 1e-9
        if still_ol and closes[i] >= day_open + min_pullback and closes[i] < closes[i - 1]:
            return i, day_open
        running_low = min(running_low, lows[i])
    return None, day_open


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

    # ---- O=H detection: ceiling at open, pullback of 3pts, then an uptick ----
    # Bar 0's own high is pinned exactly at the open (no upper wick) so the
    # O=H premise (open==day-high-so-far) actually holds from the start;
    # every later bar's high stays a bit padded but still below the open.
    idx = pd.date_range("2026-01-06 09:15", periods=10, freq="1min")
    closes_ = [50, 49.5, 48, 47, 46.5, 47, 47.5, 48, 48.5, 49]
    opens_ = [50] + closes_[:-1]
    highs_ = [50] + [c + 0.2 for c in closes_[1:]]
    lows_ = [c - 0.2 for c in closes_]
    prem = pd.DataFrame({"open": opens_, "high": highs_, "low": lows_, "close": closes_}, index=idx)
    entry_idx, day_open = find_oh_signal(prem)
    # day_open=50; pullback of >=2pts below 50 is <=48; first uptick after that: bar4(46.5)->bar5(47) i=5
    assert day_open == 50
    assert entry_idx == 5, f"expected entry at bar 5, got {entry_idx}"
    print("  find_oh_signal pullback+uptick detection check ok")

    # no signal if it never pulls back enough
    closes_shallow = [50, 49.5, 49.2, 49.5, 49.8, 49.9, 50, 49.9, 49.8, 49.7]
    opens_shallow = [50] + closes_shallow[:-1]
    highs_shallow = [50] + [c + 0.05 for c in closes_shallow[1:]]
    prem2 = pd.DataFrame({"open": opens_shallow, "high": highs_shallow,
                          "low": [c - 0.05 for c in closes_shallow], "close": closes_shallow}, index=idx)
    entry_idx2, _ = find_oh_signal(prem2)
    assert entry_idx2 is None, "shallow pullback (<2pts) should not trigger a signal"
    print("  find_oh_signal no-signal-on-shallow-pullback check ok")

    # no signal once ceiling already breached (a bar's high exceeds the open before the pullback candidate)
    closes_breach = [50, 51.5, 49, 47, 46.5, 47, 47.5, 48, 48.5, 49]
    opens_breach = [50] + closes_breach[:-1]
    highs_breach = [50, 52] + [c + 0.2 for c in closes_breach[2:]]
    prem3 = pd.DataFrame({"open": opens_breach, "high": highs_breach,
                          "low": [c - 0.2 for c in closes_breach], "close": closes_breach}, index=idx)
    entry_idx3, _ = find_oh_signal(prem3)
    assert entry_idx3 is None, "O=H should be invalidated once a bar's high exceeds the open"
    print("  find_oh_signal ceiling-already-breached check ok")

    # ---- Grid A: fixed stop + fixed target ----
    idx2 = pd.date_range("2026-02-02 09:15", periods=8, freq="1min")
    closes_t = [50, 52, 55, 58, 60, 59, 58, 57]
    highs_t = [c + 0.5 for c in closes_t]
    lows_t = [c - 0.5 for c in closes_t]
    prem_t = pd.DataFrame({"open": [50] + closes_t[:-1], "high": highs_t, "low": lows_t, "close": closes_t}, index=idx2)
    t1 = simulate_trade_fixed(prem_t, entry_idx=0, initial_sl=5.0, target=8.0, lot=75)
    # entry=50, target=58; bar3 high=58.5>=58 -> TARGET at 58, pts=8
    assert t1["exit_reason"] == "TARGET" and abs(t1["points_gained"] - 8.0) < 0.01
    print("  simulate_trade_fixed TARGET-hit check ok")

    closes_s = [50, 48, 46, 44, 43, 44, 45, 46]
    highs_s = [c + 0.5 for c in closes_s]
    lows_s = [c - 0.5 for c in closes_s]
    prem_s = pd.DataFrame({"open": [50] + closes_s[:-1], "high": highs_s, "low": lows_s, "close": closes_s}, index=idx2)
    t2 = simulate_trade_fixed(prem_s, entry_idx=0, initial_sl=5.0, target=8.0, lot=75)
    # entry=50, stop=45; bar3 low=43.5<=45 -> STOP at 45, pts=-5
    assert t2["exit_reason"] == "STOP" and abs(t2["points_gained"] - (-5.0)) < 0.01
    print("  simulate_trade_fixed STOP-hit check ok")

    # ---- Grid B: initial stop + trailing ----
    closes_tr = [50, 52, 54, 53, 52, 51, 50, 49]
    highs_tr = [c + 0.3 for c in closes_tr]
    lows_tr = [c - 0.3 for c in closes_tr]
    prem_tr = pd.DataFrame({"open": [50] + closes_tr[:-1], "high": highs_tr, "low": lows_tr, "close": closes_tr}, index=idx2)
    t3 = simulate_trade_trailing(prem_tr, entry_idx=0, initial_sl=5.0, trail_pts=2.0, lot=75)
    # entry=50, stop=45; bar1 c=52>50 -> trailing_active, stop=max(45, 51.7-2=49.7)=49.7
    # bar2 c=54, stop=max(49.7, 53.7-2=51.7)=51.7
    # bar3 c=53, low=52.7, stop=max(51.7,52.7-2=50.7)=51.7 (no tighten, 50.7<51.7)
    # bar4 c=52, low=51.7, stop=max(51.7,51.7-2=49.7)=51.7 -> low(51.7)<=stop(51.7) TRAIL_SL at bar4
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


def main():
    run_self_checks()
    print("=== O=H / O=L Retest Strategy (APPROXIMATION) -- stop x target/trail sweep ===\n")
    print("Grid A (fixed stop x fixed target): stop in", INITIAL_SL_VALUES, "x target in", TARGET_VALUES)
    print("Grid B (initial stop x trailing-after-profit): stop in", INITIAL_SL_VALUES, "x trail in", TRAIL_VALUES)
    print(f"{len(INITIAL_SL_VALUES)*len(TARGET_VALUES)} + {len(INITIAL_SL_VALUES)*len(TRAIL_VALUES)} "
          f"= {len(INITIAL_SL_VALUES)*len(TARGET_VALUES) + len(INITIAL_SL_VALUES)*len(TRAIL_VALUES)} "
          "total combinations across the two grids\n")
    print("REMINDER: this is an approximation of the O=H/O=L concept using simulated option premium")
    print("(no OI Pulse tool access, no 'probability' filter, entry trigger is my own definition) --")
    print("see the script docstring for the full list of assumptions before trusting these numbers.\n")

    print(f"Fetching Nifty index 1-min data ({FROM_DATE} to {TO_DATE})...")
    df1m = fetch_index_chunked(FROM_DATE, TO_DATE)
    days = sorted(set(df1m.index.date))
    print(f"Got {len(df1m)} 1-min bars across {len(days)} trading day(s): {days[0]} to {days[-1]}\n")

    # ---- find one entry per day (shared across all grid combos, since the
    # entry signal doesn't depend on stop/target/trail values) ----
    day_entries = []
    for d in days:
        day_df = df1m[df1m.index.date == d]
        if len(day_df) < 10:
            continue
        day_open_spot = day_df["open"].iloc[0]
        strike = nearest_strike(day_open_spot)

        ce_prem = build_premium_series(day_df, strike, day_open_spot, "CE")
        pe_prem = build_premium_series(day_df, strike, day_open_spot, "PE")

        ce_entry, _ = find_oh_signal(ce_prem)
        pe_entry, _ = find_oh_signal(pe_prem)   # same detector applied to PE's own premium space

        ce_confirm, _ = find_ol_signal(pe_prem) if ce_entry is not None else (None, None)
        pe_confirm, _ = find_ol_signal(ce_prem) if pe_entry is not None else (None, None)

        candidates = []
        if ce_entry is not None:
            candidates.append(("CE", ce_entry, ce_prem, ce_confirm is not None))
        if pe_entry is not None:
            candidates.append(("PE", pe_entry, pe_prem, pe_confirm is not None))
        if not candidates:
            continue
        candidates.sort(key=lambda c: c[1])
        side, entry_idx, prem_df, cross_confirmed = candidates[0]
        lot = lot_size_for_date(d)
        day_entries.append((d, side, prem_df, entry_idx, lot, cross_confirmed))

    print(f"Days with a qualifying entry: {len(day_entries)} / {len(days)}\n")
    if not day_entries:
        print("No trades generated -- no O=H/O=L setups detected under these assumptions.")
        return

    n_confirmed = sum(1 for e in day_entries if e[5])
    print(f"Cross-confirmed (opposite side also active) at entry: {n_confirmed} / {len(day_entries)} "
          "(informational only -- not used as a filter)\n")

    # ---------------- Grid A: fixed stop x fixed target ----------------
    print("--- Grid A: fixed stop x fixed target ---")
    resultsA = {}
    logsA = {}
    for sl in INITIAL_SL_VALUES:
        for tgt in TARGET_VALUES:
            key = f"sl{sl:g}_tp{tgt:g}"
            trades = []
            for d, side, prem_df, entry_idx, lot, confirmed in day_entries:
                t = simulate_trade_fixed(prem_df, entry_idx, sl, tgt, lot)
                t.update(date=d, side=side, cross_confirmed=confirmed)
                trades.append(t)
            tdf = pd.DataFrame(trades)
            logsA[key] = tdf
            resultsA[key] = stats_for(tdf)
            tdf.to_csv(f"nifty_oh_ol_gridA_{key}_trades.csv", index=False)

    print(f"{'Combo':<14} {'n':>5} {'win%':>7} {'PF':>7} {'net_pts':>10} {'net_Rs':>12} "
          f"{'max_DD':>12} {'End_Rs':>12}")
    for sl in INITIAL_SL_VALUES:
        for tgt in TARGET_VALUES:
            key = f"sl{sl:g}_tp{tgt:g}"
            r = resultsA[key]
            print(f"{key:<14} {r['n']:>5} {r['win_pct']:>6.1f}% {r['pf']:>7.2f} {r['net_pts']:>10.2f} "
                  f"{r['net_rs']:>12,.2f} {r['max_dd']:>12,.2f} {r['final_equity']:>12,.2f}")
    print()

    # ---------------- Grid B: initial stop x trailing-after-profit ----------------
    print("--- Grid B: initial stop x trailing-after-profit (no fixed target) ---")
    resultsB = {}
    logsB = {}
    for sl in INITIAL_SL_VALUES:
        for trail in TRAIL_VALUES:
            key = f"sl{sl:g}_trail{trail:g}"
            trades = []
            for d, side, prem_df, entry_idx, lot, confirmed in day_entries:
                t = simulate_trade_trailing(prem_df, entry_idx, sl, trail, lot)
                t.update(date=d, side=side, cross_confirmed=confirmed)
                trades.append(t)
            tdf = pd.DataFrame(trades)
            logsB[key] = tdf
            resultsB[key] = stats_for(tdf)
            tdf.to_csv(f"nifty_oh_ol_gridB_{key}_trades.csv", index=False)

    print(f"{'Combo':<16} {'n':>5} {'win%':>7} {'PF':>7} {'net_pts':>10} {'net_Rs':>12} "
          f"{'max_DD':>12} {'End_Rs':>12}")
    for sl in INITIAL_SL_VALUES:
        for trail in TRAIL_VALUES:
            key = f"sl{sl:g}_trail{trail:g}"
            r = resultsB[key]
            print(f"{key:<16} {r['n']:>5} {r['win_pct']:>6.1f}% {r['pf']:>7.2f} {r['net_pts']:>10.2f} "
                  f"{r['net_rs']:>12,.2f} {r['max_dd']:>12,.2f} {r['final_equity']:>12,.2f}")
    print()

    # ---------------- cross-confirmed vs unconfirmed comparison (best Grid A combo) ----------------
    best_keyA = max(resultsA, key=lambda k: resultsA[k]["net_rs"])
    print(f"--- Confirmed vs unconfirmed setups, best Grid A combo ({best_keyA}) ---")
    tdf_best = logsA[best_keyA]
    for flag, label in [(True, "Cross-confirmed"), (False, "Unconfirmed")]:
        g = tdf_best[tdf_best["cross_confirmed"] == flag]
        if g.empty:
            print(f"  {label}: no trades")
            continue
        print(f"  {label}: n={len(g)}  win%={100*(g['points_gained']>0).mean():.1f}  "
              f"net_Rs={g['rupee_pnl'].sum():,.2f}")
    print()

    print("Trade logs saved: nifty_oh_ol_gridA_<combo>_trades.csv and "
          "nifty_oh_ol_gridB_<combo>_trades.csv (one file per combination)")


if __name__ == "__main__":
    main()
