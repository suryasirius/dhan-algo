"""
Top 4 Ranked Strategies COMBINED -- independent, overlap allowed -- 2-year backtest
=================================================================
Combines the four strategies from the top of the ranking, exactly as
you named them:

  1. Velez 8pt (short)              -- always buys the ITM PUT
  2. Opening-Drive Momentum          -- buys ITM CALL or PUT depending on
                                         which way the first opening move goes
  3. Crash Play (REV2, CONFIRM)      -- always buys the ITM PUT
  4. Two-Touch Resistance Breakout   -- always buys the ATM CALL, with the
                                         FILTERED exclusion rule applied
                                         (Monday/Wednesday and the
                                         13:15-13:45 session dropped, since
                                         those were independently found weak
                                         across both a 6-month and a 2-year
                                         cut) -- same version flagged in the
                                         ranking as "a hypothesis worth
                                         monitoring," included here as you
                                         asked, with that caveat still
                                         attached to its own numbers below.

Does NOT touch dhan_live_bot.py, nifty_paper_trade_top3_live.py, or
nifty_two_touch_resistance_breakout_backtest.py (all locked/already
validated, per project convention). Instead this IMPORTS them and reuses
their real, already-tested signal-detection and trade-management
functions verbatim -- zero hand-transcription, zero re-derivation risk.
The only new code here is the day-by-day driver loop, a small hook that
captures each module's own log_entry/log_exit calls into an in-memory
trade list (instead of writing to disk), and the Two-Touch weekday/
session filter (a straight port of nifty_two_touch_resistance_filtered_
backtest.py's own filter, applied the same way here).

INDEPENDENT + OVERLAP ALLOWED: each of the four strategies runs with its
own persistent state (own open-position tracking) and is driven
separately every day. If two or more of them want to be in a trade at
the same time, that's fine -- each opens its own position; one never
blocks another. The only restriction is the normal one already built
into each strategy: a SINGLE strategy can't open a second trade while
its own current trade is still open.

All trades from all four strategies are combined onto ONE shared
Rs25,000 equity curve (sorted by time), so the backtest reflects what
a single account running all four at once would actually look like.

2-year window, starting capital Rs 25,000, per your request.
"""

import datetime as dt
import time
import pandas as pd

import nifty_paper_trade_top3_live as top3
import nifty_two_touch_resistance_breakout_backtest as ttr

FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

STARTING_CAPITAL = 25_000
MAX_REPEATS_PER_DAY = 12  # generous cap on round-trips/day per strategy (smoke_test used 6)

# --- Two-Touch Resistance filter, ported verbatim from
# nifty_two_touch_resistance_filtered_backtest.py ---
TTR_EXCLUDED_WEEKDAYS = {0, 2}   # Monday, Wednesday
TTR_EXCLUDED_SESSION = "13:15-13:45"

_captured_trades = []
_pending_entries = {}
_orig_log_entry = top3.log_entry
_orig_log_exit = top3.log_exit


def _capture_log_entry(strategy, event, side, strike, bar_time, price, lot, day_pnl):
    _pending_entries[strategy] = dict(event=event, side=side, strike=strike,
                                       entry_time=bar_time, entry_price=price, lot=lot)


def _capture_log_exit(strategy, event, side, strike, bar_time, price, reason, pnl_pts, rupee_pnl, lot, day_pnl):
    entry = _pending_entries.pop(strategy, None)
    _captured_trades.append({
        "strategy": strategy, "event": event, "side": side, "strike": strike,
        "entry_time": entry["entry_time"] if entry else None,
        "entry_price": entry["entry_price"] if entry else None,
        "exit_time": bar_time, "exit_price": price, "exit_reason": reason,
        "pnl_pts": pnl_pts, "rupee_pnl": rupee_pnl, "lot_size": lot,
        "date": bar_time.date(),
    })


def fetch_index_chunked(from_date, to_date, chunk_days=CHUNK_DAYS):
    """Chunked wrapper around top3.fetch_index_window (which is designed
    for short live-context windows, not a full 2-year pull)."""
    start = dt.datetime.strptime(from_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(to_date, "%Y-%m-%d").date()
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + dt.timedelta(days=chunk_days - 1), end)
        raw = top3.fetch_index_window(cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"))
        chunks.append(raw)
        cur = chunk_end + dt.timedelta(days=1)
    full = pd.concat(chunks)
    return full[~full.index.duplicated(keep="first")].sort_index()


def run_two_touch_filtered_for_day(day_df):
    """Reuses ttr.run_standalone_for_day() verbatim, then drops any trade
    that falls on an excluded weekday or in the excluded session --
    exactly the same post-hoc filter as nifty_two_touch_resistance_
    filtered_backtest.py, just applied inline here."""
    trades = ttr.run_standalone_for_day(day_df)
    if not trades:
        return []
    day = day_df.index[0].date()
    if day.weekday() in TTR_EXCLUDED_WEEKDAYS:
        return []
    kept = []
    for t in trades:
        if ttr.classify_session(t["entry_time"]) == TTR_EXCLUDED_SESSION:
            continue
        kept.append(t)
    return kept


def run_self_checks():
    print("Running self-checks (synthetic data)...")

    # re-verify the underlying, already-validated modules' own self-checks
    top3.run_self_checks()
    ttr.run_self_checks()
    print("  top3.run_self_checks() + ttr.run_self_checks() re-verified ok "
          "(Velez/OpeningDrive/CrashPlay/TwoTouch primitives)")

    # --- capture-hook check: an entry followed by an exit must pair up into
    # ONE combined trade record with the right fields, not touching disk ---
    global _captured_trades, _pending_entries
    _captured_trades, _pending_entries = [], {}
    top3.log_entry = _capture_log_entry
    top3.log_exit = _capture_log_exit
    t0 = pd.Timestamp("2025-01-06 09:20")
    t1 = pd.Timestamp("2025-01-06 09:30")
    top3.log_entry("TestStrat", "EVT", "PE", 25000, t0, 100.0, 75, 0.0)
    top3.log_exit("TestStrat", "EVT", "PE", 25000, t1, 108.0, "TRAIL_SL", 8.0, 559.0, 75, 559.0)
    assert len(_captured_trades) == 1
    tr = _captured_trades[0]
    assert tr["entry_time"] == t0 and tr["exit_time"] == t1
    assert tr["entry_price"] == 100.0 and tr["exit_price"] == 108.0
    assert tr["rupee_pnl"] == 559.0
    print("  log_entry/log_exit capture-hook pairing check ok")
    _captured_trades, _pending_entries = [], {}
    top3.log_entry, top3.log_exit = _orig_log_entry, _orig_log_exit

    # --- Two-Touch weekday/session filter check: a Monday trade and a
    # 13:15 Tuesday trade must both be dropped; an ok Tuesday trade kept ---
    mon_day = dt.date(2025, 1, 6)     # Monday
    tue_day = dt.date(2025, 1, 7)     # Tuesday

    class _FakeIndex(list):
        def __getitem__(self, k):
            return super().__getitem__(k)

    def fake_day_df(day, entry_time):
        idx = pd.DatetimeIndex([pd.Timestamp.combine(day, dt.time(9, 15))])
        df = pd.DataFrame({"open": [100.0], "high": [100.0], "low": [100.0], "close": [100.0]}, index=idx)
        return df, entry_time

    orig_run_standalone = ttr.run_standalone_for_day
    try:
        ttr.run_standalone_for_day = lambda day_df: [{"entry_time": day_df.attrs["fake_entry_time"]}]

        df_mon, _ = fake_day_df(mon_day, pd.Timestamp("2025-01-06 09:20"))
        df_mon.attrs["fake_entry_time"] = pd.Timestamp("2025-01-06 09:20")
        assert run_two_touch_filtered_for_day(df_mon) == [], "Monday trades must be dropped"

        df_tue_badsession, _ = fake_day_df(tue_day, pd.Timestamp("2025-01-07 13:20"))
        df_tue_badsession.attrs["fake_entry_time"] = pd.Timestamp("2025-01-07 13:20")
        assert run_two_touch_filtered_for_day(df_tue_badsession) == [], "13:15-13:45 session trades must be dropped"

        df_tue_ok, _ = fake_day_df(tue_day, pd.Timestamp("2025-01-07 10:20"))
        df_tue_ok.attrs["fake_entry_time"] = pd.Timestamp("2025-01-07 10:20")
        kept = run_two_touch_filtered_for_day(df_tue_ok)
        assert len(kept) == 1, "a normal Tuesday, ok-session trade must be kept"
    finally:
        ttr.run_standalone_for_day = orig_run_standalone
    print("  Two-Touch Resistance weekday/session exclusion filter check ok")

    print("All self-checks passed.\n")


def stats_for(g):
    if g.empty:
        return dict(n=0, win_pct=float("nan"), pf=float("nan"), net_rs=0.0,
                    max_dd=0.0, sqn=float("nan"), final_equity=STARTING_CAPITAL)
    wins = g[g["rupee_pnl"] > 0]
    losses = g[g["rupee_pnl"] <= 0]
    win_rate = len(wins) / len(g)
    pf = (wins["rupee_pnl"].sum() / abs(losses["rupee_pnl"].sum())
          if losses["rupee_pnl"].sum() != 0 else float("inf"))
    g = g.sort_values("entry_time")
    equity = STARTING_CAPITAL + g["rupee_pnl"].cumsum()
    peak = equity.cummax()
    max_dd = (equity - peak).min()
    r = g["rupee_pnl"]
    sqn = (r.mean() / r.std()) * (len(r) ** 0.5) if r.std() > 0 else float("nan")
    return dict(n=len(g), win_pct=100 * win_rate, pf=pf, net_rs=g["rupee_pnl"].sum(),
                max_dd=max_dd, sqn=sqn, final_equity=equity.iloc[-1])


def main():
    run_self_checks()
    print("=== Top 4 Ranked Strategies COMBINED -- independent, overlap allowed -- 2-year backtest ===\n")
    print("Strategies: Velez 8pt (short), Opening-Drive Momentum, Crash Play (REV2/CONFIRM), "
          "Two-Touch Resistance Breakout (filtered: excl. Mon/Wed + 13:15-13:45)\n")

    print(f"Fetching Nifty index 1-min data ({FROM_DATE} to {TO_DATE}, ~2 years)...")
    raw = fetch_index_chunked(FROM_DATE, TO_DATE)
    full_2min = top3.resample_2min_per_day(raw)
    full_velez = top3.compute_velez_indicators(full_2min)
    days = sorted(set(full_2min.index.date))
    print(f"Got {len(full_2min)} 2-min bars across {len(days)} trading day(s): {days[0]} to {days[-1]}\n")

    global _captured_trades, _pending_entries
    _captured_trades, _pending_entries = [], {}
    top3.log_entry = _capture_log_entry
    top3.log_exit = _capture_log_exit

    ttr_trades_all = []
    t_start = time.time()

    for di, day in enumerate(days):
        day_df = full_2min[full_2min.index.date == day].copy()
        day_df_velez = full_velez[full_velez.index.date == day].copy()
        if day_df.empty:
            continue

        # Fresh per-day state for each of the 3 top3-driven strategies --
        # safe because step_long_premium_position/step_crash_position both
        # force an EOD exit by 15:20, so no position ever survives past the
        # end of a day's own data (verified in their own self-checks).
        states = {
            "velez": top3.new_strategy_state(),
            "od": top3.new_strategy_state(),
            "crash": top3.new_strategy_state(),
        }
        for name, fn, df_to_use in (
            ("velez", top3.process_velez, day_df_velez),
            ("od", top3.process_opening_drive, day_df),
            ("crash", top3.process_crash_play, day_df),
        ):
            st = states[name]
            for _ in range(MAX_REPEATS_PER_DAY):
                before = (st["position"] is not None, st["trades_today"])
                fn(df_to_use, st, verbose=False)
                after = (st["position"] is not None, st["trades_today"])
                if before == after:
                    break

        ttr_trades_all.extend(run_two_touch_filtered_for_day(day_df))

    top3.log_entry, top3.log_exit = _orig_log_entry, _orig_log_exit
    print(f"Scan complete in {time.time() - t_start:.1f}s.\n")

    combined = []
    for t in _captured_trades:
        combined.append({
            "strategy": t["strategy"], "side": t["side"], "strike": t["strike"],
            "entry_time": t["entry_time"], "exit_time": t["exit_time"],
            "entry_price": t["entry_price"], "exit_price": t["exit_price"],
            "exit_reason": t["exit_reason"], "pnl_pts": round(t["pnl_pts"], 2),
            "rupee_pnl": round(t["rupee_pnl"], 2), "date": t["date"],
        })
    for t in ttr_trades_all:
        combined.append({
            "strategy": "TwoTouchResistance", "side": t["side"], "strike": t["strike"],
            "entry_time": t["entry_time"], "exit_time": t["exit_time"],
            "entry_price": t["entry_price"], "exit_price": t["exit_price"],
            "exit_reason": t["reason"], "pnl_pts": round(t["pnl_pts"], 2),
            "rupee_pnl": round(t["rupee_pnl"], 2), "date": t["date"],
        })

    if not combined:
        print("No trades generated across any of the 4 strategies.")
        return

    tdf = pd.DataFrame(combined).sort_values("entry_time").reset_index(drop=True)
    tdf.to_csv("top4_strategies_combined_trades.csv", index=False)

    r = stats_for(tdf)
    print("--- COMBINED (all 4 strategies, one shared Rs25,000 account) ---")
    print(f"n={r['n']}  win%={r['win_pct']:.1f}  PF={r['pf']:.2f}  net_Rs={r['net_rs']:,.2f}  "
          f"maxDD_Rs={r['max_dd']:,.2f}  SQN={r['sqn']:.2f}  final_equity_Rs={r['final_equity']:,.2f}\n")

    print("--- By strategy (isolated stats -- how each performed on its own within this run) ---")
    for strat, g in tdf.groupby("strategy"):
        rg = stats_for(g)
        print(f"  [{strat}] n={rg['n']}  win%={rg['win_pct']:.1f}  PF={rg['pf']:.2f}  "
              f"net_Rs={rg['net_rs']:,.2f}  SQN={rg['sqn']:.2f}")

    print("\n--- By exit reason (combined) ---")
    for reason, g in tdf.groupby("exit_reason"):
        print(f"  {reason}: n={len(g)} ({100*len(g)/len(tdf):.1f}%)  win%={100*(g['rupee_pnl']>0).mean():.1f}  "
              f"net_Rs={g['rupee_pnl'].sum():,.2f}")

    # overlap diagnostic
    events = []
    for _, row in tdf.iterrows():
        events.append((row["entry_time"], 1))
        events.append((row["exit_time"], -1))
    events.sort()
    concurrent, max_concurrent = 0, 0
    for _, delta in events:
        concurrent += delta
        max_concurrent = max(max_concurrent, concurrent)
    print(f"\n--- Overlap diagnostic ---\nMax simultaneous open positions across all 4 strategies: {max_concurrent}")

    dates = pd.to_datetime(tdf["date"])
    mid = dates.min() + (dates.max() - dates.min()) / 2
    fh, sh = tdf[dates <= mid], tdf[dates > mid]
    m1, m2 = stats_for(fh), stats_for(sh)
    print(f"\n--- Stability check (in-sample/out-of-sample split) ---")
    print(f"  First half:  n={m1['n']} net_Rs={m1['net_rs']:,.2f} PF={m1['pf']:.2f} SQN={m1['sqn']:.2f}")
    print(f"  Second half: n={m2['n']} net_Rs={m2['net_rs']:,.2f} PF={m2['pf']:.2f} SQN={m2['sqn']:.2f}")

    if r["pf"] != float("inf") and not pd.isna(r["pf"]) and (r["pf"] > 2.5 or (not pd.isna(r["sqn"]) and r["sqn"] > 5)):
        print("\n*** RED FLAG: PF>2.5 or SQN>5 -- extraordinary claim, treat with extra scrutiny "
              "per this project's standing calibration. ***")

    print(f"\nSaved: top4_strategies_combined_trades.csv ({len(tdf)} rows)")
    print("\nReminder: Two-Touch Resistance's filtered numbers are still flagged in the ranking as a "
          "monitored hypothesis (several cuts deep into the same 2-year sample), not a fully validated "
          "edge on its own -- worth keeping that caveat in mind when reading its slice of this combined result.")


if __name__ == "__main__":
    main()
