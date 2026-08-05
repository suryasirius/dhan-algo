"""
Top 2 Ranked Strategies COMBINED -- Velez 8pt + Opening-Drive Momentum
(CE-ONLY variant) -- standalone comparison test
=================================================================
Follow-up to top2_strategies_combined_backtest.py. That run's real
2-year numbers showed Opening-Drive's PUT (PE) side is meaningfully
weaker than its CALL (CE) side:

    CE: n=229, win%=53.7, PF=1.66, net_Rs=31,257.74
    PE: n=266, win%=47.4, PF=1.27, net_Rs=17,539.00

Per your request, this tests DROPPING the PE side of Opening-Drive
entirely and compares net P&L against the original CE+PE combined
version.

IMPORTANT: this is NOT the same as just deleting the PE rows from the
existing trade CSV. find_opening_drive() returns on the FIRST >=5pt
move away from the open, whichever direction comes first. On days
where the first qualifying move was a down-move (PE trigger), the
original logic stops searching right there -- it never looks for a
later up-move before the 09:45 cutoff. A true CE-only backtest must
keep scanning past a down-move and only take a signal on an up-move,
so it can pick up trades on days the original version never traded.
That's why this is run as a real, independent backtest, not a filter.

Does NOT touch dhan_live_bot.py, nifty_paper_trade_top3_live.py, or
top2_strategies_combined_backtest.py (all locked/already validated).
This is a fully standalone, non-destructive script that imports
nifty_paper_trade_top3_live and reuses its Velez/Opening-Drive
position-management, premium-building, and logging machinery verbatim.

The ONLY new logic here is find_opening_drive_ce_only(), a drop-in
replacement for top3.find_opening_drive() that behaves identically
except it ignores down-moves instead of triggering a PE entry on them
and keeps searching for an up-move until the 09:45 cutoff. It is
installed via a temporary monkeypatch of top3.find_opening_drive
(restored after the run), so process_opening_drive() itself -- which
already handles "side = CE if direction==LONG else PE" -- is reused
completely unmodified; since our patched finder only ever returns
"LONG" or None, side will always resolve to CE.

Same 2-year window, same Rs25,000 starting capital, same reuse-via-
import-and-monkeypatch pattern already established in this project.
"""

import datetime as dt
import time
import pandas as pd

import nifty_paper_trade_top3_live as top3

FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

STARTING_CAPITAL = 25_000
MAX_REPEATS_PER_DAY = 12

_orig_find_opening_drive = top3.find_opening_drive


def find_opening_drive_ce_only(day_df, min_move=top3.OD_MIN_MOVE_PTS, cutoff=top3.OD_ENTRY_CUTOFF):
    """Drop-in replacement for top3.find_opening_drive(): only ever
    returns a LONG (CE) signal. A down-move that would have triggered
    a PE entry in the original is ignored -- scanning continues past
    it, still looking for an up-move before the cutoff."""
    day_open = day_df["open"].iloc[0]
    closes = day_df["close"].to_numpy()
    times = day_df.index
    for i in range(len(day_df)):
        if times[i].time() >= cutoff:
            return None, None
        move = closes[i] - day_open
        if move >= min_move:
            return i, "LONG"
    return None, None


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


def run_self_checks():
    print("Running self-checks (synthetic data)...")

    top3.run_self_checks()
    print("  top3.run_self_checks() re-verified ok (Velez/OpeningDrive primitives)")

    # --- find_opening_drive_ce_only behavior checks ---
    idx = pd.date_range("2025-01-06 09:15", periods=40, freq="1min")

    # Case A: original would trigger PE (down-move first) -- CE-only
    # version must NOT trigger there, and since no later up-move occurs
    # before cutoff, must return None entirely.
    closes = [25000.0] * 40
    closes[5] = 25000.0 - 6.0   # down-move at bar 5, big enough to trigger SHORT in original
    for i in range(6, 40):
        closes[i] = 25000.0 - 6.0  # stays down, no up-move before cutoff
    df = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes}, index=idx)
    df.loc[df.index[0], "open"] = 25000.0
    orig_i, orig_dir = _orig_find_opening_drive(df)
    assert orig_dir == "SHORT", f"expected original to trigger SHORT, got {orig_dir}"
    ce_i, ce_dir = find_opening_drive_ce_only(df)
    assert ce_i is None and ce_dir is None, f"expected CE-only to find nothing, got {(ce_i, ce_dir)}"
    print("  Case A (down-move only, no later up-move): original=SHORT, CE-only=None -- ok")

    # Case B: down-move first (would trigger PE in original), THEN an
    # up-move later before cutoff -- CE-only version must skip the
    # down-move and pick up the later up-move as a LONG signal.
    closes2 = [25000.0] * 40
    closes2[5] = 25000.0 - 6.0
    for i in range(6, 15):
        closes2[i] = 25000.0 - 6.0
    closes2[20] = 25000.0 + 7.0
    for i in range(21, 40):
        closes2[i] = 25000.0 + 7.0
    df2 = pd.DataFrame({"open": closes2, "high": closes2, "low": closes2, "close": closes2}, index=idx)
    orig_i2, orig_dir2 = _orig_find_opening_drive(df2)
    assert orig_dir2 == "SHORT" and orig_i2 == 5
    ce_i2, ce_dir2 = find_opening_drive_ce_only(df2)
    assert ce_dir2 == "LONG" and ce_i2 == 20, f"expected LONG at bar 20, got {(ce_i2, ce_dir2)}"
    print("  Case B (down-move then later up-move): original=SHORT@5, CE-only=LONG@20 -- ok")

    # Case C: up-move first -- both versions must agree (CE-only should
    # behave identically to original when the first qualifying move is
    # already an up-move).
    closes3 = [25000.0] * 40
    closes3[4] = 25000.0 + 6.0
    for i in range(5, 40):
        closes3[i] = 25000.0 + 6.0
    df3 = pd.DataFrame({"open": closes3, "high": closes3, "low": closes3, "close": closes3}, index=idx)
    orig_i3, orig_dir3 = _orig_find_opening_drive(df3)
    ce_i3, ce_dir3 = find_opening_drive_ce_only(df3)
    assert orig_dir3 == "LONG" and ce_dir3 == "LONG" and orig_i3 == ce_i3 == 4
    print("  Case C (up-move first): both agree LONG@4 -- ok")

    # --- process_opening_drive with the monkeypatched finder installed:
    # verify a full entry+exit round-trip only ever produces CE trades.
    # Extend the up-move day through 15:20 so the position is forced to
    # an EOD exit -- otherwise a completed (entry+exit) trade never gets
    # logged and the check below would be trivially/vacuously true. ---
    idx_full_day = pd.date_range("2025-01-06 09:15", "2025-01-06 15:25", freq="1min")
    closes3_full = [25000.0] * len(idx_full_day)
    up_from = 4
    for i in range(len(closes3_full)):
        closes3_full[i] = 25000.0 + (6.0 if i >= up_from else 0.0)
    df3_full = pd.DataFrame({"open": closes3_full, "high": closes3_full,
                             "low": closes3_full, "close": closes3_full}, index=idx_full_day)

    global _captured_trades, _pending_entries
    _captured_trades, _pending_entries = [], {}
    top3.log_entry = _capture_log_entry
    top3.log_exit = _capture_log_exit
    top3.find_opening_drive = find_opening_drive_ce_only
    st = top3.new_strategy_state()
    for _ in range(MAX_REPEATS_PER_DAY):
        before = (st["position"] is not None, st["trades_today"])
        top3.process_opening_drive(df3_full, st, verbose=False)
        after = (st["position"] is not None, st["trades_today"])
        if before == after:
            break
    assert len(_captured_trades) >= 1, "expected at least one completed (entry+exit) trade in this check"
    assert all(t["side"] == "CE" for t in _captured_trades), \
        f"expected only CE trades, got sides={[t['side'] for t in _captured_trades]}"
    print(f"  process_opening_drive w/ patched finder produced {len(_captured_trades)} completed trade(s), all CE -- ok")
    top3.find_opening_drive = _orig_find_opening_drive
    top3.log_entry, top3.log_exit = _orig_log_entry, _orig_log_exit
    _captured_trades, _pending_entries = [], {}

    # --- capture-hook pairing check (same as top2_strategies_combined) ---
    _captured_trades, _pending_entries = [], {}
    top3.log_entry = _capture_log_entry
    top3.log_exit = _capture_log_exit
    t0 = pd.Timestamp("2025-01-06 09:20")
    t1 = pd.Timestamp("2025-01-06 09:30")
    top3.log_entry("TestStrat", "EVT", "CE", 25000, t0, 100.0, 75, 0.0)
    top3.log_exit("TestStrat", "EVT", "CE", 25000, t1, 108.0, "TRAIL_SL", 8.0, 559.0, 75, 559.0)
    assert len(_captured_trades) == 1
    tr = _captured_trades[0]
    assert tr["entry_time"] == t0 and tr["exit_time"] == t1
    assert tr["rupee_pnl"] == 559.0
    print("  log_entry/log_exit capture-hook pairing check ok")
    _captured_trades, _pending_entries = [], {}
    top3.log_entry, top3.log_exit = _orig_log_entry, _orig_log_exit

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
    print("=== Top 2 Ranked Strategies COMBINED -- Velez 8pt + Opening-Drive (CE-ONLY variant) -- 2-year backtest ===\n")
    print("Opening-Drive PUT (PE) side is DISABLED for this run -- only CALL (CE) entries allowed.\n")

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
    top3.find_opening_drive = find_opening_drive_ce_only  # <-- the one behavior change

    t_start = time.time()

    for di, day in enumerate(days):
        day_df = full_2min[full_2min.index.date == day].copy()
        day_df_velez = full_velez[full_velez.index.date == day].copy()
        if day_df.empty:
            continue

        states = {
            "velez": top3.new_strategy_state(),
            "od": top3.new_strategy_state(),
        }
        for name, fn, df_to_use in (
            ("velez", top3.process_velez, day_df_velez),
            ("od", top3.process_opening_drive, day_df),
        ):
            st = states[name]
            for _ in range(MAX_REPEATS_PER_DAY):
                before = (st["position"] is not None, st["trades_today"])
                fn(df_to_use, st, verbose=False)
                after = (st["position"] is not None, st["trades_today"])
                if before == after:
                    break

    top3.log_entry, top3.log_exit = _orig_log_entry, _orig_log_exit
    top3.find_opening_drive = _orig_find_opening_drive
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

    if not combined:
        print("No trades generated across either strategy.")
        return

    tdf = pd.DataFrame(combined).sort_values("entry_time").reset_index(drop=True)
    tdf.to_csv("top2_ce_only_opening_drive_trades.csv", index=False)

    r = stats_for(tdf)
    print("--- COMBINED (Velez 8pt + Opening-Drive CE-ONLY, one shared Rs25,000 account) ---")
    print(f"n={r['n']}  win%={r['win_pct']:.1f}  PF={r['pf']:.2f}  net_Rs={r['net_rs']:,.2f}  "
          f"maxDD_Rs={r['max_dd']:,.2f}  SQN={r['sqn']:.2f}  final_equity_Rs={r['final_equity']:,.2f}\n")

    print("--- By strategy (isolated stats) ---")
    for strat, g in tdf.groupby("strategy"):
        rg = stats_for(g)
        print(f"  [{strat}] n={rg['n']}  win%={rg['win_pct']:.1f}  PF={rg['pf']:.2f}  "
              f"net_Rs={rg['net_rs']:,.2f}  SQN={rg['sqn']:.2f}")

    od = tdf[tdf["strategy"] == "OpeningDrive"]
    assert (od["side"] == "CE").all(), "BUG: a PE trade leaked into the CE-only Opening-Drive run"
    print(f"\n  Confirmed: all {len(od)} Opening-Drive trades in this run are CE (PE side fully disabled).")

    print("\n--- By exit reason (combined) ---")
    for reason, g in tdf.groupby("exit_reason"):
        print(f"  {reason}: n={len(g)} ({100*len(g)/len(tdf):.1f}%)  win%={100*(g['rupee_pnl']>0).mean():.1f}  "
              f"net_Rs={g['rupee_pnl'].sum():,.2f}")

    events = []
    for _, row in tdf.iterrows():
        events.append((row["entry_time"], 1))
        events.append((row["exit_time"], -1))
    events.sort()
    concurrent, max_concurrent = 0, 0
    for _, delta in events:
        concurrent += delta
        max_concurrent = max(max_concurrent, concurrent)
    print(f"\n--- Overlap diagnostic ---\nMax simultaneous open positions across both strategies: {max_concurrent}")

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

    print(f"\nSaved: top2_ce_only_opening_drive_trades.csv ({len(tdf)} rows)")

    print("\n=== COMPARISON vs original CE+PE combined run (top2_strategies_combined_backtest.py) ===")
    print("  Original (CE+PE):  n=779  PF=1.51  net_Rs=75,912.81  SQN=4.02  "
          "[OpeningDrive alone: n=495 PF=1.44 net_Rs=48,796.74]")
    print(f"  CE-ONLY (this run): n={r['n']}  PF={r['pf']:.2f}  net_Rs={r['net_rs']:,.2f}  SQN={r['sqn']:.2f}  "
          f"[OpeningDrive alone: n={rg['n'] if 'OpeningDrive' in tdf['strategy'].values else 0}]")


if __name__ == "__main__":
    main()
