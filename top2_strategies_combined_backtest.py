"""
Top 2 Ranked Strategies COMBINED -- Velez 8pt + Opening-Drive Momentum --
independent, overlap allowed -- 2-year backtest
=================================================================
Follow-up to top4_strategies_combined_backtest.py: that run showed
Crash Play and Two-Touch Resistance together contributed only ~7% of
total profit while accounting for ~30% of trades (Crash Play alone
stopped out on 50.6% of its own trades) -- real activity and exposure
for very little payoff. Per your instruction, both are DROPPED here.
This keeps only the two strategies that were actually carrying the
combined portfolio:

  1. Velez 8pt (short)      -- always buys the ITM PUT
  2. Opening-Drive Momentum -- buys ITM CALL or PUT depending on which
                                way the first opening move goes

Does NOT touch dhan_live_bot.py or nifty_paper_trade_top3_live.py
(both locked/already validated, per project convention). Instead this
IMPORTS nifty_paper_trade_top3_live and reuses its real, already-tested
Velez/Opening-Drive signal-detection and trade-management functions
verbatim -- zero hand-transcription, zero re-derivation risk. The only
new code here is the day-by-day driver loop and a small hook that
captures log_entry/log_exit calls into an in-memory trade list instead
of writing to disk.

INDEPENDENT + OVERLAP ALLOWED: each of the two strategies runs with its
own persistent state (own open-position tracking) and is driven
separately every day. If both want to be in a trade at the same time,
that's fine -- each opens its own position; one never blocks the
other. The only restriction is the normal one already built into each
strategy: a SINGLE strategy can't open a second trade while its own
current trade is still open.

Both strategies' trades are combined onto ONE shared Rs25,000 equity
curve (sorted by time), so the backtest reflects what a single account
running both at once would actually look like.

2-year window, starting capital Rs 25,000, per your request.
"""

import datetime as dt
import time
import pandas as pd

import nifty_paper_trade_top3_live as top3

FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

STARTING_CAPITAL = 25_000
MAX_REPEATS_PER_DAY = 12  # generous cap on round-trips/day per strategy (smoke_test used 6)

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


def run_self_checks():
    print("Running self-checks (synthetic data)...")

    # re-verify the underlying, already-validated module's own self-checks
    top3.run_self_checks()
    print("  top3.run_self_checks() re-verified ok (Velez/OpeningDrive primitives)")

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
    print("=== Top 2 Ranked Strategies COMBINED -- independent, overlap allowed -- 2-year backtest ===\n")
    print("Strategies: Velez 8pt (short), Opening-Drive Momentum "
          "(Crash Play and Two-Touch Resistance dropped per your instruction)\n")

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

    t_start = time.time()

    for di, day in enumerate(days):
        day_df = full_2min[full_2min.index.date == day].copy()
        day_df_velez = full_velez[full_velez.index.date == day].copy()
        if day_df.empty:
            continue

        # Fresh per-day state for each of the 2 strategies -- safe because
        # step_long_premium_position forces an EOD exit by 15:20, so no
        # position ever survives past the end of a day's own data (verified
        # in its own self-checks).
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
    tdf.to_csv("top2_strategies_combined_trades.csv", index=False)

    r = stats_for(tdf)
    print("--- COMBINED (Velez 8pt + Opening-Drive Momentum, one shared Rs25,000 account) ---")
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

    print(f"\nSaved: top2_strategies_combined_trades.csv ({len(tdf)} rows)")


if __name__ == "__main__":
    main()
