"""
Live PAPER (mock, NO real money / NO real orders) -- TOP 3: Velez 8pt +
Opening-Drive CE-only + Two-Touch Resistance Multi-Entry Optimized
=================================================================
Live counterpart to top3_ce_only_od_twotouch_combined_backtest.py. Same
three strategies, same reuse-via-import approach, no locked file touched.

Velez + Opening-Drive use the existing incremental, position-aware live
loop (nifty_paper_trade_top2_ce_only_live.py's pattern -- state persists
across polls, one entry/exit resolved per call). Two-Touch's day-runner
(nifty_two_touch_resistance_multi_entry_backtest.run_standalone_for_day_multi)
is NOT incremental -- it recomputes the whole day's pattern from scratch
each call. So each poll: rerun it on today's data so far, apply the
optimized filter (entry cap #1-2/day, Wednesday + 13:15-13:45 excluded),
and log only trades whose entry_time hasn't already been logged today
(a simple per-day "seen" set, since re-running produces the exact same
completed trades every time -- nothing invented, just deduped for the log).

Log file: paper_trade_top3_ce_only_od_twotouch_log_<date>.csv

Two ways to run it:
    cd "C:\\dhan algo"
    python nifty_paper_trade_top3_ce_only_od_twotouch_live.py --smoke-test
    python nifty_paper_trade_top3_ce_only_od_twotouch_live.py
"""

import sys
import csv
import os
import time
import datetime as dt

import nifty_paper_trade_top3_live as top3
import top2_ce_only_opening_drive_backtest as top2
import nifty_two_touch_resistance_multi_entry_backtest as multi
import nifty_two_touch_resistance_multi_entry_optimized_backtest as opt

LOG_FILE_PREFIX = "paper_trade_top3_ce_only_od_twotouch_log_"
TT_LOG_FIELDS = ["logged_at", "strategy", "action", "side", "strike", "bar_time",
                  "price", "exit_reason", "pnl_pts", "rupee_pnl", "lot_size", "day_running_pnl_rs"]


def _set_log_file_for(day):
    top3.LOG_FILE = f"{LOG_FILE_PREFIX}{day.isoformat()}.csv"


def _append_tt_exit_row(t):
    is_new = not os.path.exists(top3.LOG_FILE)
    with open(top3.LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TT_LOG_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow({
            "logged_at": dt.datetime.now().isoformat(timespec="seconds"),
            "strategy": "TwoTouchOptimized", "action": "EXIT", "side": t["side"], "strike": t["strike"],
            "bar_time": t["exit_time"], "price": round(t["exit_price"], 2), "exit_reason": t["reason"],
            "pnl_pts": round(t["pnl_pts"], 2), "rupee_pnl": round(t["rupee_pnl"], 2),
            "lot_size": "", "day_running_pnl_rs": "",
        })


def run_self_checks():
    print("Running self-checks (synthetic data)...")
    top2.run_self_checks()
    print("  top2.run_self_checks() re-verified ok (Velez + Opening-Drive CE-only)")
    opt.run_self_checks()
    print("  opt.run_self_checks() re-verified ok (Two-Touch multi-entry, optimized filter)")
    print("All self-checks passed.\n")


def smoke_test():
    run_self_checks()
    print("=== SMOKE TEST: fetching real recent Nifty data, replaying most recent trading day ===\n")
    print("Strategies: Velez 8pt (short), Opening-Drive CE-only, Two-Touch Optimized (multi-entry)\n")
    to_date = dt.date.today().strftime("%Y-%m-%d")
    from_date = (dt.date.today() - dt.timedelta(days=top3.HISTORY_CONTEXT_DAYS)).strftime("%Y-%m-%d")
    try:
        raw = top3.fetch_index_window(from_date, to_date)
    except Exception as e:
        print(f"Could not fetch real data to smoke-test against: {e}\n")
        print("All self-checks above still passed. Re-run this on the machine with a working Dhan "
              "connection before trusting the smoke test fully.")
        return
    full = top3.resample_2min_per_day(raw)
    full_velez = top3.compute_velez_indicators(full)
    days = sorted(set(full.index.date))
    if not days:
        print("No data returned -- market may be fully closed with no recent data. Try again later.")
        return
    last_day = days[-1]
    day_df_1min = raw[raw.index.date == last_day].copy()
    day_df_2min = full[full.index.date == last_day].copy()
    day_df_velez = full_velez[full_velez.index.date == last_day].copy()
    print(f"Replaying {last_day} (most recent day, {len(day_df_1min)} 1-min / {len(day_df_2min)} 2-min bars)\n")

    orig_find = top3.find_opening_drive
    top3.find_opening_drive = top2.find_opening_drive_ce_only
    try:
        for name, fn, df_to_use in (("Velez8pt", top3.process_velez, day_df_velez),
                                     ("OpeningDrive-CE-only", top3.process_opening_drive, day_df_2min)):
            print(f"--- {name} ---")
            st = top3.new_strategy_state()
            for _ in range(6):
                before = (st["position"] is not None, st["trades_today"])
                fn(df_to_use, st, verbose=True)
                after = (st["position"] is not None, st["trades_today"])
                if before == after:
                    break
            print(f"  trades_today={st['trades_today']}  day_pnl_Rs={st['day_pnl_rs']:,.2f}\n")
    finally:
        top3.find_opening_drive = orig_find

    print("--- Two-Touch Optimized (multi-entry, filtered) ---")
    tt_raw = multi.run_standalone_for_day_multi(day_df_1min)
    tt_df = opt.apply_filters(__import__("pandas").DataFrame(tt_raw)) if tt_raw else __import__("pandas").DataFrame()
    print(f"  {len(tt_raw)} raw pattern(s) found, {len(tt_df)} survive the filter (entry #1-2, "
          f"not Wednesday, not 13:15-13:45)")
    for _, t in tt_df.iterrows():
        print(f"  {t['side']}@{t['strike']} entry={t['entry_price']:.2f} exit={t['exit_price']:.2f} "
              f"({t['reason']}) pnl_Rs={t['rupee_pnl']:,.2f}")

    print("\nSmoke test complete -- no crashes, all self-checks passed, all three strategies ran against "
          "real data end-to-end. Nothing was written to the trade log CSV (smoke test only).")


def is_market_open():
    now = dt.datetime.now().time()
    return top3.MARKET_OPEN <= now <= top3.MARKET_CLOSE


def main_live():
    run_self_checks()
    today = dt.date.today()
    _set_log_file_for(today)
    top3.find_opening_drive = top2.find_opening_drive_ce_only

    print("=== LIVE PAPER TRADE (MOCK, NO REAL ORDERS) -- Velez8pt / OpeningDrive-CE-only / TwoTouchOptimized ===")
    print(f"Log file: {top3.LOG_FILE}")
    print(f"Polling every {top3.POLL_SECONDS}s during market hours ({top3.MARKET_OPEN}-{top3.MARKET_CLOSE} IST)\n")

    states = {"velez": top3.new_strategy_state(), "od": top3.new_strategy_state()}
    tt_logged_entry_times = set()

    while True:
        now = dt.datetime.now()
        if now.date() != today:
            today = now.date()
            states = {"velez": top3.new_strategy_state(), "od": top3.new_strategy_state()}
            tt_logged_entry_times = set()
            _set_log_file_for(today)
            print(f"\nNew trading day: {today}. Log file: {top3.LOG_FILE}")

        if not is_market_open():
            print(f"{now.strftime('%H:%M:%S')} Market closed, sleeping 60s...")
            time.sleep(60)
            continue

        try:
            to_date = today.strftime("%Y-%m-%d")
            from_date = (today - dt.timedelta(days=top3.HISTORY_CONTEXT_DAYS)).strftime("%Y-%m-%d")
            raw = top3.fetch_index_window(from_date, to_date)
            full = top3.resample_2min_per_day(raw)
        except Exception:
            print("Failed to fetch candles, retrying...")
            time.sleep(top3.POLL_SECONDS)
            continue

        day_df_2min = full[full.index.date == today].copy()
        if day_df_2min.empty:
            time.sleep(top3.POLL_SECONDS)
            continue
        day_df_1min = raw[raw.index.date == today].copy()
        day_df_velez = top3.compute_velez_indicators(full)
        day_df_velez = day_df_velez[day_df_velez.index.date == today].copy()

        latest_close = day_df_2min["close"].iloc[-1]
        print(f"{now.strftime('%H:%M:%S')} bars={len(day_df_2min)} spot={latest_close:.2f}  "
              f"Velez[{'OPEN' if states['velez']['position'] else 'flat'}]  "
              f"OD-CE[{'OPEN' if states['od']['position'] else 'flat'}]")

        top3.process_velez(day_df_velez, states["velez"])
        top3.process_opening_drive(day_df_2min, states["od"])

        try:
            tt_raw = multi.run_standalone_for_day_multi(day_df_1min)
            tt_df = opt.apply_filters(__import__("pandas").DataFrame(tt_raw)) if tt_raw else None
            if tt_df is not None and not tt_df.empty:
                for _, t in tt_df.iterrows():
                    key = str(t["entry_time"])
                    if key not in tt_logged_entry_times:
                        tt_logged_entry_times.add(key)
                        _append_tt_exit_row(t)
                        print(f"  [TwoTouchOptimized] {t['side']}@{t['strike']} entry={t['entry_price']:.2f} "
                              f"exit={t['exit_price']:.2f} ({t['reason']}) pnl_Rs={t['rupee_pnl']:,.2f}")
        except Exception as e:
            print(f"  [TwoTouchOptimized] scan failed this poll ({e}), will retry next poll")

        time.sleep(top3.POLL_SECONDS)


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        smoke_test()
    else:
        main_live()
