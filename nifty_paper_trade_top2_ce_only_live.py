"""
Live PAPER (mock, NO real money / NO real orders) test -- TOP 2 strategies
ONLY, Opening-Drive Momentum in its CE-ONLY variant
=================================================================
Per your instruction to adopt the CE-only Opening-Drive configuration
going forward, after comparing it against the original CE+PE version
over the real 2-year backtest:

    CE+PE (original):  n=779  PF=1.51  net_Rs=75,912.81  maxDD_Rs=-4,936.29  SQN=4.02
    CE-only (chosen):  n=618  PF=1.66  net_Rs=71,037.86  maxDD_Rs=-4,071.34  SQN=4.37

CE-only wins on PF, SQN and drawdown at a ~6% cost in absolute 2-year
profit -- your call. This live/paper script runs ONLY:

  1. Velez 8pt (short)              -- always buys the ITM PUT, 8pt stop/4pt trail
  2. Opening-Drive Momentum, CE-only -- buys the ITM CALL only; a down-move
     no longer triggers a PUT entry, it's ignored and scanning continues
     for a later up-move before the 09:45 cutoff (identical logic to
     find_opening_drive_ce_only, already self-checked and 2yr-backtested
     in top2_ce_only_opening_drive_backtest.py)

Crash Play and Two-Touch Resistance are NOT included (dropped earlier
per your instruction -- see top2_strategies_combined_backtest.py).

Does NOT touch nifty_paper_trade_top3_live.py or
top2_ce_only_opening_drive_backtest.py (both locked/already validated).
This IMPORTS both and reuses their real signal-detection, position-
management, self-check, and CSV-logging machinery verbatim. The only
new code here is a dedicated live loop/smoke test that excludes Crash
Play and writes to its OWN log file (so it never collides with the
original 3-strategy live script's log).

THIS SCRIPT NEVER PLACES A REAL ORDER -- same read-only Dhan API calls,
same proxy-premium simulation, same conservative intrabar stop/trail
ordering already validated in the backtest.

Two ways to run it:

  Smoke test (safe any time, market open or closed, no infinite loop,
  doesn't touch the CSV log):
      cd "C:\\dhan algo"
      python nifty_paper_trade_top2_ce_only_live.py --smoke-test

  Live mock run (run DURING market hours, 09:15-15:30 IST, on a trading
  day): polls every 30s, prints a status line each poll, and appends a
  row to paper_trade_top2_ce_only_log_<today's date>.csv every time
  either strategy enters or exits a mock position:
      cd "C:\\dhan algo"
      python nifty_paper_trade_top2_ce_only_live.py

  Leave the terminal window open. Safe to Ctrl+C any time -- nothing is
  lost, every fill so far is already on disk in the CSV.
"""

import sys
import time
import datetime as dt

import nifty_paper_trade_top3_live as top3
from top2_ce_only_opening_drive_backtest import find_opening_drive_ce_only
import top2_ce_only_opening_drive_backtest as ce_mod

LOG_FILE_PREFIX = "paper_trade_top2_ce_only_log_"


def _set_log_file_for(day):
    # Same technique top3.main_live() already uses for its own daily
    # rotation (globals()["LOG_FILE"] = ...) -- reassigning top3.LOG_FILE
    # from here works identically because append_log_row/log_entry/
    # log_exit are all defined inside top3 and read the module-global
    # LOG_FILE by name at call time.
    top3.LOG_FILE = f"{LOG_FILE_PREFIX}{day.isoformat()}.csv"


def run_self_checks():
    print("Running self-checks (synthetic data)...")
    top3.run_self_checks()
    print("  top3.run_self_checks() re-verified ok (Velez/OpeningDrive primitives)")
    ce_mod.run_self_checks()
    print("  top2_ce_only_opening_drive_backtest.run_self_checks() re-verified ok "
          "(CE-only finder + patched process_opening_drive)")
    print("All self-checks passed.\n")


def smoke_test():
    run_self_checks()
    print("=== SMOKE TEST: fetching real recent Nifty data, replaying most recent trading day ===\n")
    print("Strategies: Velez 8pt (short), Opening-Drive Momentum CE-ONLY "
          "(Crash Play and Two-Touch Resistance excluded)\n")
    to_date = dt.date.today().strftime("%Y-%m-%d")
    from_date = (dt.date.today() - dt.timedelta(days=top3.HISTORY_CONTEXT_DAYS)).strftime("%Y-%m-%d")
    try:
        raw = top3.fetch_index_window(from_date, to_date)
    except Exception as e:
        print(f"Could not fetch real data to smoke-test against: {e}\n")
        print("All self-checks above still passed (the logic itself is verified). This failure is "
              "specifically about reaching the Dhan API from wherever this was just run -- if you're "
              "seeing this from your own machine with a working Dhan connection, check your network/"
              "ACCESS_TOKEN. Re-run this smoke test on the machine you'll actually run the live version "
              "from before trusting it fully.")
        return
    full = top3.resample_2min_per_day(raw)
    full_velez = top3.compute_velez_indicators(full)
    days = sorted(set(full.index.date))
    if not days:
        print("No data returned -- can't smoke test right now (market may be fully closed with no recent "
              "data). This is a data-availability issue, not necessarily a bug -- try again closer to/"
              "during market hours.")
        return
    last_day = days[-1]
    print(f"Got {len(full)} 2-min bars across {len(days)} day(s), replaying {last_day} (most recent)\n")

    day_df = full[full.index.date == last_day].copy()
    day_df_velez = full_velez[full_velez.index.date == last_day].copy()

    orig_find = top3.find_opening_drive
    top3.find_opening_drive = find_opening_drive_ce_only
    try:
        for name, fn, df_to_use in (
            ("Velez8pt", top3.process_velez, day_df_velez),
            ("OpeningDrive-CE-only", top3.process_opening_drive, day_df),
        ):
            print(f"--- {name} ---")
            state_batch = top3.new_strategy_state()
            fn(df_to_use, state_batch, verbose=True)
            for _ in range(5):
                before = (state_batch["position"] is not None, state_batch["trades_today"])
                fn(df_to_use, state_batch, verbose=False)
                after = (state_batch["position"] is not None, state_batch["trades_today"])
                if before == after:
                    break
            print(f"  trades_today={state_batch['trades_today']}  day_pnl_Rs={state_batch['day_pnl_rs']:,.2f}  "
                  f"position_open_at_dataend={state_batch['position'] is not None}\n")
    finally:
        top3.find_opening_drive = orig_find

    print("Smoke test complete -- no crashes, all self-checks passed, both strategies ran against "
          "real data end-to-end. Nothing was written to the trade log CSV (smoke test only).")


def is_market_open():
    now = dt.datetime.now().time()
    return top3.MARKET_OPEN <= now <= top3.MARKET_CLOSE


def main_live():
    run_self_checks()
    today = dt.date.today()
    _set_log_file_for(today)
    top3.find_opening_drive = find_opening_drive_ce_only  # stays patched for this process's lifetime

    print("=== LIVE PAPER TRADE (MOCK, NO REAL ORDERS) -- Velez8pt / OpeningDrive(CE-ONLY) ===")
    print(f"Log file: {top3.LOG_FILE}")
    print(f"Polling every {top3.POLL_SECONDS}s during market hours ({top3.MARKET_OPEN}-{top3.MARKET_CLOSE} IST)\n")

    states = {"velez": top3.new_strategy_state(), "od": top3.new_strategy_state()}

    while True:
        now = dt.datetime.now()
        if now.date() != today:
            today = now.date()
            states = {"velez": top3.new_strategy_state(), "od": top3.new_strategy_state()}
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

        day_df = full[full.index.date == today].copy()
        if day_df.empty:
            time.sleep(top3.POLL_SECONDS)
            continue
        day_df_velez = top3.compute_velez_indicators(full)
        day_df_velez = day_df_velez[day_df_velez.index.date == today].copy()

        latest_close = day_df["close"].iloc[-1]
        print(f"{now.strftime('%H:%M:%S')} bars={len(day_df)} spot={latest_close:.2f}  "
              f"Velez[{'OPEN' if states['velez']['position'] else 'flat'}]  "
              f"OD-CE[{'OPEN' if states['od']['position'] else 'flat'}]")

        top3.process_velez(day_df_velez, states["velez"])
        top3.process_opening_drive(day_df, states["od"])

        time.sleep(top3.POLL_SECONDS)


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        smoke_test()
    else:
        main_live()
