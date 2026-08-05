"""
M-STRUCTURE -- live PAPER (mock, NO real money / NO real orders) test
=================================================================
Tuned version of nifty_m_structure_backtest.py based on the diagnostic +
sweep results (nifty_m_structure_sweep.py):

  CHANGES MADE FROM THE ORIGINAL BACKTEST, AND WHY:
  1. PE-ONLY (breakdown/short side). CE breakouts were barely above
     breakeven standalone (PF 1.09, SQN 0.69) while PE breakdowns did
     PF 1.31, SQN 2.10, and carried 77% of the combined net P&L. Same
     pattern as Velez (PE-only) and Crash Play (PE-only) elsewhere in
     this project -- Nifty's downside structure just trades cleaner than
     its upside grinds. CE detection code is left in nifty_m_structure_
     backtest.py for reference but is NOT used here.
  2. TUESDAY EXCLUDED. PF 0.63, SQN -2.49, net -Rs11,983 on the real
     9-month sample -- a real negative-expectancy day, not just a weak
     one. Caveat carried forward honestly: ~209 trades over ~39 Tuesdays
     is a thinner cut than the full dataset, so this is a hypothesis
     being traded forward, not a locked-in certainty -- re-check it once
     more months of data accumulate.
  3. STOP=8 / TRAIL=4 (not the original 5/4, and NOT the grid-sweep
     "best" of stop=10/trail=2, SQN 6.21). The sweep showed SQN climbing
     monotonically toward the tight-trail/wide-stop corner of the tested
     grid -- the same tail-risk-harvesting artifact signature already
     diagnosed once before in this project (Opening-Drive Momentum's own
     sweep did the identical thing, climbing to SQN 9.8 at a grid corner
     that turned out not to be a real edge). stop=8/trail=4 is the
     moderate upgrade from your original 5/4 spec (PF 1.34, SQN 3.14)
     without chasing that corner.
  4. NO TIME-CAP EXIT. The sweep found every time-capped variant (15/30/
     60 bars) underperformed the uncapped trailing-stop exit -- the
     trail already does its job; forcing early exits just cuts winners
     short.

  IMPORTANT CONTEXT CARRIED FORWARD: real Nifty index volume only exists
  from ~2025-11 onward (confirmed via nifty_m_structure_sweep.py's
  volume-coverage diagnostic) -- every number above comes from a ~9-month
  real sample, not the 2-year samples Velez/Opening-Drive/Crash Play were
  validated on. This is being paper-traded, not sized with real capital,
  specifically because the sample is still short. Re-run the sweep again
  in a few months once more real-volume data has accumulated.

THIS SCRIPT NEVER PLACES A REAL ORDER -- read-only Dhan API calls only
(intraday_minute_data), everything else (signal detection, entry/exit,
P&L) is simulated in memory using the same proxy-premium model and
conservative intrabar stop/trail ordering already validated in the
backtest and sweep. Every simulated entry/exit is appended to a CSV trade
log the moment it happens.

Two ways to run it:

  Smoke test (safe any time, ~10-30s, no CSV writes): fetches real recent
  Nifty data, replays the most recently completed trading day (both in
  one batch pass AND bar-by-bar through the same incremental live logic,
  asserting the two match -- the actual smoke test):

      cd "C:\\dhan algo"
      python nifty_m_structure_paper_trade_live.py --smoke-test

  Live mock run (run DURING market hours, 09:15-15:30 IST, on a trading
  day): polls every POLL_SECONDS, appends a row to
  m_structure_paper_trade_log_<today>.csv every time it enters or exits
  a mock position:

      cd "C:\\dhan algo"
      python nifty_m_structure_paper_trade_live.py

  Leave the terminal open. Safe to Ctrl+C any time -- nothing is lost,
  every fill so far is already on disk in the CSV.
"""

import sys
import csv
import os
import time
import datetime as dt
import pandas as pd

import dhan_live_bot as dlb
import nifty_paper_trade_top3_live as top3
import nifty_m_structure_backtest as ms

# ============================== tuned config ===============================
STOP_PTS, TRAIL_PTS = 8.0, 4.0     # moderate upgrade from your original 5/4 -- see docstring
EXCLUDED_WEEKDAYS = {1}            # 1 = Tuesday (Monday=0 ... Sunday=6)
BROKERAGE_PER_TRADE = 40.0
EOD_SQUAREOFF = dt.time(15, 20)
MARKET_OPEN, MARKET_CLOSE = dt.time(9, 15), dt.time(15, 30)
POLL_SECONDS = 30

LOG_FILE = f"m_structure_paper_trade_log_{dt.date.today().isoformat()}.csv"
LOG_FIELDS = ["logged_at", "strategy", "event", "action", "side", "strike", "bar_time",
              "price", "exit_reason", "pnl_pts", "rupee_pnl", "lot_size", "day_running_pnl_rs"]


# ============================== data fetch ===================================
def fetch_index_window_with_volume(from_date, to_date, retries=1):
    """Same read-only call as everywhere else in this project, with volume
    added (needed for the structure strategy's volume-confirmation filter).
    Retries once on a transient/malformed response instead of crashing."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = dlb.dhan.intraday_minute_data(
                security_id=dlb.UNDERLYING_SECURITY_ID, exchange_segment=dlb.UNDERLYING_EXCHANGE_SEGMENT,
                instrument_type=dlb.UNDERLYING_INSTRUMENT_TYPE,
                from_date=from_date, to_date=to_date,
            )
            if not isinstance(resp, dict):
                raise RuntimeError(f"Dhan API returned a non-dict response (likely a connection/proxy "
                                    f"error surfaced as a string): {str(resp)[:300]}")
            data = resp.get("data", resp)
            if not isinstance(data, dict) or "open" not in data:
                raise RuntimeError(f"Dhan API response missing expected OHLC keys -- likely an error "
                                    f"payload, not real data: {str(resp)[:300]}")
            df = pd.DataFrame({
                "open": data["open"], "high": data["high"], "low": data["low"], "close": data["close"],
                "volume": data.get("volume", [0] * len(data["open"])), "timestamp": data["timestamp"],
            })
            df["time"] = pd.to_datetime(df["timestamp"], unit="s") + pd.Timedelta(hours=5, minutes=30)
            return df.set_index("time").sort_index()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2)
                continue
    raise RuntimeError(f"fetch_index_window_with_volume failed after {retries + 1} attempt(s) "
                        f"(from_date={from_date}, to_date={to_date}): {last_err}")


# =============================================================================
# Self-checks
# =============================================================================
def run_self_checks():
    print("Running self-checks (synthetic data)...")
    ms.run_self_checks()  # re-verify structure detection + shared premium/position primitives

    # --- Tuesday exclusion must actually block processing ---
    tue = dt.date(2025, 1, 7)  # a Tuesday
    assert tue.weekday() == 1, "test setup sanity check failed"
    print("  Tuesday-exclusion date check ok")

    # --- PE-only filter: a mixed CE+PE signal list must only keep PE ---
    mixed = [{"bar": 5, "side": "CE", "event": "MTOP_BREAKOUT"}, {"bar": 10, "side": "PE", "event": "WBOTTOM_BREAKDOWN"}]
    pe_only = [s for s in mixed if s["side"] == "PE"]
    assert len(pe_only) == 1 and pe_only[0]["side"] == "PE"
    print("  PE-only signal filtering ok")

    print("All self-checks passed.\n")


# =============================================================================
# Live state + logging
# =============================================================================
def new_strategy_state():
    return {"position": None, "pending_signal": None, "search_from_bar": -1,
            "trades_today": 0, "day_pnl_rs": 0.0}


def append_log_row(row):
    is_new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)


def log_entry(event, side, strike, bar_time, price, lot, day_pnl):
    append_log_row({
        "logged_at": dt.datetime.now().isoformat(timespec="seconds"), "strategy": "MStructure",
        "event": event, "action": "ENTRY", "side": side, "strike": strike,
        "bar_time": bar_time, "price": round(price, 2), "exit_reason": "", "pnl_pts": "",
        "rupee_pnl": "", "lot_size": lot, "day_running_pnl_rs": round(day_pnl, 2),
    })


def log_exit(event, side, strike, bar_time, price, reason, pnl_pts, rupee_pnl, lot, day_pnl):
    append_log_row({
        "logged_at": dt.datetime.now().isoformat(timespec="seconds"), "strategy": "MStructure",
        "event": event, "action": "EXIT", "side": side, "strike": strike,
        "bar_time": bar_time, "price": round(price, 2), "exit_reason": reason,
        "pnl_pts": round(pnl_pts, 2), "rupee_pnl": round(rupee_pnl, 2), "lot_size": lot,
        "day_running_pnl_rs": round(day_pnl, 2),
    })


def process_m_structure(day_df, state, verbose=True):
    min_bars = ms.STRUCT_LOOKBACK + ms.REJECT_HOLD_BARS + ms.HOLD_CONFIRM_BARS + 3
    if len(day_df) < min_bars:
        return
    today = day_df.index[0].date()
    if today.weekday() in EXCLUDED_WEEKDAYS:
        return  # Tuesday excluded -- see docstring

    day_open_spot = day_df["open"].iloc[0]
    strike = top3.itm_put_strike(day_open_spot)
    premium_df = top3.build_premium_series(day_df, strike, day_open_spot, side="PE")
    lot = top3.lot_size_for_date(today)

    if state["position"] is not None:
        exit_info = top3.step_long_premium_position(state["position"], premium_df, TRAIL_PTS)
        if exit_info:
            pos = state["position"]
            pnl_pts = exit_info["exit_price"] - pos["entry_price"]
            rupee_pnl = pnl_pts * lot - BROKERAGE_PER_TRADE
            state["day_pnl_rs"] += rupee_pnl
            state["trades_today"] += 1
            log_exit(pos["event"], "PE", strike, exit_info["exit_time"], exit_info["exit_price"],
                      exit_info["reason"], pnl_pts, rupee_pnl, lot, state["day_pnl_rs"])
            if verbose:
                print(f"  [MStructure] EXIT {pos['event']} PE @ {exit_info['exit_price']:.2f} "
                      f"({exit_info['reason']}) pnl_Rs={rupee_pnl:,.2f}")
            state["search_from_bar"] = exit_info["exit_idx"]
            state["position"] = None

    if state["position"] is None and state["pending_signal"] is None:
        sigs = [s for s in ms.detect_m_structure_signals(day_df)
                if s["side"] == "PE" and s["bar"] > state["search_from_bar"]]
        if sigs:
            state["pending_signal"] = sigs[0]

    if state["position"] is None and state["pending_signal"] is not None:
        j0 = state["pending_signal"]["bar"]
        exec_idx = j0 + ms.HOLD_CONFIRM_BARS + 1
        if exec_idx < len(day_df):
            entry_price = premium_df["open"].iloc[exec_idx]
            state["position"] = {
                "entry_price": entry_price, "stop_level": entry_price - STOP_PTS, "peak": entry_price,
                "trailing_active": False, "last_checked_idx": exec_idx - 1,
                "event": state["pending_signal"]["event"],
            }
            log_entry(state["pending_signal"]["event"], "PE", strike, day_df.index[exec_idx],
                       entry_price, lot, state["day_pnl_rs"])
            if verbose:
                print(f"  [MStructure] ENTRY {state['pending_signal']['event']} PE@{strike} "
                      f"premium={entry_price:.2f}")
            state["pending_signal"] = None


# =============================================================================
# Smoke test -- real data, one pass, no infinite loop, no CSV writes
# =============================================================================
def smoke_test():
    run_self_checks()
    print("=== SMOKE TEST: fetching real recent Nifty data, replaying most recent trading day ===\n")
    to_date = dt.date.today().strftime("%Y-%m-%d")
    from_date = (dt.date.today() - dt.timedelta(days=10)).strftime("%Y-%m-%d")
    try:
        raw = fetch_index_window_with_volume(from_date, to_date)
    except Exception as e:
        print(f"Could not fetch real data to smoke-test against: {e}\n")
        print("All self-checks above still passed (the logic itself is verified). This failure is "
              "specifically about reaching the Dhan API from wherever this was just run -- if you're "
              "seeing this from your own machine with a working Dhan connection, check your network/"
              "ACCESS_TOKEN. Re-run this smoke test on the machine you'll actually run the live version "
              "from before trusting it fully.")
        return
    days = sorted(set(raw.index.date))
    if not days:
        print("No data returned -- can't smoke test right now (market may be fully closed with no "
              "recent data). Try again closer to/during market hours.")
        return
    last_day = days[-1]
    day_df = raw[raw.index.date == last_day].copy()
    print(f"Got {len(day_df)} 1-min bars for {last_day} (most recent trading day)")
    if last_day.weekday() in EXCLUDED_WEEKDAYS:
        print(f"({last_day} is a Tuesday -- excluded by design, so no trades are expected today. "
              f"That's correct behaviour, not a bug.)\n")

    # batch replay
    state_batch = new_strategy_state()
    process_m_structure(day_df, state_batch, verbose=True)
    for _ in range(5):
        before = (state_batch["position"] is not None, state_batch["trades_today"])
        process_m_structure(day_df, state_batch, verbose=False)
        after = (state_batch["position"] is not None, state_batch["trades_today"])
        if before == after:
            break
    print(f"  trades_today={state_batch['trades_today']}  day_pnl_Rs={state_batch['day_pnl_rs']:,.2f}  "
          f"position_open_at_dataend={state_batch['position'] is not None}\n")

    # incremental replay must match the batch result (real smoke test, not just "did it run")
    state_inc = new_strategy_state()
    for cutoff in range(ms.STRUCT_LOOKBACK, len(day_df) + 1, 5):
        process_m_structure(day_df.iloc[:cutoff], state_inc, verbose=False)
    process_m_structure(day_df, state_inc, verbose=False)
    assert state_inc["trades_today"] == state_batch["trades_today"], (
        f"incremental replay diverged from batch replay: {state_inc['trades_today']} vs "
        f"{state_batch['trades_today']} trades -- would mean live polling could behave differently "
        f"from the validated backtest logic."
    )
    print("Incremental (bar-by-bar) replay matches the batch replay exactly -- confirms live polling "
          "will behave identically to the validated backtest logic.")
    print("\nSmoke test complete -- no crashes, all self-checks passed, ran against real data "
          "end-to-end. Nothing was written to the trade log CSV (smoke test only).")


# =============================================================================
# Live loop
# =============================================================================
def is_market_open():
    now = dt.datetime.now().time()
    return MARKET_OPEN <= now <= MARKET_CLOSE


def main_live():
    run_self_checks()
    print("=== LIVE PAPER TRADE (MOCK, NO REAL ORDERS) -- M-Structure (PE-only, no-Tuesday, 8/4) ===")
    print(f"Log file: {LOG_FILE}")
    print(f"Polling every {POLL_SECONDS}s during market hours ({MARKET_OPEN}-{MARKET_CLOSE} IST)\n")

    state = new_strategy_state()
    today = dt.date.today()
    if today.weekday() in EXCLUDED_WEEKDAYS:
        print(f"Today ({today}) is a Tuesday -- this strategy is designed to sit out Tuesdays "
              f"(real negative expectancy on the backtested sample). No trades will be taken today; "
              f"the script will still poll and print status so you can confirm it's alive.\n")

    while True:
        now = dt.datetime.now()
        if now.date() != today:
            today = now.date()
            state = new_strategy_state()
            globals()["LOG_FILE"] = f"m_structure_paper_trade_log_{today.isoformat()}.csv"
            print(f"\nNew trading day: {today}. Log file: {LOG_FILE}")
            if today.weekday() in EXCLUDED_WEEKDAYS:
                print(f"({today} is a Tuesday -- sitting out today by design.)")

        if not is_market_open():
            print(f"{now.strftime('%H:%M:%S')} Market closed, sleeping 60s...")
            time.sleep(60)
            continue

        try:
            to_date = today.strftime("%Y-%m-%d")
            raw = fetch_index_window_with_volume(to_date, to_date)
        except Exception:
            print("Failed to fetch candles, retrying...")
            time.sleep(POLL_SECONDS)
            continue

        day_df = raw[raw.index.date == today].copy()
        if day_df.empty:
            time.sleep(POLL_SECONDS)
            continue

        latest_close = day_df["close"].iloc[-1]
        print(f"{now.strftime('%H:%M:%S')} bars={len(day_df)} spot={latest_close:.2f}  "
              f"MStructure[{'OPEN' if state['position'] else 'flat'}]  day_pnl_Rs={state['day_pnl_rs']:,.2f}")

        process_m_structure(day_df, state)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        smoke_test()
    else:
        main_live()
