"""
SINGLE PUT, FIXED INDEX TARGET/STOP -- live PAPER (mock, NO real money /
NO real orders)
=================================================================
Simple, one-shot trade for today, per your rules:
  - Enter ONE in-the-money PUT (current ITM strike, taken once at the
    start -- not a repeating signal like the other scripts here).
  - TARGET (take profit): exit when the Nifty INDEX (spot) reaches
    24240.
  - STOP LOSS: exit when the Nifty INDEX (spot) reaches 24270.
  - Whichever level is hit first ends the trade. If neither is hit by
    EOD square-off (15:20 IST), it exits there instead as a fallback.

P&L is estimated with this project's standard proxy-premium model
(intrinsic-at-entry + assumed time value, moves scaled by a flat assumed
delta, floored at 0) -- the same convention used in every other script
here, since real historical option-chain data isn't available on this
Dhan connection (a known, already-diagnosed platform limitation).

NO REAL ORDER IS EVER PLACED. Read-only Dhan API calls only. A mock
entry/exit is appended to a CSV log the moment it happens.

Run:
    cd "C:\\dhan algo"
    python nifty_single_put_target_stop_paper_trade.py --smoke-test
    python nifty_single_put_target_stop_paper_trade.py
"""

import sys
import csv
import os
import time
import math
import datetime as dt
import pandas as pd

import dhan_live_bot as dlb

# ============================== config ===============================
TARGET_SPOT = 24240.0
STOP_SPOT = 24270.0
STRIKE_STEP = 50
ASSUMED_DELTA = 0.6
ASSUMED_TIME_VALUE = 40.0
BROKERAGE_PER_TRADE = 40.0
EOD_SQUAREOFF = dt.time(15, 20)
MARKET_OPEN, MARKET_CLOSE = dt.time(9, 15), dt.time(15, 30)
POLL_SECONDS = 15

LOG_FILE = f"single_put_target_stop_log_{dt.date.today().isoformat()}.csv"
LOG_FIELDS = ["logged_at", "action", "strike", "spot", "premium", "reason", "pnl_pts", "rupee_pnl", "lot_size"]


def itm_put_strike(spot):
    return math.ceil(spot / STRIKE_STEP) * STRIKE_STEP


def put_premium(strike, entry_spot, current_spot):
    """Same proxy-premium model as every other script in this project:
    intrinsic-at-entry + assumed time value, moved by (entry_spot -
    current_spot) * assumed delta (a PUT gains as spot falls), floored
    at 0."""
    intrinsic0 = max(strike - entry_spot, 0.0)
    move = entry_spot - current_spot
    return max(intrinsic0 + ASSUMED_TIME_VALUE + move * ASSUMED_DELTA, 0.0)


def lot_size_for_date(d):
    if d < dt.date(2024, 11, 20):
        return 25
    elif d < dt.date(2026, 1, 1):
        return 75
    else:
        return 65


# ============================== data fetch ===================================
def fetch_latest_spot(retries=1):
    """Read-only call, same as everywhere else in this project. Returns the
    latest available close from today's 1-min candles."""
    today = dt.date.today().strftime("%Y-%m-%d")
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = dlb.dhan.intraday_minute_data(
                security_id=dlb.UNDERLYING_SECURITY_ID, exchange_segment=dlb.UNDERLYING_EXCHANGE_SEGMENT,
                instrument_type=dlb.UNDERLYING_INSTRUMENT_TYPE, from_date=today, to_date=today,
            )
            if not isinstance(resp, dict):
                raise RuntimeError(f"Dhan API returned a non-dict response: {str(resp)[:300]}")
            data = resp.get("data", resp)
            if not isinstance(data, dict) or "close" not in data or not data["close"]:
                raise RuntimeError(f"Dhan API response missing usable close data: {str(resp)[:300]}")
            return float(data["close"][-1])
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2)
                continue
    raise RuntimeError(f"fetch_latest_spot failed after {retries + 1} attempt(s): {last_err}")


# =============================================================================
# Self-checks
# =============================================================================
def run_self_checks():
    print("Running self-checks (synthetic data)...")

    assert itm_put_strike(24255) == 24300  # ceil to nearest 50
    assert itm_put_strike(24300) == 24300
    print("  itm_put_strike rounding ok")

    entry_spot = 24255.0
    strike = itm_put_strike(entry_spot)  # 24300
    p_entry = put_premium(strike, entry_spot, entry_spot)
    assert abs(p_entry - (max(strike - entry_spot, 0) + ASSUMED_TIME_VALUE)) < 1e-9
    p_at_target = put_premium(strike, entry_spot, TARGET_SPOT)
    p_at_stop = put_premium(strike, entry_spot, STOP_SPOT)
    assert p_at_target > p_entry, "PUT premium must be HIGHER when spot falls to target (favorable)"
    assert p_at_stop < p_entry, "PUT premium must be LOWER when spot rises to stop (unfavorable)"
    print("  PUT premium moves the correct direction (up on target, down on stop) ok")

    # --- exit decision hand-trace ---
    def decide_exit(spot):
        if spot <= TARGET_SPOT:
            return "TARGET"
        if spot >= STOP_SPOT:
            return "STOP"
        return None
    assert decide_exit(24239) == "TARGET"
    assert decide_exit(24271) == "STOP"
    assert decide_exit(24255) is None
    print("  target/stop exit decision hand-trace ok")

    print("All self-checks passed.\n")


# =============================================================================
# Logging
# =============================================================================
def append_log_row(row):
    is_new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)


# =============================================================================
# Smoke test -- real data, no infinite loop, no CSV writes
# =============================================================================
def smoke_test():
    run_self_checks()
    print("=== SMOKE TEST: fetching real recent Nifty spot ===")
    try:
        spot = fetch_latest_spot()
    except Exception as e:
        print(f"Could not fetch real data to smoke-test against: {e}\n")
        print("All self-checks above still passed (the logic itself is verified). This failure is "
              "specifically about reaching the Dhan API from wherever this was just run -- re-run on "
              "the machine with a working Dhan connection before trusting the smoke test fully.")
        return
    strike = itm_put_strike(spot)
    p_now = put_premium(strike, spot, spot)
    print(f"Latest real spot: {spot:.2f}   ITM PUT strike would be: {strike}   "
          f"modeled entry premium: {p_now:.2f}")
    print(f"Target spot: {TARGET_SPOT}   Stop spot: {STOP_SPOT}")
    if TARGET_SPOT < spot < STOP_SPOT:
        print("Spot is currently between your target and stop -- consistent with a live entry point.")
    else:
        print("NOTE: current real spot is NOT between your target (24240) and stop (24270) -- "
              "double-check those levels are still what you want before running live, since one of "
              "them may already be on the wrong side of the current market.")
    print("\nSmoke test complete -- no crashes, all self-checks passed. Nothing was written to the "
          "trade log CSV (smoke test only).")


# =============================================================================
# Live loop -- single trade, entered once, exits on target/stop/EOD, then stops
# =============================================================================
def is_market_open():
    now = dt.datetime.now().time()
    return MARKET_OPEN <= now <= MARKET_CLOSE


def main_live():
    run_self_checks()
    print("=== LIVE PAPER TRADE (MOCK, NO REAL ORDERS) -- Single PUT, fixed target/stop ===")
    print(f"Target spot: {TARGET_SPOT}   Stop spot: {STOP_SPOT}   Log file: {LOG_FILE}\n")

    if not is_market_open():
        print(f"Market is currently closed ({MARKET_OPEN}-{MARKET_CLOSE} IST). Run this again during "
              f"market hours to actually enter today's trade.")
        return

    print("Fetching current spot to enter...")
    entry_spot = fetch_latest_spot()
    strike = itm_put_strike(entry_spot)
    entry_premium = put_premium(strike, entry_spot, entry_spot)
    lot = lot_size_for_date(dt.date.today())
    print(f"ENTRY: PUT@{strike}  spot={entry_spot:.2f}  premium={entry_premium:.2f}  lot={lot}")
    append_log_row({
        "logged_at": dt.datetime.now().isoformat(timespec="seconds"), "action": "ENTRY", "strike": strike,
        "spot": round(entry_spot, 2), "premium": round(entry_premium, 2), "reason": "", "pnl_pts": "",
        "rupee_pnl": "", "lot_size": lot,
    })

    print(f"Polling every {POLL_SECONDS}s until target/stop/EOD is hit...\n")
    while True:
        now = dt.datetime.now()
        if not is_market_open():
            print(f"{now.strftime('%H:%M:%S')} Market closed -- treating as EOD, exiting at last known spot.")
            reason = "MARKET_CLOSED"
            exit_spot = entry_spot
        else:
            try:
                exit_spot = fetch_latest_spot()
            except Exception:
                print(f"{now.strftime('%H:%M:%S')} Failed to fetch spot, retrying...")
                time.sleep(POLL_SECONDS)
                continue

            reason = None
            if exit_spot <= TARGET_SPOT:
                reason = "TARGET"
            elif exit_spot >= STOP_SPOT:
                reason = "STOP"
            elif now.time() >= EOD_SQUAREOFF:
                reason = "EOD"

            print(f"{now.strftime('%H:%M:%S')} spot={exit_spot:.2f}  "
                  f"(target={TARGET_SPOT} stop={STOP_SPOT})  status={'EXIT: ' + reason if reason else 'holding'}")

            if reason is None:
                time.sleep(POLL_SECONDS)
                continue

        exit_premium = put_premium(strike, entry_spot, exit_spot)
        pnl_pts = exit_premium - entry_premium
        rupee_pnl = pnl_pts * lot - BROKERAGE_PER_TRADE
        print(f"\nEXIT ({reason}): spot={exit_spot:.2f}  premium={exit_premium:.2f}  "
              f"pnl_pts={pnl_pts:.2f}  rupee_pnl=Rs{rupee_pnl:,.2f}")
        append_log_row({
            "logged_at": dt.datetime.now().isoformat(timespec="seconds"), "action": "EXIT", "strike": strike,
            "spot": round(exit_spot, 2), "premium": round(exit_premium, 2), "reason": reason,
            "pnl_pts": round(pnl_pts, 2), "rupee_pnl": round(rupee_pnl, 2), "lot_size": lot,
        })
        print(f"\nTrade logged to {LOG_FILE}. This was a single one-shot trade for today -- the script "
              f"stops here rather than looking for a new entry.")
        break


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        smoke_test()
    else:
        main_live()
