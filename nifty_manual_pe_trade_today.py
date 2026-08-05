"""
MANUAL PE TRADE -- live PAPER (mock, NO real money / NO real orders)
=================================================================
A one-off manual trade for today, not a rule-based strategy: buy the
current ITM PUT right when this script starts, then just watch the Nifty
INDEX level against your three numbers:

    TARGET1 = 24251   -- book half the position here
    TARGET2 = 24195   -- book the rest here (or all of it, if price gaps
                         straight past 24251 to 24195 between polls)
    STOP    = 24260   -- exit everything if hit first

Premium P&L uses the same proxy-premium model as every other script in
this project (intrinsic-at-entry-spot + assumed time value, moves scaled
by a flat assumed delta) so the numbers are comparable -- reused directly
from nifty_paper_trade_top3_live.py, not reimplemented. Fills are logged
at whatever spot price was actually observed on the poll that crossed a
level, not forced to the exact round number (honest about slippage).

NEVER PLACES A REAL ORDER -- read-only Dhan API calls only. Every fill is
appended to a CSV log the moment it happens.

Run (today, during market hours):
    cd "C:\\dhan algo"
    python nifty_manual_pe_trade_today.py

Ctrl+C any time -- nothing is lost, every fill so far is already on disk.
"""

import csv
import os
import time
import datetime as dt
import pandas as pd

import dhan_live_bot as dlb
import nifty_paper_trade_top3_live as top3

# ============================== YOUR LEVELS ===============================
TARGET1 = 24251.0
TARGET2 = 24195.0
STOP = 24260.0

BROKERAGE_PER_TRADE = 40.0
POLL_SECONDS = 10
MARKET_OPEN, MARKET_CLOSE = dt.time(9, 15), dt.time(15, 30)

LOG_FILE = f"manual_pe_trade_log_{dt.date.today().isoformat()}.csv"
LOG_FIELDS = ["logged_at", "action", "qty_pct", "nifty_spot", "premium",
              "pnl_pts", "rupee_pnl", "day_running_pnl_rs", "note"]


def fetch_latest_spot():
    resp = dlb.dhan.intraday_minute_data(
        security_id=dlb.UNDERLYING_SECURITY_ID, exchange_segment=dlb.UNDERLYING_EXCHANGE_SEGMENT,
        instrument_type=dlb.UNDERLYING_INSTRUMENT_TYPE,
        from_date=dt.date.today().strftime("%Y-%m-%d"), to_date=dt.date.today().strftime("%Y-%m-%d"),
    )
    data = resp.get("data", resp)
    if not isinstance(data, dict) or "close" not in data or not data["close"]:
        raise RuntimeError(f"Unexpected response: {str(resp)[:300]}")
    return float(data["close"][-1])


def premium_at(spot, entry_spot, strike):
    """Same proxy-premium model as everywhere else in this project (PE
    side): intrinsic-at-entry + assumed time value, moves scaled by a
    flat assumed delta, floored at 0."""
    intrinsic0 = max(strike - entry_spot, 0.0)
    move = entry_spot - spot  # positive as Nifty falls -- good for a PUT
    return max(intrinsic0 + top3.ASSUMED_TIME_VALUE + move * top3.ASSUMED_DELTA, 0.0)


def append_log_row(row):
    is_new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)


def log_row(action, qty_pct, spot, premium, pnl_pts, rupee_pnl, day_pnl, note=""):
    append_log_row({
        "logged_at": dt.datetime.now().isoformat(timespec="seconds"), "action": action,
        "qty_pct": qty_pct, "nifty_spot": spot, "premium": round(premium, 2),
        "pnl_pts": round(pnl_pts, 2), "rupee_pnl": round(rupee_pnl, 2),
        "day_running_pnl_rs": round(day_pnl, 2), "note": note,
    })


def run_self_checks():
    print("Running self-checks (synthetic data)...")
    assert TARGET1 > TARGET2, "TARGET1 should be closer to entry (higher) than TARGET2 for a PE trade"
    assert STOP > TARGET1, "STOP should be above both targets for a PE trade (invalidation is upward)"
    p_entry = premium_at(24258, 24258, top3.itm_put_strike(24258))
    p_at_target1 = premium_at(TARGET1, 24258, top3.itm_put_strike(24258))
    assert p_at_target1 > p_entry, "premium must be higher (more profit) once Nifty has fallen to TARGET1"
    print("  level ordering + PE premium model (falls in Nifty = higher premium) ok")
    print("All self-checks passed.\n")


def is_market_open():
    now = dt.datetime.now().time()
    return MARKET_OPEN <= now <= MARKET_CLOSE


def main():
    run_self_checks()
    print("=== MANUAL PE PAPER TRADE (MOCK, NO REAL ORDERS) ===")
    print(f"TARGET1={TARGET1}  TARGET2={TARGET2}  STOP={STOP}")
    print(f"Log file: {LOG_FILE}\n")

    if not is_market_open():
        print(f"Market is closed right now ({MARKET_OPEN}-{MARKET_CLOSE} IST) -- waiting for open...")
        while not is_market_open():
            time.sleep(30)

    # -------- entry: buy the current ITM PUT right now --------
    entry_spot = fetch_latest_spot()
    strike = top3.itm_put_strike(entry_spot)
    lot = top3.lot_size_for_date(dt.date.today())
    entry_premium = premium_at(entry_spot, entry_spot, strike)
    day_pnl = 0.0
    print(f"ENTRY: PE {strike} @ Nifty spot {entry_spot:.2f}, premium={entry_premium:.2f}, lot={lot}")
    log_row("ENTRY", 100, entry_spot, entry_premium, 0, 0, day_pnl, note=f"strike={strike} lot={lot}")

    qty_open_pct = 100  # 100 -> 50 after target1 -> 0 after target2/stop
    half_booked = False

    while qty_open_pct > 0:
        if not is_market_open():
            print("Market closed -- squaring off whatever remains at the last known price.")
            spot = fetch_latest_spot()
            premium = premium_at(spot, entry_spot, strike)
            pnl_pts = premium - entry_premium
            rupee_pnl = pnl_pts * lot * (qty_open_pct / 100) - BROKERAGE_PER_TRADE
            day_pnl += rupee_pnl
            log_row("EXIT_EOD", qty_open_pct, spot, premium, pnl_pts, rupee_pnl, day_pnl, note="market closed")
            print(f"  EXIT_EOD remaining {qty_open_pct}% @ spot {spot:.2f} premium={premium:.2f} "
                  f"pnl_Rs={rupee_pnl:,.2f}  day_pnl_Rs={day_pnl:,.2f}")
            break

        try:
            spot = fetch_latest_spot()
        except Exception as e:
            print(f"  fetch failed ({e}), retrying...")
            time.sleep(POLL_SECONDS)
            continue

        premium = premium_at(spot, entry_spot, strike)
        print(f"{dt.datetime.now().strftime('%H:%M:%S')} spot={spot:.2f}  premium={premium:.2f}  "
              f"open={qty_open_pct}%  day_pnl_Rs={day_pnl:,.2f}")

        # STOP checked first, always -- conservative, same priority rule as every other script here
        if spot >= STOP:
            pnl_pts = premium - entry_premium
            rupee_pnl = pnl_pts * lot * (qty_open_pct / 100) - BROKERAGE_PER_TRADE
            day_pnl += rupee_pnl
            log_row("STOP", qty_open_pct, spot, premium, pnl_pts, rupee_pnl, day_pnl)
            print(f"  STOP hit -- exiting remaining {qty_open_pct}% @ spot {spot:.2f} "
                  f"pnl_Rs={rupee_pnl:,.2f}  day_pnl_Rs={day_pnl:,.2f}")
            qty_open_pct = 0
            break

        if not half_booked and spot <= TARGET1:
            pnl_pts = premium - entry_premium
            rupee_pnl = pnl_pts * lot * 0.5 - (BROKERAGE_PER_TRADE / 2)
            day_pnl += rupee_pnl
            log_row("TARGET1", 50, spot, premium, pnl_pts, rupee_pnl, day_pnl)
            print(f"  TARGET1 hit -- booking 50% @ spot {spot:.2f} pnl_Rs={rupee_pnl:,.2f}  "
                  f"day_pnl_Rs={day_pnl:,.2f}")
            half_booked = True
            qty_open_pct = 50

        if spot <= TARGET2:
            pnl_pts = premium - entry_premium
            rupee_pnl = pnl_pts * lot * (qty_open_pct / 100) - (BROKERAGE_PER_TRADE / 2 if half_booked else BROKERAGE_PER_TRADE)
            day_pnl += rupee_pnl
            note = "gapped past TARGET1 straight to TARGET2" if not half_booked else ""
            log_row("TARGET2", qty_open_pct, spot, premium, pnl_pts, rupee_pnl, day_pnl, note=note)
            print(f"  TARGET2 hit -- booking remaining {qty_open_pct}% @ spot {spot:.2f} "
                  f"pnl_Rs={rupee_pnl:,.2f}  day_pnl_Rs={day_pnl:,.2f}")
            qty_open_pct = 0
            break

        time.sleep(POLL_SECONDS)

    print(f"\nDone. Total day P&L: Rs{day_pnl:,.2f}. Full trade log in {LOG_FILE}")


if __name__ == "__main__":
    main()
