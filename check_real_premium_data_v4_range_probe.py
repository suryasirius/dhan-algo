"""
Diagnostic only -- follow-up to check_real_premium_data_v3.py. Before
building a full 2-year backtest of Velez 8pt / Opening-Drive Momentum
against REAL Nifty ITM option premium data, this answers two open
questions that the v1-v3 diagnostics left unresolved:

  1. What expiry_code values actually return real data (v3 was still
     guessing among 0/1/2)?
  2. How far BACK in time can expired_options_data() actually go? The
     "rollingoption" endpoint name and the "caps at 30 days/call"
     comment in earlier diagnostics suggest this may only cover
     recently-expired weekly contracts, not arbitrary history from
     2 years ago. If that's true, a full 2-year real-premium backtest
     for both strategies isn't possible as-is, and we need to know
     that BEFORE building it.

Does not modify dhan_config.py, dhan_live_bot.py, or any other file.
Safe to run any time -- read-only historical data calls, no orders.
"""

import datetime as dt
import inspect

from dhan_live_bot import dhan

NIFTY_UNDERLYING_SECURITY_ID = 13


def show_ce_summary(resp, label):
    if resp is None:
        print(f"  [{label}] No response.")
        return False
    data = resp.get("data", resp) if isinstance(resp, dict) else resp
    if isinstance(data, dict) and "data" in data and not any(k in data for k in ("ce", "pe")):
        data = data["data"]
    ce = data.get("ce") if isinstance(data, dict) else None
    if not ce:
        print(f"  [{label}] No 'ce' key in response. Raw (first 500 chars): {str(resp)[:500]}")
        return False
    closes = ce.get("close") or []
    timestamps = ce.get("timestamp") or []
    strike = ce.get("strike")
    print(f"  [{label}] SUCCESS -- {len(closes)} close values, strike={strike}")
    if timestamps:
        first_dt = dt.datetime.fromtimestamp(timestamps[0])
        last_dt = dt.datetime.fromtimestamp(timestamps[-1])
        print(f"    Data spans: {first_dt} to {last_dt}")
    if closes:
        print(f"    First 3 closes: {closes[:3]}   Last 3 closes: {closes[-3:]}")
    return True


def try_call(expiry_code, from_date, to_date, label):
    try:
        resp = dhan.expired_options_data(
            security_id=NIFTY_UNDERLYING_SECURITY_ID,
            exchange_segment="IDX_I",
            instrument_type="OPTIDX",
            expiry_flag="WEEK",
            expiry_code=expiry_code,
            strike="ATM",
            drv_option_type="CALL",
            required_data=["open", "high", "low", "close", "volume", "oi", "iv", "spot"],
            from_date=from_date,
            to_date=to_date,
            interval=1,
        )
        return show_ce_summary(resp, label)
    except Exception as e:
        print(f"  [{label}] FAILED: {e}")
        return False


def main():
    print("Signature of expired_options_data:")
    print(" ", inspect.signature(dhan.expired_options_data))
    print("Signature of expiry_list:")
    print(" ", inspect.signature(dhan.expiry_list))

    print("\n--- Step 1: expiry_list() for NIFTY (what expiries does Dhan know about?) ---")
    expiry_resp = None
    for kwargs in (
        {"under_security_id": NIFTY_UNDERLYING_SECURITY_ID, "under_exchange_segment": "IDX_I"},
        {"security_id": NIFTY_UNDERLYING_SECURITY_ID, "exchange_segment": "IDX_I"},
    ):
        try:
            expiry_resp = dhan.expiry_list(**kwargs)
            print(f"  expiry_list({kwargs}) -> {expiry_resp}")
            break
        except Exception as e:
            print(f"  expiry_list({kwargs}) failed: {e}")

    print("\n--- Step 2: expiry_code sweep, RECENT window (last 20 days) ---")
    to_date_recent = dt.date.today().strftime("%Y-%m-%d")
    from_date_recent = (dt.date.today() - dt.timedelta(days=20)).strftime("%Y-%m-%d")
    any_recent_success = False
    for code in [0, 1, 2, 3]:
        ok = try_call(code, from_date_recent, to_date_recent, f"expiry_code={code}, recent 20d")
        any_recent_success = any_recent_success or ok

    print("\n--- Step 3: HOW FAR BACK does history actually go? ---")
    print("(Only runs if step 2 found at least one working expiry_code)")
    if any_recent_success:
        working_code = 0
        windows = [
            ("3 months back", 90, 110),
            ("6 months back", 180, 200),
            ("1 year back", 365, 385),
            ("2 years back", 730, 750),
        ]
        for label, start_days_ago, end_days_ago in windows:
            f = (dt.date.today() - dt.timedelta(days=end_days_ago)).strftime("%Y-%m-%d")
            t = (dt.date.today() - dt.timedelta(days=start_days_ago)).strftime("%Y-%m-%d")
            try_call(working_code, f, t, label)
    else:
        print("  Skipped -- no expiry_code worked in step 2, so there's no point testing older windows yet.")

    print("\n=== SUMMARY ===")
    print("If step 2 found a working expiry_code but step 3 shows older windows failing or")
    print("returning empty, real option data is only available for recent weeks/months, and")
    print("the full 2-year Velez 8pt / Opening-Drive Momentum backtests will need to run on")
    print("whatever shorter window actually returns real data.")
    print("If step 3's older windows ALSO succeed, a full 2-year real-option backtest is viable.")


if __name__ == "__main__":
    main()
