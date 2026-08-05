"""
Diagnostic only -- fixes two issues found in v4's real run:
  1. expiry_code=0 errors with DH-905 "expiryCode is required" (falsy-zero
     bug -- 0 gets treated as not-provided). Codes 1/2/3 didn't error, so
     this version only uses those.
  2. v4's Step 3 (the "how far back" probe) had a bug -- it hardcoded
     expiry_code=0 regardless of what worked in Step 2, so every older
     window failed with the SAME error for the wrong reason. Fixed to use
     an actually-working code.
  3. v4 used strike="ATM" (a literal string), which returned an empty
     response (no error, but 0 data points) for codes 1/2/3 even in the
     recent window. This version also tries a real numeric strike (a
     round-50 value near current Nifty spot) to see if that's what this
     endpoint actually expects instead of a moneyness keyword.

Does not modify dhan_config.py, dhan_live_bot.py, or any other file.
Safe to run any time -- read-only historical data calls, no orders.
"""

import datetime as dt

from dhan_live_bot import dhan

NIFTY_UNDERLYING_SECURITY_ID = 13
APPROX_SPOT_GUESS = 25000  # just needs to be roughly near current Nifty spot, rounded to 50


def show_ce_summary(resp, label):
    if resp is None:
        print(f"  [{label}] No response.")
        return False
    if isinstance(resp, dict) and resp.get("status") == "failure":
        print(f"  [{label}] API ERROR: {resp.get('remarks')}")
        return False
    data = resp.get("data", resp) if isinstance(resp, dict) else resp
    if isinstance(data, dict) and "data" in data and not any(k in data for k in ("ce", "pe")):
        data = data["data"]
    ce = data.get("ce") if isinstance(data, dict) else None
    if not ce:
        print(f"  [{label}] No 'ce' key. Raw (first 400 chars): {str(resp)[:400]}")
        return False
    closes = ce.get("close") or []
    timestamps = ce.get("timestamp") or []
    strike = ce.get("strike")
    if not closes:
        print(f"  [{label}] Response OK but EMPTY (0 close values). strike={strike}")
        return False
    print(f"  [{label}] REAL DATA -- {len(closes)} close values, strike={strike}")
    if timestamps:
        first_dt = dt.datetime.fromtimestamp(timestamps[0])
        last_dt = dt.datetime.fromtimestamp(timestamps[-1])
        print(f"    Data spans: {first_dt} to {last_dt}")
    print(f"    First 3 closes: {closes[:3]}   Last 3 closes: {closes[-3:]}")
    return True


def try_call(expiry_code, strike, from_date, to_date, label):
    try:
        resp = dhan.expired_options_data(
            security_id=NIFTY_UNDERLYING_SECURITY_ID,
            exchange_segment="IDX_I",
            instrument_type="OPTIDX",
            expiry_flag="WEEK",
            expiry_code=expiry_code,
            strike=strike,
            drv_option_type="CALL",
            required_data=["open", "high", "low", "close", "volume", "oi", "iv", "spot"],
            from_date=from_date,
            to_date=to_date,
            interval=1,
        )
        return show_ce_summary(resp, label)
    except Exception as e:
        print(f"  [{label}] EXCEPTION: {e}")
        return False


def main():
    to_date_recent = dt.date.today().strftime("%Y-%m-%d")
    from_date_recent = (dt.date.today() - dt.timedelta(days=20)).strftime("%Y-%m-%d")

    print("--- Step 1: retry codes 1/2/3 with a NUMERIC strike instead of 'ATM' (recent 20d) ---")
    working_combos = []
    for code in [1, 2, 3]:
        for strike_val in [APPROX_SPOT_GUESS, "ATM"]:
            ok = try_call(code, strike_val, from_date_recent, to_date_recent,
                           f"code={code}, strike={strike_val}")
            if ok:
                working_combos.append((code, strike_val))

    if not working_combos:
        print("\nNo combo returned real data yet. Trying a wider numeric strike sweep "
              "(in case current spot guess is off) ...")
        for code in [1, 2, 3]:
            for strike_val in [23000, 23500, 24000, 24500, 25000, 25500, 26000]:
                ok = try_call(code, strike_val, from_date_recent, to_date_recent,
                              f"code={code}, strike={strike_val}")
                if ok:
                    working_combos.append((code, strike_val))

    print(f"\nWorking combos found: {working_combos}")

    if not working_combos:
        print("\n=== SUMMARY ===")
        print("Still no real data with any code/strike combo tested. This needs either a "
              "different parameter (e.g. drv_option_type spelling, expiry_flag value) or "
              "confirmation from Dhan's own docs/support on the exact expected strike format.")
        return

    print("\n--- Step 2: HOW FAR BACK does history go, using a confirmed working combo? ---")
    code, strike_val = working_combos[0]
    windows = [
        ("3 months back", 90, 110),
        ("6 months back", 180, 200),
        ("1 year back", 365, 385),
        ("2 years back", 730, 750),
    ]
    for label, start_days_ago, end_days_ago in windows:
        f = (dt.date.today() - dt.timedelta(days=end_days_ago)).strftime("%Y-%m-%d")
        t = (dt.date.today() - dt.timedelta(days=start_days_ago)).strftime("%Y-%m-%d")
        try_call(code, strike_val, f, t, label)

    print("\n=== SUMMARY ===")
    print(f"Confirmed working combo: expiry_code={code}, strike={strike_val}")
    print("If the older windows above returned real data too, a full multi-year real-option")
    print("backtest is viable. If only the recent window worked, real option data is limited")
    print("to recently-expired contracts, and the backtest window will need to be shortened")
    print("to match whatever range actually returns data.")


if __name__ == "__main__":
    main()
