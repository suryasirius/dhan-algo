"""
Diagnostic only -- built directly from Dhan's official docs
(https://dhanhq.co/docs/v2/expired-options-data/ and /annexure/), after
v1-v6 exhausted blind parameter guessing. Two concrete fixes identified
from the docs themselves:

  1. Expiry Code enum is 0=Current/Near Expiry, 1=Next Expiry,
     2=Far Expiry -- there is no "3" (we were testing an undocumented
     value). 0 is almost certainly the one with real historical data
     (the nearest/most recently expired contract) -- but literal int 0
     is exactly the value that broke with "DH-905 expiryCode is
     required" earlier, a classic falsy-zero bug (something in the SDK
     likely does `if expiry_code:` and silently drops a real 0). Testing
     a string "0" as a workaround, since the confirmed drv_option_type
     validation shows the SDK does its own client-side checks.
  2. The docs' own example payload uses exchangeSegment="NSE_FNO", not
     "IDX_I" -- options trade on the F&O segment, not the index segment.
     Every version of this diagnostic so far used "IDX_I". Testing
     "NSE_FNO" instead.

Also includes a raw REST fallback (bypassing the SDK method entirely)
in case the SDK can't be made to send a literal 0 no matter what's
passed to it -- constructing the JSON body directly guarantees a real
`0` gets serialized, not silently dropped.

Does not modify dhan_config.py, dhan_live_bot.py, or any other file.
Safe to run any time -- read-only historical data calls, no orders.
"""

import time
import datetime as dt
import requests

from dhan_live_bot import dhan
from dhan_config import ACCESS_TOKEN, CLIENT_ID

NIFTY_UNDERLYING_SECURITY_ID = 13
DELAY_SECONDS = 4


def show_summary(resp, label):
    if isinstance(resp, dict) and resp.get("status") == "failure":
        print(f"  [{label}] API ERROR: {resp.get('remarks')}")
        return False
    data = resp.get("data", resp) if isinstance(resp, dict) else resp
    if isinstance(data, dict) and "data" in data and not any(k in data for k in ("ce", "pe")):
        data = data["data"]
    found_any = False
    for side in ("ce", "pe"):
        block = data.get(side) if isinstance(data, dict) else None
        if block:
            closes = block.get("close") or []
            if closes:
                print(f"  [{label}] '{side}' REAL DATA -- {len(closes)} closes, "
                      f"strike={block.get('strike')}, first 3={closes[:3]}")
                found_any = True
            else:
                print(f"  [{label}] '{side}' key present but empty")
    if not found_any:
        print(f"  [{label}] No usable data. Raw (first 350 chars): {str(resp)[:350]}")
    return found_any


def try_sdk_call(exchange_segment, expiry_code, expiry_flag, from_date, to_date, label):
    try:
        resp = dhan.expired_options_data(
            security_id=NIFTY_UNDERLYING_SECURITY_ID,
            exchange_segment=exchange_segment,
            instrument_type="OPTIDX",
            expiry_flag=expiry_flag,
            expiry_code=expiry_code,
            strike="ATM",
            drv_option_type="CALL",
            required_data=["open", "high", "low", "close", "volume", "oi", "iv", "spot"],
            from_date=from_date,
            to_date=to_date,
            interval=1,
        )
        return show_summary(resp, label)
    except Exception as e:
        print(f"  [{label}] EXCEPTION: {e}")
        return False


def try_raw_rest_call(exchange_segment, expiry_code, expiry_flag, from_date, to_date, label):
    url = "https://api.dhan.co/v2/charts/rollingoption"
    headers = {
        "Accept": "application/json", "Content-Type": "application/json",
        "access-token": ACCESS_TOKEN, "client-id": CLIENT_ID,
    }
    payload = {
        "exchangeSegment": exchange_segment, "interval": "1",
        "securityId": NIFTY_UNDERLYING_SECURITY_ID, "instrument": "OPTIDX",
        "expiryFlag": expiry_flag, "expiryCode": expiry_code, "strike": "ATM",
        "drvOptionType": "CALL",
        "requiredData": ["open", "high", "low", "close", "volume", "oi", "iv", "spot"],
        "fromDate": from_date, "toDate": to_date,
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        data = resp.json()
        return show_summary(data, label)
    except Exception as e:
        print(f"  [{label}] EXCEPTION: {e}")
        return False


def main():
    to_date_recent = dt.date.today().strftime("%Y-%m-%d")
    from_date_recent = (dt.date.today() - dt.timedelta(days=25)).strftime("%Y-%m-%d")

    print("--- Fix attempt 1: SDK call, exchange_segment='NSE_FNO', expiry_code='0' (string) ---")
    working = []
    tests = [
        ("NSE_FNO", "0", "WEEK"),
        ("NSE_FNO", "0", "MONTH"),
        ("NSE_FNO", 0, "WEEK"),
        ("IDX_I", "0", "MONTH"),
    ]
    for i, (seg, code, flag) in enumerate(tests):
        label = f"SDK seg={seg}, code={code!r}, flag={flag}"
        ok = try_sdk_call(seg, code, flag, from_date_recent, to_date_recent, label)
        if ok:
            working.append(("sdk", seg, code, flag))
        time.sleep(DELAY_SECONDS)

    print("\n--- Fix attempt 2: raw REST call (bypasses SDK entirely), literal JSON 0 ---")
    for flag in ["WEEK", "MONTH"]:
        label = f"RAW seg=NSE_FNO, code=0, flag={flag}"
        ok = try_raw_rest_call("NSE_FNO", 0, flag, from_date_recent, to_date_recent, label)
        if ok:
            working.append(("raw", "NSE_FNO", 0, flag))
        time.sleep(DELAY_SECONDS)

    print(f"\nWorking combos: {working}")
    print("\n=== SUMMARY ===")
    if working:
        print("Found real data. Next step: build both strategies' real-option backtests")
        print("using this exact combo, and probe how far back it actually goes.")
    else:
        print("Still empty across every documented-correct combination (right enum values,")
        print("right segment, right strike format, both SDK and raw REST). At this point the")
        print("most likely remaining explanation is account-level: this Dhan plan/subscription")
        print("may not include entitlement to the Expired Options Data product itself (a")
        print("separate subscription from live/current option chain access), which the API")
        print("would signal exactly like this -- accepting the request but returning nothing,")
        print("without a clear permission error. Worth checking directly with Dhan support.")


if __name__ == "__main__":
    main()
