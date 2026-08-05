"""
Diagnostic only -- checks whether we can actually get REAL historical option
premiums (not the synthetic delta-scaled point estimates every past money
simulation in this project has used) via Dhan's "rollingoption" endpoint
(https://dhanhq.co/docs/v2/expired-options-data/).

Step 1: see if the already-installed dhanhq Python library has a ready-made
method for this endpoint (safer/preferred -- reuses the same auth handling
as everything else).
Step 2: if not, call the raw REST endpoint directly using the SAME
access token / client ID already configured in dhan_config.py (imported,
never printed -- consistent with how every other script in this project
handles credentials).

This does NOT modify dhan_config.py, dhan_live_bot.py, or any other file.
"""

import datetime as dt
import requests

from dhan_live_bot import dhan
from dhan_config import ACCESS_TOKEN, CLIENT_ID

TO_DATE = dt.date.today().strftime("%Y-%m-%d")
FROM_DATE = (dt.date.today() - dt.timedelta(days=20)).strftime("%Y-%m-%d")  # endpoint caps at 30 days/call

NIFTY_UNDERLYING_SECURITY_ID = "13"  # Dhan's well-known NIFTY 50 index security id


def try_library_method():
    print("--- Step 1: looking for a ready-made method on the dhanhq library ---")
    candidates = [m for m in dir(dhan) if not m.startswith("_") and
                  any(k in m.lower() for k in ("roll", "expired", "expiry"))]
    print(f"Candidate method names found: {candidates}")
    return candidates


def try_raw_endpoint():
    print("\n--- Step 2: calling the raw REST endpoint directly ---")
    url = "https://api.dhan.co/v2/charts/rollingoption"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "access-token": ACCESS_TOKEN,
        "client-id": CLIENT_ID,
    }
    payload = {
        "exchangeSegment": "NSE_FNO",
        "interval": "1",
        "securityId": NIFTY_UNDERLYING_SECURITY_ID,
        "instrument": "OPTIDX",
        "expiryFlag": "WEEK",
        "expiryCode": 0,
        "strike": "ATM",
        "drvOptionType": "CALL",
        "requiredData": ["open", "high", "low", "close", "volume", "oi", "iv", "spot"],
        "fromDate": FROM_DATE,
        "toDate": TO_DATE,
    }
    print(f"POST {url}")
    print(f"Payload (token/client-id omitted from this printout): "
          f"{ {k: v for k, v in payload.items()} }")
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
    except Exception as e:
        print(f"Request failed: {e}")
        return

    print(f"HTTP status: {resp.status_code}")
    try:
        data = resp.json()
    except Exception:
        print("Response was not JSON:")
        print(resp.text[:2000])
        return

    print("Top-level keys:", list(data.keys()) if isinstance(data, dict) else type(data))
    ce = data.get("data", {}).get("ce") if isinstance(data, dict) else None
    if ce:
        print("\n'ce' sub-keys:", list(ce.keys()))
        for key in ("close", "oi", "iv", "spot", "strike"):
            vals = ce.get(key)
            if vals:
                print(f"  {key}: {len(vals)} values, first 5 = {vals[:5]}")
            else:
                print(f"  {key}: empty/missing")
    else:
        print("Full response (first 2000 chars):")
        print(str(data)[:2000])


def main():
    try_library_method()
    try_raw_endpoint()


if __name__ == "__main__":
    main()
