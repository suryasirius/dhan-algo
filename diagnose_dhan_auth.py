"""
Module: diagnose_dhan_auth.py
Description: Isolates WHY Dhan is rejecting intraday history requests with
             DH-901 even though the access token itself isn't expired.
             Compares two different code paths that both hit Dhan, using
             the SAME token, to see if the failure is token-wide or specific
             to one API path:
               A. step1_config_and_connector.py's DhanConnector (raw REST
                  call to /v2/charts/intraday) -- what step9/step10 use.
               B. dhan_live_bot.py's dhanhq SDK client
                  (dhan.intraday_minute_data) -- what om/sc/dhan_live_bot
                  and everything built on top of them use.
             Both should work if it's a token problem; if only B fails,
             it's specific to that SDK call / that API's permission scope,
             not the token being invalid.

Run:
    cd "C:\\dhan algo"
    python diagnose_dhan_auth.py
"""

import sys
import os
import datetime as dt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RESEARCH_DIR = os.path.join(_THIS_DIR, "Research")
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)

today = dt.date.today().strftime("%Y-%m-%d")

print("=" * 78)
print("TEST A: step1_config_and_connector.py's DhanConnector (raw REST call)")
print("=" * 78)
try:
    from step1_config_and_connector import PlatformConfig, DhanConnector
    cfg = PlatformConfig()
    print(f"  Using client_id={cfg.dhan_client_id}, token starts with {cfg.dhan_access_token[:15]}...")
    conn = DhanConnector(cfg)
    df = conn.fetch_historical_intraday(cfg.nifty_security_id, cfg.nifty_exchange_segment)
    if df is not None and not df.empty:
        print(f"  RESULT: SUCCESS -- got {len(df)} candles for today.")
    else:
        print("  RESULT: EMPTY (check the log line just above this for the exact API error).")
except Exception as e:
    print(f"  RESULT: EXCEPTION -- {e}")

print()
print("=" * 78)
print("TEST B: dhan_live_bot.py's dhanhq SDK client (intraday_minute_data)")
print("=" * 78)
try:
    import dhan_live_bot as dlb
    print(f"  Using client_id={dlb.CLIENT_ID}, token starts with {dlb.ACCESS_TOKEN[:15]}...")
    resp = dlb.dhan.intraday_minute_data(
        security_id=dlb.UNDERLYING_SECURITY_ID,
        exchange_segment=dlb.UNDERLYING_EXCHANGE_SEGMENT,
        instrument_type=dlb.UNDERLYING_INSTRUMENT_TYPE,
        from_date=today, to_date=today,
    )
    if isinstance(resp, dict) and resp.get("status") == "failure":
        print(f"  RESULT: REJECTED -- {resp.get('remarks', resp)}")
    else:
        data = resp.get("data", resp) if isinstance(resp, dict) else resp
        if isinstance(data, dict) and "open" in data:
            print(f"  RESULT: SUCCESS -- got {len(data.get('open', []))} candles for today.")
        else:
            print(f"  RESULT: UNEXPECTED -- {str(resp)[:300]}")
except Exception as e:
    print(f"  RESULT: EXCEPTION -- {e}")

print()
print("=" * 78)
print("If A succeeded and B failed: the token itself is fine, the issue is specific")
print("to the dhanhq SDK's intraday_minute_data call (om/sc/dhan_live_bot's path) --")
print("likely a Dhan account permission/API-access-type issue, not the token value.")
print("If BOTH failed: something is wrong with the token/client ID itself despite it")
print("not being expired yet (e.g. client ID mismatch, account-level API access not")
print("enabled). If BOTH succeeded: the earlier failure may have been transient --")
print("try sirius_5yr_data_probe.py again.")
print("=" * 78)
