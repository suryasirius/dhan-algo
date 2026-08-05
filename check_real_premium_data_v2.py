"""
Diagnostic only -- follow-up to check_real_premium_data.py. That script found
the dhanhq library already has expiry_list() and expired_options_data()
methods for exactly this. Using those directly instead of the raw endpoint,
since the raw call failed only because expiryCode=0 wasn't accepted (likely
needs a real code from expiry_list() first, not a literal 0).

Does not modify any other file.
"""

import inspect
import datetime as dt

from dhan_live_bot import dhan

NIFTY_UNDERLYING_SECURITY_ID = "13"
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
FROM_DATE = (dt.date.today() - dt.timedelta(days=20)).strftime("%Y-%m-%d")


def main():
    print("Signature of expiry_list:")
    print(" ", inspect.signature(dhan.expiry_list))
    print("Signature of expired_options_data:")
    print(" ", inspect.signature(dhan.expired_options_data))

    print("\n--- Calling expiry_list() for NIFTY ---")
    try:
        resp = dhan.expiry_list(
            under_security_id=int(NIFTY_UNDERLYING_SECURITY_ID),
            under_exchange_segment="IDX_I",
        )
        print(resp)
    except Exception as e:
        print(f"expiry_list call #1 failed: {e}")
        print("Trying alternate parameter names...")
        try:
            resp = dhan.expiry_list(
                security_id=int(NIFTY_UNDERLYING_SECURITY_ID),
                exchange_segment="IDX_I",
            )
            print(resp)
        except Exception as e2:
            print(f"expiry_list call #2 also failed: {e2}")
            resp = None

    expiry_dates = None
    if isinstance(resp, dict):
        expiry_dates = resp.get("data", resp)
    print("\nParsed expiry data:", expiry_dates)

    print("\n--- Calling expired_options_data() for the nearest expiry, ATM CALL ---")
    try:
        opt_resp = dhan.expired_options_data(
            under_security_id=int(NIFTY_UNDERLYING_SECURITY_ID),
            under_exchange_segment="IDX_I",
            interval="1",
            instrument="OPTIDX",
            expiry_flag="WEEK",
            expiry_code=1,
            strike="ATM",
            drv_option_type="CALL",
            required_data=["open", "high", "low", "close", "volume", "oi", "iv", "spot"],
            from_date=FROM_DATE,
            to_date=TO_DATE,
        )
        print("Top-level keys:", list(opt_resp.keys()) if isinstance(opt_resp, dict) else type(opt_resp))
        data = opt_resp.get("data", opt_resp) if isinstance(opt_resp, dict) else opt_resp
        ce = data.get("ce") if isinstance(data, dict) else None
        if ce:
            print("'ce' sub-keys:", list(ce.keys()))
            for key in ("close", "oi", "iv", "spot", "strike"):
                vals = ce.get(key)
                if vals:
                    print(f"  {key}: {len(vals)} values, first 5 = {vals[:5]}")
                else:
                    print(f"  {key}: empty/missing")
        else:
            print("Full response:", str(opt_resp)[:2000])
    except Exception as e:
        print(f"expired_options_data call failed: {e}")


if __name__ == "__main__":
    main()
