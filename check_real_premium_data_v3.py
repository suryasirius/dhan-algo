"""
Diagnostic only -- fixes the parameter names for expired_options_data() based
on the real signature we just saw:
  expired_options_data(security_id, exchange_segment, instrument_type,
                        expiry_flag, expiry_code, strike, drv_option_type,
                        required_data, from_date, to_date, interval=1)

Does not modify any other file.
"""

import datetime as dt
from dhan_live_bot import dhan

NIFTY_UNDERLYING_SECURITY_ID = 13
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
FROM_DATE = (dt.date.today() - dt.timedelta(days=20)).strftime("%Y-%m-%d")


def show(resp):
    print("Top-level keys:", list(resp.keys()) if isinstance(resp, dict) else type(resp))
    data = resp.get("data", resp) if isinstance(resp, dict) else resp
    # unwrap potential double-nesting like expiry_list showed ({"data": {"data": [...]}})
    if isinstance(data, dict) and "data" in data and not any(k in data for k in ("ce", "pe")):
        data = data["data"]
    ce = data.get("ce") if isinstance(data, dict) else None
    if ce:
        print("'ce' sub-keys:", list(ce.keys()))
        for key in ("close", "oi", "iv", "spot", "strike", "timestamp"):
            vals = ce.get(key)
            if vals:
                print(f"  {key}: {len(vals)} values, first 5 = {vals[:5]}")
            else:
                print(f"  {key}: empty/missing")
    else:
        print("Full response (first 2000 chars):", str(resp)[:2000])


def main():
    for expiry_code in [1, 0, 2]:
        print(f"\n=== Trying expiry_code={expiry_code} ===")
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
                from_date=FROM_DATE,
                to_date=TO_DATE,
            )
            show(resp)
        except Exception as e:
            print(f"Failed: {e}")


if __name__ == "__main__":
    main()
