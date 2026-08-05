"""
Diagnostic only. Dhan's own API docs (https://dhanhq.co/docs/v2/historical-data/)
show that historical intraday data DOES support open interest, but only if
you explicitly pass "oi": true in the request -- it's off by default, which
is exactly why our last check came back with no oi field at all.

This tries passing oi=True through the dhanhq Python library's
intraday_minute_data() call. If the library supports it, we should see an
"open_interest" key in the response with real changing values for the NIFTY
futures/option contracts.

Does not modify any other file.
"""

import datetime as dt
import pandas as pd
import inspect

from dhan_live_bot import dhan, load_instrument_master, resolve_itm_option

TO_DATE = dt.date.today().strftime("%Y-%m-%d")
FROM_DATE = (dt.date.today() - dt.timedelta(days=5)).strftime("%Y-%m-%d")


def inspect_response(label, security_id, exchange_segment, instrument_type):
    print(f"\n--- {label} ---")
    try:
        resp = dhan.intraday_minute_data(
            security_id=str(security_id),
            exchange_segment=exchange_segment,
            instrument_type=instrument_type,
            from_date=FROM_DATE,
            to_date=TO_DATE,
            oi=True,
        )
    except TypeError as e:
        print(f"The installed dhanhq library's intraday_minute_data() does not "
              f"accept an 'oi' argument: {e}")
        return
    except Exception as e:
        print(f"Request failed: {e}")
        return

    if isinstance(resp, dict) and resp.get("status") == "failure":
        print("Dhan returned an error:")
        print(resp)
        return

    data = resp.get("data", resp)
    print("Keys present:", list(data.keys()))

    for key in ("oi", "open_interest", "openInterest"):
        if key in data:
            oi = pd.Series(data[key])
            n_total = len(oi)
            n_zero = int((oi == 0).sum())
            n_unique = oi.nunique()
            print(f"Found '{key}' field: {n_total} candles, {n_zero} zero, "
                  f"{n_unique} unique values (min={oi.min()}, max={oi.max()}).")
            if n_unique <= 1:
                print(">>> All identical/flat -- likely NOT a real changing time series.")
            else:
                print(">>> Values actually change over time -- looks like a usable OI series.")
            return
    print(">>> Still no OI/open_interest field in the response.")


def main():
    print("Checking the installed dhanhq library's intraday_minute_data signature...")
    try:
        sig = inspect.signature(dhan.intraday_minute_data)
        print(f"Signature: intraday_minute_data{sig}\n")
    except Exception as e:
        print(f"Could not inspect signature: {e}\n")

    df = load_instrument_master()

    futs = df[
        (df["UNDERLYING_SYMBOL"].astype(str).str.upper() == "NIFTY")
        & (df["EXCH_ID"].astype(str).str.upper() == "NSE")
        & (df["INSTRUMENT_TYPE"].astype(str).str.upper() == "FUT")
    ].copy()
    if not futs.empty:
        futs["SM_EXPIRY_DATE"] = pd.to_datetime(futs["SM_EXPIRY_DATE"])
        row = futs.sort_values("SM_EXPIRY_DATE").iloc[0]
        inspect_response(
            f"NIFTY FUT ({row['DISPLAY_NAME']}, expiry {row['SM_EXPIRY_DATE'].date()})",
            row["SECURITY_ID"], "NSE_FNO", "FUTIDX",
        )

    for opt_type in ["CE", "PE"]:
        try:
            contract = resolve_itm_option(opt_type, 25000)
        except Exception as e:
            print(f"\nCould not resolve a {opt_type} contract: {e}")
            continue
        inspect_response(
            f"NIFTY {opt_type} {contract['strike']} (expiry {contract['expiry']})",
            contract["security_id"], "NSE_FNO", "OPTIDX",
        )


if __name__ == "__main__":
    main()
