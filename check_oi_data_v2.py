"""
Diagnostic only -- re-run of the OI check, now using the FIXED instrument
lookup (dhan_live_bot.py now correctly finds real NIFTY contracts via the
detailed scrip master). Checks whether Dhan's intraday_minute_data actually
returns historical open-interest for NIFTY futures/options, or nothing.

Does not modify any other file.
"""

import datetime as dt
import pandas as pd

from dhan_live_bot import dhan, load_instrument_master, resolve_itm_option

TO_DATE = dt.date.today().strftime("%Y-%m-%d")
FROM_DATE = (dt.date.today() - dt.timedelta(days=5)).strftime("%Y-%m-%d")


def inspect_response(label, security_id, exchange_segment, instrument_type):
    print(f"\n--- {label} (security_id={security_id}, segment={exchange_segment}, type={instrument_type}) ---")
    try:
        resp = dhan.intraday_minute_data(
            security_id=str(security_id),
            exchange_segment=exchange_segment,
            instrument_type=instrument_type,
            from_date=FROM_DATE,
            to_date=TO_DATE,
        )
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
    print(">>> No OI/open_interest field found in this response at all.")


def main():
    print("Loading instrument master (via the now-fixed dhan_live_bot.load_instrument_master)...")
    df = load_instrument_master()
    print(f"Loaded {len(df)} rows.\n")

    # --- Current-month NIFTY FUTURES ---
    futs = df[
        (df["UNDERLYING_SYMBOL"].astype(str).str.upper() == "NIFTY")
        & (df["EXCH_ID"].astype(str).str.upper() == "NSE")
        & (df["INSTRUMENT_TYPE"].astype(str).str.upper() == "FUT")
    ].copy()
    if futs.empty:
        print("No NIFTY FUT rows found -- unexpected given the earlier check. Skipping futures test.")
    else:
        futs["SM_EXPIRY_DATE"] = pd.to_datetime(futs["SM_EXPIRY_DATE"])
        futs = futs.sort_values("SM_EXPIRY_DATE")
        row = futs.iloc[0]
        inspect_response(
            f"NIFTY FUT ({row['DISPLAY_NAME']}, expiry {row['SM_EXPIRY_DATE'].date()})",
            row["SECURITY_ID"], "NSE_FNO", "FUTIDX",
        )

    # --- Nearest-expiry ATM-ish CE/PE options (reuse the fixed live-bot lookup) ---
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
