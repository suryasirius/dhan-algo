"""
Diagnostic only. The compact scrip master (api-scrip-master.csv, what
dhan_live_bot.py currently downloads) came back with ZERO NIFTY FUTIDX/OPTIDX
rows -- only SENSEX/BANKEX/BITOPT/MCX-index products. That's a real problem
for the LIVE bot too (resolve_itm_option() would never find a NIFTY contract
to actually trade).

Dhan also publishes a "detailed" scrip master with a proper UNDERLYING_SYMBOL
column (per their docs), which should be a much more reliable way to find
NIFTY contracts than pattern-matching the trading symbol string. This script
downloads that file directly (does not touch dhan_live_bot.py) and checks
whether NIFTY NSE index futures/options show up there.
"""

import requests
import io
import pandas as pd

DETAILED_URL = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"


def main():
    print(f"Downloading detailed scrip master from {DETAILED_URL} ...")
    resp = requests.get(DETAILED_URL, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    print(f"Loaded {len(df)} rows.")
    print("Columns:", list(df.columns))

    if "UNDERLYING_SYMBOL" not in df.columns:
        print("\nNo UNDERLYING_SYMBOL column in this file either -- unexpected, stopping here.")
        return

    nifty = df[
        (df["UNDERLYING_SYMBOL"].astype(str).str.upper() == "NIFTY")
        & (df["EXCH_ID"].astype(str).str.upper() == "NSE")
    ]
    print(f"\nRows with UNDERLYING_SYMBOL == 'NIFTY' and EXCH_ID == 'NSE': {len(nifty)}")
    if not nifty.empty:
        print("\nINSTRUMENT_TYPE breakdown:")
        print(nifty["INSTRUMENT_TYPE"].value_counts())
        print("\nSample rows (FUTIDX):")
        print(nifty[nifty["INSTRUMENT_TYPE"] == "FUTIDX"][
            ["SYMBOL_NAME", "DISPLAY_NAME", "SM_EXPIRY_DATE", "SECURITY_ID"] if "SECURITY_ID" in nifty.columns
            else ["SYMBOL_NAME", "DISPLAY_NAME", "SM_EXPIRY_DATE"]
        ].head(10).to_string())
        print("\nSample rows (OPTIDX):")
        print(nifty[nifty["INSTRUMENT_TYPE"] == "OPTIDX"][
            ["SYMBOL_NAME", "DISPLAY_NAME", "STRIKE_PRICE", "OPTION_TYPE", "SM_EXPIRY_DATE"]
        ].head(10).to_string())
    else:
        print("Still no NIFTY rows found -- something else is going on, needs more digging.")

    print("\n--- Also checking under alternate underlying spellings, just in case ---")
    alt = df[df["UNDERLYING_SYMBOL"].astype(str).str.upper().str.contains("NIFTY", na=False)]
    print(f"Rows where UNDERLYING_SYMBOL contains 'NIFTY' anywhere: {len(alt)}")
    if not alt.empty:
        print(alt["UNDERLYING_SYMBOL"].value_counts().head(10))


if __name__ == "__main__":
    main()
