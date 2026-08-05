"""
Diagnostic only -- figures out why the NIFTY-symbol regex match came back
empty for FUTIDX/OPTIDX rows, by printing the actual raw values Dhan's
instrument master uses for these rows (no filtering by symbol name yet).
This also matters for the LIVE bot -- resolve_itm_option() in
dhan_live_bot.py uses the exact same SEM_TRADING_SYMBOL matching logic,
so if that's broken, the live bot would fail to find any option contract
to actually trade.
Does not modify any other file.
"""

import pandas as pd
from dhan_live_bot import load_instrument_master

pd.set_option("display.max_colwidth", None)


def main():
    df = load_instrument_master()
    print("Columns:", list(df.columns))
    print()

    for itype in ["FUTIDX", "OPTIDX"]:
        sub = df[df["SEM_EXCH_INSTRUMENT_TYPE"] == itype]
        print(f"--- {itype}: {len(sub)} total rows ---")
        if sub.empty:
            print("(none at all)")
            continue

        # show raw trading symbol values, a few different underlyings, to see the real format
        sample = sub["SEM_TRADING_SYMBOL"].astype(str).head(10).tolist()
        print("First 10 SEM_TRADING_SYMBOL values:")
        for s in sample:
            print(f"  repr={s!r}")

        # now specifically look for anything containing NIFTY (case-insensitive, anywhere in string)
        contains_nifty = sub[sub["SEM_TRADING_SYMBOL"].astype(str).str.upper().str.contains("NIFTY")]
        print(f"\nRows where SEM_TRADING_SYMBOL contains 'NIFTY' anywhere (case-insensitive): {len(contains_nifty)}")
        if not contains_nifty.empty:
            print("Sample of those:")
            for s in contains_nifty["SEM_TRADING_SYMBOL"].astype(str).head(10).tolist():
                print(f"  repr={s!r}")
        print()

    # Also check if there's a separate underlying/symbol-name column that might be more reliable
    for col in ["SM_SYMBOL_NAME", "SEM_CUSTOM_SYMBOL", "SEM_INSTRUMENT_NAME", "SEM_SYMBOL_NAME"]:
        if col in df.columns:
            print(f"Column '{col}' exists -- sample values for FUTIDX/OPTIDX rows:")
            sub = df[df["SEM_EXCH_INSTRUMENT_TYPE"].isin(["FUTIDX", "OPTIDX"])]
            print(sub[col].astype(str).value_counts().head(10))
            print()


if __name__ == "__main__":
    main()
