"""
Quick diagnostic -- NOT a strategy, just a check.

This new "Order Flow / Auction Market Theory" doc needs a volume profile
(where in the day's traded volume price is sitting) and low-volume-node
detection. Both require real per-candle traded VOLUME data.

The problem: Nifty 50 is an INDEX, not a traded instrument -- it may come
back from Dhan with volume always 0 (indices aren't bought/sold directly,
only their derivatives are). If that's the case here, the volume-profile
and LVN logic in this doc cannot be built on the index feed at all, and
we'd need to decide on a workaround (e.g. use the current-week ATM option's
volume instead, or a price-time-based proxy for "value area" instead of a
true volume profile).

This script fetches a few recent days of Nifty 1-min data and just reports
what Dhan actually gives us for volume, so we know before building anything.
Does not modify any other file.
"""

import datetime as dt
import pandas as pd

from dhan_live_bot import (
    dhan, UNDERLYING_SECURITY_ID, UNDERLYING_EXCHANGE_SEGMENT, UNDERLYING_INSTRUMENT_TYPE,
)

FROM_DATE = (dt.date.today() - dt.timedelta(days=5)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")


def main():
    print(f"Fetching Nifty 1-min data from {FROM_DATE} to {TO_DATE} to inspect the raw response...")
    resp = dhan.intraday_minute_data(
        security_id=UNDERLYING_SECURITY_ID,
        exchange_segment=UNDERLYING_EXCHANGE_SEGMENT,
        instrument_type=UNDERLYING_INSTRUMENT_TYPE,
        from_date=FROM_DATE,
        to_date=TO_DATE,
    )

    if isinstance(resp, dict) and resp.get("status") == "failure":
        print("\nDhan returned an error:")
        print(resp)
        return

    data = resp.get("data", resp)

    print("\n--- Keys present in the response ---")
    print(list(data.keys()))

    if "volume" not in data:
        print("\n>>> RESULT: No 'volume' field at all in the response.")
        print(">>> This means the Order-Flow/AMT doc's volume-profile and LVN logic")
        print(">>> CANNOT be built on this Nifty index feed. We'd need a different")
        print(">>> data source (e.g. option contract volume) or a non-volume proxy.")
        return

    vol = pd.Series(data["volume"])
    n_total = len(vol)
    n_zero = int((vol == 0).sum())
    n_nonzero = n_total - n_zero

    print(f"\n--- Volume field stats (over {n_total} candles) ---")
    print(f"Zero-volume candles:     {n_zero}  ({100*n_zero/n_total:.1f}%)")
    print(f"Non-zero volume candles: {n_nonzero}  ({100*n_nonzero/n_total:.1f}%)")
    if n_nonzero > 0:
        nz = vol[vol > 0]
        print(f"Non-zero volume -- min: {nz.min()}  max: {nz.max()}  mean: {nz.mean():.1f}")

    print("\n--- What this means ---")
    if n_nonzero == 0:
        print(">>> RESULT: Volume field exists but is always 0 for the Nifty index.")
        print(">>> Confirms indices don't carry real traded volume -- the volume-profile")
        print(">>> and LVN parts of this doc cannot be backtested on this feed as-is.")
    elif n_nonzero < n_total * 0.5:
        print(">>> RESULT: Volume is mostly zero/sparse -- likely NOT reliable enough")
        print(">>> to build a meaningful volume profile or LVN detection from.")
    else:
        print(">>> RESULT: Real volume data looks present and populated. The volume-")
        print(">>> profile + LVN baseline from this doc COULD be built on this feed.")


if __name__ == "__main__":
    main()
