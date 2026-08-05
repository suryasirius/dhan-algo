"""
Module: sirius_5yr_data_probe.py
Description: Diagnostic only -- finds the REAL earliest date Dhan actually
             returns usable intraday Nifty data for, instead of trusting the
             documented "5 years" figure blindly. sirius_5yr_backtest.py
             crashed on one of the older chunks because Dhan's own
             fetch_index_chunked() (in nifty_open_momentum_stoptrail_sweep.py)
             doesn't check the response status before parsing it -- this
             script calls the same underlying API directly, chunk by chunk,
             and prints exactly what Dhan says for each one (success, row
             count, or the real error/remarks), so we can see where real
             data actually starts.

Does not modify nifty_open_momentum_stoptrail_sweep.py or any other script
-- read-only diagnostic, same spirit as the OI-window diagnostic already
done in oi_gated_concepts_backtest.py.

Run:
    cd "C:\\dhan algo"
    python sirius_5yr_data_probe.py
"""

import datetime as dt
import dhan_live_bot as dlb

CHUNK_DAYS = 85
FROM_DATE_5YR = (dt.date.today() - dt.timedelta(days=1825)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")


def main():
    start = dt.datetime.strptime(FROM_DATE_5YR, "%Y-%m-%d").date()
    end = dt.datetime.strptime(TO_DATE, "%Y-%m-%d").date()
    print(f"Probing Dhan intraday data chunk-by-chunk from {start} to {end} "
          f"({CHUNK_DAYS}-day chunks)...\n")

    cur = start
    first_good_date = None
    while cur <= end:
        chunk_end = min(cur + dt.timedelta(days=CHUNK_DAYS - 1), end)
        try:
            resp = dlb.dhan.intraday_minute_data(
                security_id=dlb.UNDERLYING_SECURITY_ID,
                exchange_segment=dlb.UNDERLYING_EXCHANGE_SEGMENT,
                instrument_type=dlb.UNDERLYING_INSTRUMENT_TYPE,
                from_date=cur.strftime("%Y-%m-%d"),
                to_date=chunk_end.strftime("%Y-%m-%d"),
            )
        except Exception as e:
            print(f"  [{cur} to {chunk_end}] EXCEPTION: {e}")
            cur = chunk_end + dt.timedelta(days=1)
            continue

        if isinstance(resp, dict) and resp.get("status") == "failure":
            print(f"  [{cur} to {chunk_end}] REJECTED -- {resp.get('remarks', resp)}")
        elif isinstance(resp, dict) and "data" in resp and isinstance(resp["data"], str):
            print(f"  [{cur} to {chunk_end}] REJECTED (data field is a string, not candles) -- {resp['data']}")
        else:
            data = resp.get("data", resp) if isinstance(resp, dict) else resp
            if isinstance(data, dict) and "open" in data:
                n = len(data.get("open", []))
                print(f"  [{cur} to {chunk_end}] OK -- {n} candles")
                if n > 0 and first_good_date is None:
                    first_good_date = cur
            else:
                print(f"  [{cur} to {chunk_end}] UNEXPECTED SHAPE -- {str(resp)[:200]}")

        cur = chunk_end + dt.timedelta(days=1)

    print("\n" + "=" * 78)
    if first_good_date:
        print(f"First chunk with real usable data starts around: {first_good_date}")
        print(f"That's ~{(end - first_good_date).days} days of real intraday history "
              f"({(end - first_good_date).days / 365.25:.1f} years), not the full 5 years documented.")
    else:
        print("No usable data found anywhere in this 5-year window -- something else is wrong "
              "(check the token/connection, not just the date range).")
    print("=" * 78)


if __name__ == "__main__":
    main()
