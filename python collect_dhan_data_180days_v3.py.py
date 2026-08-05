import datetime as dt
import time
import pandas as pd

from dhanhq import DhanContext, dhanhq
from dhan_config import (
    CLIENT_ID,
    ACCESS_TOKEN,
    UNDERLYING_SECURITY_ID,
    UNDERLYING_EXCHANGE_SEGMENT,
    UNDERLYING_INSTRUMENT_TYPE,
)

OUTPUT_FILE = "dhan_historical_data_180days.csv"

TOTAL_DAYS = 180
CHUNK_DAYS = 30

dhan = dhanhq(DhanContext(CLIENT_ID, ACCESS_TOKEN))

end_date = dt.datetime.now().date()
start_date = end_date - dt.timedelta(days=TOTAL_DAYS)

all_data = []

current_start = start_date

while current_start <= end_date:

    current_end = min(
        current_start + dt.timedelta(days=CHUNK_DAYS - 1),
        end_date
    )

    print(f"Downloading {current_start} -> {current_end}")

    try:

        response = dhan.intraday_minute_data(
            security_id=UNDERLYING_SECURITY_ID,
            exchange_segment=UNDERLYING_EXCHANGE_SEGMENT,
            instrument_type=UNDERLYING_INSTRUMENT_TYPE,
            from_date=current_start.strftime("%Y-%m-%d"),
            to_date=current_end.strftime("%Y-%m-%d")
        )

        if not isinstance(response, dict):
            print("Unexpected response:", response)
            break

        if "data" not in response:
            print(response)
            break

        data = response["data"]

        df = pd.DataFrame({
            "timestamp": data["timestamp"],
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
        })

        if "volume" in data:
            df["volume"] = data["volume"]

        df["datetime"] = (
            pd.to_datetime(df["timestamp"], unit="s")
            + pd.Timedelta(hours=5, minutes=30)
        )

        all_data.append(df)

        print(f"Downloaded {len(df)} candles")

        time.sleep(1)

    except Exception as e:
        print("ERROR:", e)
        break

    current_start = current_end + dt.timedelta(days=1)

if len(all_data):

    final_df = (
        pd.concat(all_data)
        .drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
    )

    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\n==============================")
    print("Download Complete")
    print("Rows :", len(final_df))
    print("File :", OUTPUT_FILE)
    print("==============================")

else:

    print("No data downloaded.")