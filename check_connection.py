print("=== check2.py running ===")

import datetime as dt
from dhanhq import DhanContext, dhanhq
from dhan_config import (
    CLIENT_ID, ACCESS_TOKEN, UNDERLYING_SECURITY_ID,
    UNDERLYING_EXCHANGE_SEGMENT, UNDERLYING_INSTRUMENT_TYPE,
)

print("Connecting to Dhan...")
dhan_context = DhanContext(CLIENT_ID, ACCESS_TOKEN)
dhan = dhanhq(dhan_context)

to_date = dt.datetime.now().strftime("%Y-%m-%d")
from_date = (dt.datetime.now() - dt.timedelta(days=5)).strftime("%Y-%m-%d")
print(f"Requesting Nifty candles from {from_date} to {to_date} ...")

raw = dhan.intraday_minute_data(
    security_id=UNDERLYING_SECURITY_ID,
    exchange_segment=UNDERLYING_EXCHANGE_SEGMENT,
    instrument_type=UNDERLYING_INSTRUMENT_TYPE,
    from_date=from_date,
    to_date=to_date,
)

print("\n--- RAW RESPONSE (copy all of this back to Claude) ---\n")
print(raw)

data = raw.get("data", raw) if isinstance(raw, dict) else {}
closes = data.get("close", [])
print(f"\nNumber of candles returned: {len(closes)}")
if closes:
    print(f"First close: {closes[0]}   Last close: {closes[-1]}")
