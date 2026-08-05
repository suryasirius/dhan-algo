import datetime as dt
import pandas as pd
from dhanhq import DhanContext,dhanhq
from dhan_config import CLIENT_ID,ACCESS_TOKEN,UNDERLYING_SECURITY_ID,UNDERLYING_EXCHANGE_SEGMENT,UNDERLYING_INSTRUMENT_TYPE
DAYS_BACK=180
OUTPUT_FILE="dhan_historical_data.csv"
dhan=dhanhq(DhanContext(CLIENT_ID,ACCESS_TOKEN))
today=dt.datetime.now().strftime("%Y-%m-%d")
from_date=(dt.datetime.now()-dt.timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")
resp=dhan.intraday_minute_data(security_id=UNDERLYING_SECURITY_ID,exchange_segment=UNDERLYING_EXCHANGE_SEGMENT,instrument_type=UNDERLYING_INSTRUMENT_TYPE,from_date=from_date,to_date=today)
data=resp.get("data",resp)
df=pd.DataFrame({"timestamp":data["timestamp"],"open":data["open"],"high":data["high"],"low":data["low"],"close":data["close"]})
if "volume" in data: df["volume"]=data["volume"]
df["datetime"]=pd.to_datetime(df["timestamp"],unit="s")+pd.Timedelta(hours=5,minutes=30)
df=df[["datetime"]+[c for c in df.columns if c!="datetime"]]
df.to_csv(OUTPUT_FILE,index=False)
print("Saved",OUTPUT_FILE)
