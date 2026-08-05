from pathlib import Path

ROOT = Path.cwd()

PROJECT = ROOT / "PREVIOUS_DAY_PIVOT"

PROJECT.mkdir(exist_ok=True)

(PROJECT / "reports").mkdir(exist_ok=True)
(PROJECT / "output").mkdir(exist_ok=True)

config = r'''import os

DATA_FILE = r"C:\dhan algo\dhan_historical_data_180days.csv"

INITIAL_CAPITAL = 100000

LOT_SIZE = 75

BROKERAGE = 20

SLIPPAGE = 0.5

OUTPUT_FOLDER="output"

REPORT_FOLDER="reports"
'''

run = r'''import pandas as pd
import numpy as np
from config import *

print("="*60)
print("PREVIOUS DAY PIVOT RESEARCH")
print("="*60)

df = pd.read_csv(DATA_FILE)

df["datetime"]=pd.to_datetime(df["datetime"])

df=df.sort_values("datetime")

df["date"]=df["datetime"].dt.date

daily=df.groupby("date").agg(
    high=("high","max"),
    low=("low","min"),
    close=("close","last"),
    open=("open","first")
)

daily["pivot"]=(daily.high+daily.low+daily.close)/3

daily["r1"]=2*daily.pivot-daily.low
daily["s1"]=2*daily.pivot-daily.high

daily["r2"]=daily.pivot+(daily.high-daily.low)
daily["s2"]=daily.pivot-(daily.high-daily.low)

daily["r3"]=daily.high+2*(daily.pivot-daily.low)
daily["s3"]=daily.low-2*(daily.high-daily.pivot)

daily=daily.shift(1)

df=df.merge(
daily,
left_on="date",
right_index=True,
how="left"
)

print(df.head())

print()

print("Rows :",len(df))

print("Unique Days :",df.date.nunique())

print()

print("Pivot Ready")

'''