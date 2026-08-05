"""
build_training_dataset.py

Reads dhan_historical_data.csv and creates an AI-ready dataset.
"""

import pandas as pd
import numpy as np

INPUT_FILE = "dhan_historical_data.csv"
OUTPUT_FILE = "training_dataset.csv"

df = pd.read_csv(INPUT_FILE, parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

# Basic features
df["body"] = df["close"] - df["open"]
df["range"] = df["high"] - df["low"]
df["upper_wick"] = df["high"] - df[["open","close"]].max(axis=1)
df["lower_wick"] = df[["open","close"]].min(axis=1) - df["low"]

# EMA
df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

# RSI(14)
delta = df["close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
rs = avg_gain / avg_loss.replace(0, np.nan)
df["rsi14"] = 100 - (100 / (1 + rs))

# ATR(14)
prev_close = df["close"].shift()
tr = pd.concat([
    df["high"] - df["low"],
    (df["high"] - prev_close).abs(),
    (df["low"] - prev_close).abs()
], axis=1).max(axis=1)
df["atr14"] = tr.ewm(alpha=1/14, adjust=False).mean()

# 15-minute grouping
mins = df["datetime"].dt.minute
df["htf_bar"] = mins // 15
df["minute_in_15m"] = (mins % 15) + 1

grp = df.groupby(df["datetime"].dt.floor("15min"))

df["htf_open"] = grp["open"].transform("first")
df["htf_high_so_far"] = grp["high"].cummax()
df["htf_low_so_far"] = grp["low"].cummin()
df["htf_close_so_far"] = df["close"]

# Final HTF values (prediction targets)
df["final_15m_close"] = grp["close"].transform("last")
df["final_15m_high"] = grp["high"].transform("max")
df["final_15m_low"] = grp["low"].transform("min")
df["target_bullish"] = (df["final_15m_close"] > df["htf_open"]).astype(int)

# Remaining minutes
df["minutes_left"] = 15 - df["minute_in_15m"]

df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {OUTPUT_FILE}")
print(df.head())
