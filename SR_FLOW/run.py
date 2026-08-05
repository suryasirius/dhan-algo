from pathlib import Path
import pandas as pd
from config import DATA_FILE

print("=" * 60)
print("SR FLOW BACKTEST ENGINE")
print("=" * 60)

data_path = Path(DATA_FILE)

if not data_path.exists():
    print(f"\nERROR: Data file not found:\n{DATA_FILE}")
    exit()

print("\nLoading Data...")

df = pd.read_csv(DATA_FILE)

print(f"Rows Loaded : {len(df)}")
print(f"Columns     : {len(df.columns)}")

print("\nColumns:")

for c in df.columns:
    print(" -", c)

print("\nConverting Datetime...")

df["datetime"] = pd.to_datetime(df["datetime"])

df = df.sort_values("datetime").reset_index(drop=True)

print("Done.")

print("\nCreating 30 Minute Candles...")

htf = (
    df.set_index("datetime")
      .resample("30min")
      .agg({
          "open":"first",
          "high":"max",
          "low":"min",
          "close":"last",
          "volume":"sum"
      })
      .dropna()
      .reset_index()
)

print(f"30 Minute Candles : {len(htf)}")

print("\nLatest 5 Candles:\n")

print(htf.tail())

print("\nSUCCESS")