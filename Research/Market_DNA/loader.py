import os
import pandas as pd
import logging
import sys
import config

logger = logging.getLogger("MarketDNA.Loader")

class DataLoader:
    def __init__(self):
        config.init_directories()

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(config.REAL_HISTORICAL_PATH):
            print(f"FAIL: Real historical data file not found at {config.REAL_HISTORICAL_PATH}.")
            logger.error(f"Critical Error: Missing real historical data file: {config.REAL_HISTORICAL_PATH}.")
            sys.exit(1)
            
        logger.info(f"Loading real historical data from {config.REAL_HISTORICAL_PATH}")
        df = pd.read_csv(config.REAL_HISTORICAL_PATH)
        
        # Standardize column names
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Verify datetime column exists
        if "datetime" not in df.columns:
            raise ValueError("Historical CSV must contain a 'datetime' column.")
            
        # 1 & 2. Read datetime as string and parse without forcing UTC offset stripping
        df["datetime"] = pd.to_datetime(df["datetime"], utc=False)
        
        # 3. Create trade_date column
        df["trade_date"] = df["datetime"].dt.date
        
        df = df.sort_values("datetime").reset_index(drop=True)
        
        candles_loaded = len(df)
        print(f"Minute candles loaded = {candles_loaded}")
        
        # 4. Print first 20 rows of datetime and trade_date
        print("\n--- First 20 Datetime and Trade Date values ---")
        print(df[["datetime", "trade_date"]].head(20))
        print("-" * 50)
        
        # Unique trading dates
        unique_dates = sorted(df["trade_date"].unique())
        total_unique_days = len(unique_dates)
        
        # 5 & 6. Print unique count, first trading day, and last trading day
        print(f"Number of unique trade_date values = {total_unique_days}")
        print(f"First trading day = {unique_dates[0] if unique_dates else 'N/A'}")
        print(f"Last trading day = {unique_dates[-1] if unique_dates else 'N/A'}")
        
        # 7. Validation check: Must have at least 100 trading days
        if total_unique_days < 100:
            print(f"\nFAIL: Unique trading days ({total_unique_days}) is below the required minimum of 100. Halting execution.")
            logger.error(f"Validation FAIL: Unique trading days ({total_unique_days}) < 100.")
            sys.exit(1)
            
        return df
