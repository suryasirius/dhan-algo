import os
import pandas as pd
import logging
import config

logger = logging.getLogger("MarketRegime.Loader")

class DataLoader:
    def __init__(self):
        config.init_directories()

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(config.INPUT_CSV_PATH):
            raise FileNotFoundError(f"Market DNA input file not found at: {config.INPUT_CSV_PATH}.")
        
        logger.info(f"Loading Market DNA dataset from {config.INPUT_CSV_PATH}")
        df = pd.read_csv(config.INPUT_CSV_PATH)
        
        print(f"Minute candles loaded / Raw input rows = {len(df)}")
        
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            # Extract unique trading days count if date has timestamps
            unique_days = df["date"].dt.date.nunique()
            print(f"Unique trading days detected = {unique_days}")
        else:
            print(f"Unique trading days detected = {len(df)} (Assumed 1 row per day)")
            
        return df

