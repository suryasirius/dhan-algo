import os
import logging
import pandas as pd
from pathlib import Path
import config

def setup_directories() -> None:
    """Ensures all output, report, and chart directories exist."""
    for directory in [config.REPORTS_DIR, config.OUTPUT_DIR, config.CHARTS_DIR]:
        os.makedirs(directory, exist_ok=True)

def setup_logger(name: str = "ORB_ML_Optimizer") -> logging.Logger:
    """Sets up dual logging to console and file."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        log_file = config.OUTPUT_DIR / "execution.log"
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

def load_and_clean_data(logger: logging.Logger) -> pd.DataFrame:
    """Loads historical CSV data, parses datetime index, and verifies schema."""
    filepath = config.DATA_FILE
    logger.info(f"Loading historical market data from {filepath}")
    try:
        df = pd.read_csv(filepath)
        df[config.COL_DATETIME] = pd.to_datetime(df[config.COL_DATETIME])
        df.set_index(config.COL_DATETIME, inplace=True)
        df.sort_index(inplace=True)
        
        required_cols = [config.COL_OPEN, config.COL_HIGH, config.COL_LOW, config.COL_CLOSE, config.COL_VOLUME]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column in dataset: {col}")
        
        df.dropna(subset=required_cols, inplace=True)
        logger.info(f"Successfully loaded {len(df)} 1-minute historical records.")
        return df
        
    except FileNotFoundError:
        logger.error(f"Data file not found at {filepath}. Please verify path.")
        raise
    except Exception as e:
        logger.error(f"Error loading historical data: {e}")
        raise