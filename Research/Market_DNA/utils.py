import logging
import os
import config

def setup_logger():
    """Sets up unified logging for the Market DNA platform."""
    config.init_directories()
    log_file = os.path.join(config.LOGS_DIR, "market_dna.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("MarketDNA")
