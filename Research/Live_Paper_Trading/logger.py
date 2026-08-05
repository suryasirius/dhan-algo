import os
import logging
import config

os.makedirs(config.LOGS_DIR, exist_ok=True)
log_path = os.path.join(config.LOGS_DIR, "live_trading.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("LivePaperTrading")
