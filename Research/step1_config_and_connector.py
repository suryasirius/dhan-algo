"""
Module: step1_config_and_connector.py
Description: Step 1 of the platform build. Sets up centralized configurations 
             and a robust Dhan API connection client with auto-retries.
"""

import os
import logging
import requests
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# This file lives at Research/step1_config_and_connector.py, so its own
# directory IS the Research root. Allow override via DHAN_ALGO_ROOT.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RESEARCH_ROOT = os.environ.get("DHAN_ALGO_ROOT", _THIS_DIR)

# Configure logging (guarded so importing this module doesn't clobber a
# logging config already set up elsewhere, e.g. by step9_live_engine.py).
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
logger = logging.getLogger("Step1_DhanSetup")


@dataclass
class PlatformConfig:
    """Centralized switches and configuration parameters."""
    enable_oi_filter: bool = True
    enable_risk_filter: bool = True
    enable_confidence_engine: bool = True
    enable_strategy_selector: bool = True
    enable_logging: bool = True

    dhan_client_id: str = os.getenv("DHAN_CLIENT_ID", "1000000001")
    dhan_access_token: str = os.getenv("DHAN_ACCESS_TOKEN", "YOUR_TOKEN_HERE")
    dhan_base_url: str = "https://api.dhan.co/v2"
    
    root_dir: str = _RESEARCH_ROOT
    log_dir: str = os.path.join(_RESEARCH_ROOT, "logs")
    output_dir: str = os.path.join(_RESEARCH_ROOT, "output_reports")
    min_confidence_threshold: float = 70.0

    # Underlying instrument used by the live engine / dashboard (Nifty 50 index)
    nifty_security_id: str = "13"
    nifty_exchange_segment: str = "IDX_I"

    def __post_init__(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        # Prefer the token saved via token_manager (e.g. from the dashboard) over env vars.
        try:
            from token_manager import get_access_token, get_client_id
            saved_token = get_access_token()
            saved_client = get_client_id()
            if saved_token:
                self.dhan_access_token = saved_token
            if saved_client:
                self.dhan_client_id = saved_client
        except Exception:
            pass


class DhanConnector:
    """Robust API connector for Dhan with built-in retry mechanisms."""
    def __init__(self, config: PlatformConfig):
        self.config = config
        self.session = self._create_robust_session()

    def _create_robust_session(self) -> requests.Session:
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.headers.update({
            "Content-Type": "application/json",
            "access-token": self.config.dhan_access_token,
            "client-id": self.config.dhan_client_id
        })
        return session

    def refresh_credentials(self):
        """Re-reads the access token / client id (e.g. after the user updates them
        from the dashboard sidebar) and applies them to the live session headers."""
        try:
            from token_manager import get_access_token, get_client_id
            token = get_access_token()
            client_id = get_client_id()
            if token:
                self.session.headers["access-token"] = token
            if client_id:
                self.session.headers["client-id"] = client_id
        except Exception as e:
            logger.warning(f"Could not refresh Dhan credentials: {e}")

    def fetch_expiry_list(self, security_id: str = None, exchange_segment: str = None):
        """Fetches available option expiry dates for the underlying. Returns [] on failure."""
        security_id = security_id or self.config.nifty_security_id
        exchange_segment = exchange_segment or self.config.nifty_exchange_segment
        endpoint = f"{self.config.dhan_base_url}/optionchain/expirylist"
        payload = {"UnderlyingScrip": int(security_id), "UnderlyingSeg": exchange_segment}
        try:
            response = self.session.post(endpoint, json=payload, timeout=10)
            if response.status_code == 200:
                return response.json().get("data", []) or []
            logger.error(f"Expiry list API error: {response.status_code} - {response.text}")
            return []
        except Exception as e:
            logger.exception(f"Expiry list fetch failed: {e}")
            return []

    def fetch_option_chain(self, expiry: str, security_id: str = None, exchange_segment: str = None):
        """Fetches the live option chain (per-strike CE/PE OI, IV, LTP, volume) for the
        given expiry. Returns None on any failure - callers must NOT synthesize data."""
        security_id = security_id or self.config.nifty_security_id
        exchange_segment = exchange_segment or self.config.nifty_exchange_segment
        endpoint = f"{self.config.dhan_base_url}/optionchain"
        payload = {
            "UnderlyingScrip": int(security_id),
            "UnderlyingSeg": exchange_segment,
            "Expiry": expiry
        }
        try:
            response = self.session.post(endpoint, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json().get("data", {})
                return data or None
            logger.error(f"Option chain API error: {response.status_code} - {response.text}")
            return None
        except Exception as e:
            logger.exception(f"Option chain fetch failed: {e}")
            return None

    def fetch_historical_intraday(self, security_id: str, exchange_segment: str,
                                   from_date: str = None, to_date: str = None,
                                   interval: str = "5") -> pd.DataFrame:
        """Fetches intraday candles. Defaults to today-only (used by the live
        engine's per-cycle polling). Pass explicit from_date/to_date (YYYY-MM-DD
        strings) to pull a multi-day warmup window instead - e.g. for indicators
        that need more history than a single day provides (added for
        step10_live_paper_best2.py's six_concepts SMA200 warmup)."""
        import datetime as _dt
        endpoint = f"{self.config.dhan_base_url}/charts/intraday"
        today_str = _dt.datetime.now().strftime("%Y-%m-%d")
        payload = {
            "securityId": security_id,
            "exchangeSegment": exchange_segment,
            "instrument": "INDEX",
            "interval": interval,
            "fromDate": from_date or today_str,
            "toDate": to_date or today_str
        }
        try:
            logger.info(f"Fetching historical data for security {security_id}...")
            response = self.session.post(endpoint, json=payload, timeout=10)
            if response.status_code == 200:
                body = response.json()
                # Dhan's intraday endpoint returns candle arrays either nested under
                # "data" or directly at the top level, depending on API version.
                data = body.get("data") if isinstance(body, dict) and body.get("data") else body
                if not data or not data.get("open"):
                    logger.warning(f"API returned no usable candle data. Raw response: {response.text[:500]}")
                    return pd.DataFrame()

                n = len(data.get("open", []))
                timestamps = data.get("timestamp") or data.get("start_Time") or ([None] * n)
                return pd.DataFrame({
                    "timestamp": timestamps,
                    "open": data.get("open", []),
                    "high": data.get("high", []),
                    "low": data.get("low", []),
                    "close": data.get("close", []),
                    "volume": data.get("volume", []),
                    "call_oi": data.get("oi", []) or [0] * len(data.get("open", [])),
                    "put_oi": data.get("oi", []) or [0] * len(data.get("open", [])),
                    "call_vol": data.get("volume", []),
                    "put_vol": data.get("volume", [])
                })
            else:
                logger.error(f"Dhan API Error: {response.status_code} - {response.text}")
                return pd.DataFrame()
        except Exception as e:
            logger.exception(f"Connection exception occurred: {e}")
            return pd.DataFrame()