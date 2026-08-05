"""
Module: dhan_integration_master.py
Description: Production-grade, unified Dhan integration layer featuring strict 
             pre-flight parameter validation, automated payload construction, 
             rate-limiting, retry logic, and detailed error diagnostics.
"""

import os
import time
import logging
import requests
import pandas as pd
from typing import Dict, Any, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DhanMasterClient")


class DhanParameterError(Exception):
    """Raised when mandatory API parameters are missing or invalid during pre-flight checks."""
    pass


class DhanRequestValidator:
    """Validates required parameters against Dhan API specifications before sending requests."""
    
    @staticmethod
    def validate_historical_params(params: Dict[str, Any]) -> None:
        required = ['securityId', 'exchangeSegment', 'instrument', 'fromDate', 'toDate']
        for field in required:
            if not params.get(field):
                raise DhanParameterError(
                    f"Missing required parameter '{field}' for Historical/Intraday request. "
                    f"Provided parameters: {params}"
                )

    @staticmethod
    def validate_option_chain_params(params: Dict[str, Any]) -> None:
        required = ['UnderlyingScrip', 'UnderlyingSeg', 'Expiry']
        for field in required:
            if not params.get(field):
                raise DhanParameterError(
                    f"Missing required parameter '{field}' for Option Chain request. "
                    f"Provided parameters: {params}"
                )

    @staticmethod
    def validate_live_feed_params(params: Dict[str, Any]) -> None:
        required = ['securityId', 'exchangeSegment']
        for field in required:
            if not params.get(field):
                raise DhanParameterError(
                    f"Missing required parameter '{field}' for Live Market Feed request. "
                    f"Provided parameters: {params}"
                )


class DhanRequestBuilder:
    """Automatically populates, formats, and validates payloads for all Dhan endpoints."""
    
    @classmethod
    def build_historical_candles(
        cls, 
        security_id: str, 
        exchange_segment: str, 
        instrument: str, 
        from_date: str, 
        to_date: str, 
        interval: str = "5", 
        oi: bool = True, 
        expiry_code: int = 0
    ) -> Dict[str, Any]:
        payload = {
            "securityId": str(security_id),
            "exchangeSegment": str(exchange_segment),
            "instrument": str(instrument),
            "interval": str(interval),
            "oi": bool(oi),
            "fromDate": str(from_date),
            "toDate": str(to_date),
            "expiryCode": int(expiry_code)
        }
        DhanRequestValidator.validate_historical_params(payload)
        return payload

    @classmethod
    def build_option_chain(
        cls, 
        underlying_scrip: str, 
        underlying_seg: str, 
        expiry_date: str
    ) -> Dict[str, Any]:
        payload = {
            "UnderlyingScrip": str(underlying_scrip),
            "UnderlyingSeg": str(underlying_seg),
            "Expiry": str(expiry_date)
        }
        DhanRequestValidator.validate_option_chain_params(payload)
        return payload

    @classmethod
    def build_live_feed_payload(
        cls, 
        security_id: str, 
        exchange_segment: str
    ) -> Dict[str, Any]:
        payload = {
            "securityId": str(security_id),
            "exchangeSegment": str(exchange_segment)
        }
        DhanRequestValidator.validate_live_feed_params(payload)
        return payload


class HardenedDhanClient:
    """Production-grade client implementing rate limits, retries, and diagnostics."""
    
    def __init__(self, client_id: str, access_token: str, base_url: str = "https://api.dhan.co/v2"):
        self.client_id = client_id
        self.access_token = access_token
        self.base_url = base_url
        self.session = self._create_rate_limited_session()
        self.last_request_time = 0.0
        self.rate_limit_delay = 0.25  # Max ~4 requests per second protection

    def _create_rate_limited_session(self) -> requests.Session:
        session = requests.Session()
        retries = Retry(
            total=4,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
            "access-token": self.access_token,
            "client-id": self.client_id
        })
        return session

    def _enforce_rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _handle_error_response(self, endpoint: str, payload: Dict[str, Any], response: requests.Response):
        status_code = response.status_code
        err_code = "UNKNOWN"
        err_msg = response.text
        
        try:
            err_json = response.json()
            err_code = err_json.get("errorCode", "UNKNOWN")
            err_msg = err_json.get("errorMessage", response.text)
        except Exception:
            pass

        suggested_fix = "Verify credentials, token expiry, and payload parameters."
        if err_code == "DH-905" or status_code == 400:
            suggested_fix = "Check if 'fromDate', 'toDate', or mandatory parameters are formatted correctly (YYYY-MM-DD HH:MM:SS) and non-empty."
        elif status_code == 401:
            suggested_fix = "Access token is invalid or expired. Regenerate your token from the Dhan dashboard."
        elif status_code == 429:
            suggested_fix = "Rate limit exceeded. Slow down request frequency."

        print("\n" + "="*50)
        print("DHAN API ERROR DIAGNOSTICS")
        print("="*50)
        print(f"Endpoint       : {endpoint}")
        print(f"Payload        : {payload}")
        print(f"HTTP Status    : {status_code}")
        print(f"Error Code     : {err_code}")
        print(f"Error Message  : {err_msg}")
        print(f"Suggested Fix  : {suggested_fix}")
        print("="*50 + "\n")
        
        logger.error(f"API Call Failed -> Endpoint: {endpoint} | Code: {err_code} | Msg: {err_msg}")

    def execute_post(self, endpoint_path: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self._enforce_rate_limit()
        endpoint = f"{self.base_url}{endpoint_path}"
        logger.info(f"Executing POST request to {endpoint} with payload: {payload}")
        
        try:
            response = self.session.post(endpoint, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info(f"Successfully received response from {endpoint_path}")
                return response.json()
            else:
                self._handle_error_response(endpoint, payload, response)
                return None
        except Exception as e:
            logger.exception(f"Network or execution exception during POST to {endpoint}: {e}")
            return None