"""
Module: token_manager.py
Description: Manages validation, retrieval, and updating of the Dhan API access token.
"""

import os

# This file lives at Research/token_manager.py, so its own directory IS the
# Research root. Allow override via DHAN_ALGO_RESEARCH_ROOT for flexibility.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RESEARCH_ROOT = os.environ.get("DHAN_ALGO_RESEARCH_ROOT", _THIS_DIR)

TOKEN_FILE = os.path.join(_RESEARCH_ROOT, "logs", "dhan_token.txt")
CLIENT_ID_FILE = os.path.join(_RESEARCH_ROOT, "logs", "dhan_client_id.txt")

def get_access_token():
    """Reads the access token from file if it exists."""
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r") as f:
                return f.read().strip()
        except Exception:
            return ""
    return os.environ.get("DHAN_ACCESS_TOKEN", "")

def get_client_id():
    """Reads the client ID from file if it exists."""
    if os.path.exists(CLIENT_ID_FILE):
        try:
            with open(CLIENT_ID_FILE, "r") as f:
                return f.read().strip()
        except Exception:
            return ""
    return os.environ.get("DHAN_CLIENT_ID", "")

def update_access_token(new_token):
    """Saves the new access token to the persistent storage file."""
    os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
    with open(TOKEN_FILE, "w") as f:
        f.write(new_token.strip())

def validate_token():
    """Validates if an access token is present and non-empty."""
    token = get_access_token()
    return bool(token and len(token) > 20)