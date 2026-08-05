"""
Module: update_dhan_token.py
Description: One small, easy way to refresh your Dhan access token daily.
             Writes directly to the exact two files every script in this
             project reads credentials from:
                 Research/logs/dhan_token.txt       (access token)
                 Research/logs/dhan_client_id.txt   (client ID)
             These are read by token_manager.py, which dhan_config.py (and
             therefore dhan_live_bot.py, om/sc's fetch functions, and every
             top-level backtest script) and step1_config_and_connector.py
             (step9/step10's live engines) both pull from -- so refreshing
             here fixes ALL of them at once, not just one script.

TWO WAYS TO USE IT:

  1. One shell command, paste the token straight in (fastest for daily use):
         python update_dhan_token.py YOUR_NEW_ACCESS_TOKEN
     or, if your client ID also changed:
         python update_dhan_token.py YOUR_NEW_ACCESS_TOKEN YOUR_CLIENT_ID

  2. Just run it with nothing after it, and it will ask you to paste the
     token (and client ID, optional) interactively:
         python update_dhan_token.py

Either way, it prints back what got saved (token partially masked) so you
can confirm it worked, and it does NOT touch anything else in the project.
"""

import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RESEARCH_DIR = os.path.join(_THIS_DIR, "Research")
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)

from token_manager import update_access_token, get_client_id, TOKEN_FILE, CLIENT_ID_FILE


def update_client_id(new_client_id: str):
    os.makedirs(os.path.dirname(CLIENT_ID_FILE), exist_ok=True)
    with open(CLIENT_ID_FILE, "w") as f:
        f.write(new_client_id.strip())


def mask(token: str) -> str:
    token = token.strip()
    if len(token) <= 12:
        return "*" * len(token)
    return f"{token[:6]}...{token[-6:]} ({len(token)} chars)"


def main():
    args = sys.argv[1:]

    if args:
        new_token = args[0].strip()
        new_client_id = args[1].strip() if len(args) > 1 else None
    else:
        new_token = input("Paste your new Dhan ACCESS TOKEN: ").strip()
        new_client_id = input("Paste your CLIENT ID (press Enter to leave unchanged): ").strip() or None

    if not new_token:
        print("No token entered -- nothing was changed.")
        sys.exit(1)

    update_access_token(new_token)
    print(f"Access token saved to: {TOKEN_FILE}")
    print(f"  -> {mask(new_token)}")

    if new_client_id:
        update_client_id(new_client_id)
        print(f"Client ID saved to: {CLIENT_ID_FILE}")
        print(f"  -> {new_client_id}")
    else:
        print(f"Client ID left unchanged: {get_client_id()}")

    print("\nDone. This is now live for every script in the project -- "
          "step9_live_engine.py, step10_live_paper_best2.py, and every "
          "top-level backtest/probe script all read from the same file.")


if __name__ == "__main__":
    main()
