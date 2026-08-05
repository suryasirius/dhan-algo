"""
Diagnostic only -- follow-up to v5, which hit two things:
  1. A rate limit (DH-904) from firing too many calls back-to-back --
     fixed here with a delay between every call.
  2. Every call that DIDN'T rate-limit or error still came back with
     genuinely empty data (0 close values) across expiry_code 1/2/3 and
     both a numeric strike and the literal "ATM", for drv_option_type
     ="CALL". That's a real, repeated signal -- not a fluke.

This tries the one clear remaining candidate fix: drv_option_type="CE"
instead of "CALL" (every other part of this project -- resolve_itm_option,
live order placement -- uses "CE"/"PE", never "CALL"/"PUT"). Deliberately
small number of calls (6 total, 4s apart) to stay well clear of the rate
limit and get a clean read.

Does not modify dhan_config.py, dhan_live_bot.py, or any other file.
Safe to run any time -- read-only historical data calls, no orders.
"""

import time
import datetime as dt

from dhan_live_bot import dhan

NIFTY_UNDERLYING_SECURITY_ID = 13
DELAY_SECONDS = 4


def show_ce_pe_summary(resp, label):
    if isinstance(resp, dict) and resp.get("status") == "failure":
        print(f"  [{label}] API ERROR: {resp.get('remarks')}")
        return False
    data = resp.get("data", resp) if isinstance(resp, dict) else resp
    if isinstance(data, dict) and "data" in data and not any(k in data for k in ("ce", "pe")):
        data = data["data"]
    found_any = False
    for side in ("ce", "pe"):
        block = data.get(side) if isinstance(data, dict) else None
        if block:
            closes = block.get("close") or []
            if closes:
                print(f"  [{label}] '{side}' REAL DATA -- {len(closes)} closes, "
                      f"strike={block.get('strike')}, first 3={closes[:3]}")
                found_any = True
            else:
                print(f"  [{label}] '{side}' key present but empty (0 closes)")
    if not found_any:
        print(f"  [{label}] No usable data. Raw (first 400 chars): {str(resp)[:400]}")
    return found_any


def try_call(expiry_code, strike, option_type, from_date, to_date, label):
    try:
        resp = dhan.expired_options_data(
            security_id=NIFTY_UNDERLYING_SECURITY_ID,
            exchange_segment="IDX_I",
            instrument_type="OPTIDX",
            expiry_flag="WEEK",
            expiry_code=expiry_code,
            strike=strike,
            drv_option_type=option_type,
            required_data=["open", "high", "low", "close", "volume", "oi", "iv", "spot"],
            from_date=from_date,
            to_date=to_date,
            interval=1,
        )
        return show_ce_pe_summary(resp, label)
    except Exception as e:
        print(f"  [{label}] EXCEPTION: {e}")
        return False


def main():
    to_date_recent = dt.date.today().strftime("%Y-%m-%d")
    from_date_recent = (dt.date.today() - dt.timedelta(days=20)).strftime("%Y-%m-%d")

    print("--- Testing drv_option_type='CE' (project's actual convention) vs 'CALL' ---")
    combos = [
        (1, 25000, "CE"),
        (1, 25000, "CALL"),
        (2, 25000, "CE"),
        (2, "ATM", "CE"),
        (3, 25000, "PE"),
        (1, 25000, "PE"),
    ]
    working = []
    for i, (code, strike_val, opt_type) in enumerate(combos):
        label = f"code={code}, strike={strike_val}, type={opt_type}"
        ok = try_call(code, strike_val, opt_type, from_date_recent, to_date_recent, label)
        if ok:
            working.append((code, strike_val, opt_type))
        if i < len(combos) - 1:
            time.sleep(DELAY_SECONDS)

    print(f"\nWorking combos: {working}")
    print("\n=== SUMMARY ===")
    if working:
        print("Found at least one combo with real data -- next step is the historical-range probe.")
    else:
        print("Still nothing. At this point every plausible parameter fix (strike format, ")
        print("option-type spelling, expiry_code 0/1/2/3) has been tried and all come back ")
        print("either rate-limited or genuinely empty. This looks less like a parameter bug ")
        print("and more like this Dhan plan/account not being entitled to real historical ")
        print("option premium data via this endpoint.")


if __name__ == "__main__":
    main()
