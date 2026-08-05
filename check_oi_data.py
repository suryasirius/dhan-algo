"""
Quick diagnostic -- NOT a strategy, just a check (same idea as check_volume_data.py).

Your idea: use Nifty OPEN INTEREST (OI) buildup -- one side (calls vs puts, or
futures OI direction) heavily loaded -- as a confirming filter alongside the
30m/15m/5m/1m trend-agreement logic we already use in Concepts 2&4, plus
volume confirmation.

Before building this, there's a real question to answer first: Nifty 50 the
INDEX has no OI at all (only its derivatives -- futures and options -- carry
open interest). So this needs a DIFFERENT security than the one all our other
scripts use (UNDERLYING_SECURITY_ID = the index). This script:

  1. Uses the SAME instrument-master lookup dhan_live_bot.py already has
     (load_instrument_master) to find the current-month NIFTY FUTURES
     contract, and the nearest-expiry ATM CE/PE option contracts.
  2. Requests 1-min intraday data for each of those from Dhan and checks
     whether the response actually includes an "oi"/"open_interest" field,
     and if so, whether it's populated with real changing values or just
     zeros/flat.

Depending on what comes back, one of three things is possible:
  - Real per-minute OI history is available -> we can build the full concept
    as you described.
  - OI is only available as a live/current snapshot (not a historical
    time series) -> we CANNOT backtest OI-based rules on the past, only
    watch it during actual live trading -- a very different (and much more
    limited) thing.
  - Not available at all via this API -> we'd need a different data source
    entirely, or drop the OI part and keep the trend+volume part alone.

Does not modify any other file.
"""

import datetime as dt
import pandas as pd

from dhan_live_bot import (
    dhan, load_instrument_master,
)

TO_DATE = dt.date.today().strftime("%Y-%m-%d")
FROM_DATE = (dt.date.today() - dt.timedelta(days=5)).strftime("%Y-%m-%d")


def inspect_response(label, security_id, exchange_segment, instrument_type):
    print(f"\n--- {label} (security_id={security_id}, segment={exchange_segment}, type={instrument_type}) ---")
    try:
        resp = dhan.intraday_minute_data(
            security_id=str(security_id),
            exchange_segment=exchange_segment,
            instrument_type=instrument_type,
            from_date=FROM_DATE,
            to_date=TO_DATE,
        )
    except Exception as e:
        print(f"Request failed: {e}")
        return

    if isinstance(resp, dict) and resp.get("status") == "failure":
        print("Dhan returned an error:")
        print(resp)
        return

    data = resp.get("data", resp)
    print("Keys present:", list(data.keys()))

    for key in ("oi", "open_interest", "openInterest"):
        if key in data:
            oi = pd.Series(data[key])
            n_total = len(oi)
            n_zero = int((oi == 0).sum())
            n_unique = oi.nunique()
            print(f"Found '{key}' field: {n_total} candles, {n_zero} are zero, "
                  f"{n_unique} unique values (min={oi.min()}, max={oi.max()}).")
            if n_unique <= 1:
                print(">>> All identical/flat -- likely NOT a real changing time series.")
            else:
                print(">>> Values actually change over time -- looks like a usable OI series.")
            return
    print(">>> No OI/open_interest field found in this response at all.")


def main():
    print("Loading Dhan instrument master...")
    df = load_instrument_master()
    print(f"Instrument master loaded: {len(df)} rows.")
    print("Available SEM_EXCH_INSTRUMENT_TYPE values:", df["SEM_EXCH_INSTRUMENT_TYPE"].unique())

    # --- 1. Current-month NIFTY FUTURES contract ---
    symbol_is_nifty = df["SEM_TRADING_SYMBOL"].astype(str).str.upper().str.match(r"^NIFTY[\s\-]")
    futs = df[symbol_is_nifty & (df["SEM_EXCH_INSTRUMENT_TYPE"] == "FUTIDX")].copy()
    if not futs.empty:
        futs["SEM_EXPIRY_DATE"] = pd.to_datetime(futs["SEM_EXPIRY_DATE"])
        today = pd.Timestamp(dt.date.today())
        futs = futs[futs["SEM_EXPIRY_DATE"] >= today].sort_values("SEM_EXPIRY_DATE")
        if not futs.empty:
            row = futs.iloc[0]
            inspect_response(
                f"NIFTY FUT ({row['SEM_TRADING_SYMBOL']}, expiry {row['SEM_EXPIRY_DATE'].date()})",
                row["SEM_SMST_SECURITY_ID"], "NSE_FNO", "FUTIDX",
            )
        else:
            print("\nNo upcoming-expiry NIFTY FUTIDX rows found.")
    else:
        print("\nNo NIFTY FUTIDX rows found at all -- check instrument master column/format.")

    # --- 2. Nearest-expiry ATM CE and PE options ---
    opts = df[symbol_is_nifty & (df["SEM_EXCH_INSTRUMENT_TYPE"] == "OPTIDX")].copy()
    if opts.empty:
        print("\nNo NIFTY OPTIDX rows found at all.")
        return

    opts["SEM_EXPIRY_DATE"] = pd.to_datetime(opts["SEM_EXPIRY_DATE"])
    today = pd.Timestamp(dt.date.today())
    opts = opts[opts["SEM_EXPIRY_DATE"] >= today]
    nearest_expiry = opts["SEM_EXPIRY_DATE"].min()
    opts = opts[opts["SEM_EXPIRY_DATE"] == nearest_expiry]

    # crude ATM guess: median strike of the nearest-expiry chain
    strikes = sorted(opts["SEM_STRIKE_PRICE"].unique())
    atm_strike = strikes[len(strikes) // 2]

    for opt_type in ["CE", "PE"]:
        row_df = opts[(opts["SEM_STRIKE_PRICE"] == atm_strike) & (opts["SEM_OPTION_TYPE"] == opt_type)]
        if row_df.empty:
            print(f"\nNo {opt_type} row found at strike {atm_strike}.")
            continue
        row = row_df.iloc[0]
        inspect_response(
            f"NIFTY {opt_type} {atm_strike} (expiry {nearest_expiry.date()}, approx ATM)",
            row["SEM_SMST_SECURITY_ID"], "NSE_FNO", "OPTIDX",
        )


if __name__ == "__main__":
    main()
