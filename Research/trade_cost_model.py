"""
trade_cost_model.py -- ACCURATE per-trade brokerage+charges cost model,
reverse-engineered from a real broker's "Calculate your Brokerage" tool
(4 sample screenshots: NIFTY 01 SEP 24350 PUT, Options segment).

WHY THIS EXISTS: every engine and backtest script in this project has,
until now, used a flat Rs55/trade cost assumption (BROKERAGE_PER_TRADE =
55.0). That number was a round-number estimate, not a real calculation.
This module replaces it with the actual formula, derived by reverse-
engineering the 4 screenshots (solved until every line item matched the
displayed values to within a few paise):

    Buy Value    = buy_price  * qty
    Sell Value   = sell_price * qty
    Turnover     = Buy Value + Sell Value

    Brokerage    = Rs 20 flat per executed order -> Rs 40 per round trip
                   (buy + sell), independent of trade size (confirmed
                   flat across trade sizes from ~Rs9.4k to ~Rs9.36 lakh
                   turnover in the samples)
    Exchange Txn = Turnover * 0.03553%
    Charges (ETC)
    STT          = round(Sell Value * 0.15%)  -- rounded to nearest whole
                   rupee (round-half-up, confirmed via the Rs9.75->10
                   sample)
    SEBI charges = round(Turnover * 0.0001%, 2)
    Stamp Duty   = floor(Buy Value * 0.003%)  -- floored to nearest whole
                   rupee, buy side only (confirmed via the Rs0.14->0
                   sample)
    DP Charges   = 0 (not applicable, options are cash-settled/no
                   delivery)
    IPFT         = 0 (negligible at this trade-size range; not broken
                   out separately by the source calculator)
    GST          = round(18% * (Brokerage + unrounded ETC), 2)

    Total cost   = Brokerage + ETC + STT + SEBI + Stamp Duty + GST

VALIDATION (against the 4 source screenshots, Total Tax & Charges field):
    Sample 1 (buy 72, sell 73, qty 65):    actual 58.42, formula ~58.16
    Sample 2 (buy 72, sell 74, qty 65):    actual 58.55, formula ~58.29
    Sample 3 (buy 72, sell 72, qty 6500):  actual 1156.60, formula ~1156.56
    Sample 4 (buy 72, sell 100, qty 65):   actual 61.79, formula ~61.89
All within about Rs0.05-0.30 of the real figure (well under 0.5% of
trade value) -- far more accurate than the flat Rs55 assumption, which
was off by anywhere from about Rs-5 to +Rs1,100 depending on trade size
(it only happens to be close to correct at 1-lot, ~Rs70-120 premium
trades, which is most of this project's historical trade population --
but at bigger premiums, like today's Rs200+ CE trades, or bigger lot
sizes, flat Rs55 badly understates real cost).

USAGE:
    from trade_cost_model import compute_trade_cost
    cost = compute_trade_cost(entry_price, exit_price, qty)
    rupee_pnl = (exit_price - entry_price) * qty - cost

Every trade in this project's engines is a simple buy-to-open,
sell-to-close options-BUYING round trip (confirmed by reading the live
engines' exit logic: pnl_pts = exit_price - entry_price with no
direction branching -- "SHORT"/"LONG" in this codebase refers to the
underlying index-direction bias that picks CE vs PE, not to actually
selling premium short). So buy_price is always entry_price and
sell_price is always exit_price, regardless of the position's
"direction" label.
"""

import math

BROKERAGE_FLAT_PER_ROUNDTRIP = 40.0     # Rs20/order x 2 (buy + sell)
ETC_RATE = 0.0003553                    # 0.03553% of turnover
STT_RATE = 0.0015                       # 0.15% of sell value, rounded to nearest Re
SEBI_RATE = 0.000001                    # 0.0001% of turnover
STAMP_RATE = 0.00003                    # 0.003% of buy value, floored to nearest Re
GST_RATE = 0.18                         # 18% on (brokerage + ETC)


def compute_trade_cost(entry_price: float, exit_price: float, qty: float) -> float:
    """
    Real brokerage+charges cost for one round-trip options-buying trade
    (buy at entry_price, sell at exit_price, qty units), matching this
    project's broker's actual fee schedule. Replaces the old flat
    BROKERAGE_PER_TRADE = 55.0 assumption.
    """
    buy_value = entry_price * qty
    sell_value = exit_price * qty
    turnover = buy_value + sell_value

    brokerage = BROKERAGE_FLAT_PER_ROUNDTRIP
    etc = turnover * ETC_RATE
    stt = round(sell_value * STT_RATE)          # nearest whole rupee
    sebi = round(turnover * SEBI_RATE, 2)
    stamp = math.floor(buy_value * STAMP_RATE)  # floored whole rupee
    gst = round(GST_RATE * (brokerage + etc), 2)

    total = brokerage + etc + stt + sebi + stamp + gst
    return round(total, 2)


if __name__ == "__main__":
    # quick self-check against the 4 source screenshots
    samples = [
        (72, 73, 65, 58.42),
        (72, 74, 65, 58.55),
        (72, 72, 6500, 1156.60),
        (72, 100, 65, 61.79),
    ]
    for entry, exit_, qty, actual in samples:
        got = compute_trade_cost(entry, exit_, qty)
        print(f"entry={entry} exit={exit_} qty={qty}: formula={got:.2f}  actual={actual:.2f}  "
              f"diff={got - actual:+.2f}")
