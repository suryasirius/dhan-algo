"""
One-time verification that the fix to dhan_live_bot.py's resolve_itm_option()
actually works now. Does NOT place any order -- this only calls the lookup
function to prove it can find a real NIFTY option contract. Safe to run
any time, including with DRY_RUN=True or False, since no order call happens.
"""

from dhan_live_bot import resolve_itm_option

# Approximate spot price just to pick a strike near-the-money -- doesn't need
# to be exact, only close enough to find some ITM strikes to test with.
TEST_SPOT_PRICE = 25000

print("Testing resolve_itm_option('CE', ...) ...")
call_contract = resolve_itm_option("CE", TEST_SPOT_PRICE)
print("CALL contract found:")
for k, v in call_contract.items():
    print(f"  {k}: {v}")

print("\nTesting resolve_itm_option('PE', ...) ...")
put_contract = resolve_itm_option("PE", TEST_SPOT_PRICE)
print("PUT contract found:")
for k, v in put_contract.items():
    print(f"  {k}: {v}")

print("\nIf both printed a real security_id, trading_symbol, strike, and expiry "
      "above with no error, the fix worked.")
