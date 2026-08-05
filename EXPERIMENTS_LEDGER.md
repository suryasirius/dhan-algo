# Experiments Ledger — Sirius Project

Every filter, engine, and model tested on the combined open_momentum + six_concepts
strategy ("Sirius"), with real results and the honest verdict. Update this file
after every new test — this is the project's memory, so nothing gets silently
re-tried or forgotten.

---

## Baseline (no filter)

**Sirius, 2-year backtest:** 973 trades, 67.4% win rate, ₹225,862 net profit, max DD ₹4,853 (19.4%), SQN 11.59.
**Sirius, 5-year backtest:** 2,429 trades, 58.7% win rate, ₹296,066 net profit, max DD ₹4,853 (19.4%), SQN 12.67. Year-by-year uneven: 2023 weak (SQN 2.57), 2025-2026 strong (SQN 8.91 / 6.63) — ~70% of 5yr profit from the most recent 18 months.
**Verdict:** the only version proven to work without qualification. This is what's live in `step10_live_paper_best2.py`.

---

## step6 "AI" Strategy Selector

**What it was:** hand-tuned scorecard (+20/+15/+25 constants), not a trained model.
**Test:** backtested top 5 strategies with and without its filtering.
**Result:** no measurable improvement.
**Verdict:** FAILED. Relabeled (not deleted, step9 depends on it) to stop calling it AI.

## Market DNA trend filter

**What it was:** price-only heuristic (spot vs VWAP, % change), "trend agrees with direction" filter.
**Test 1 (6mo, in-sample):** 239→169 trades, win 66.5%→71.6%, net ₹60,097→₹66,220, maxDD ₹4,853→₹1,891, SQN 5.61→7.75. Looked strong.
**Test 2 (split-half, same 6mo):** held up directionally both halves (win rate, drawdown, SQN all improved) but profit was mixed (won big first half, lost slightly second half).
**Test 3 (out-of-sample, 18mo never seen, 2024-08 to 2026-02):** FAILED completely — win 67.7%→66.7%, net ₹163,804→₹91,260, maxDD ₹3,223→₹3,402, SQN 10.08→7.8. Trend-agreement showed literally no relationship to outcome (82.4% win either way for one strategy).
**Verdict:** FAILED. Looked real on 6 months, was noise. Do not use.

## OI gate (futures OI, price-vs-OI quadrant regime)

**Test 1 (Concepts 2&4, real ~90-day OI window, 6 exit variants):** worse ending capital in 4 of 6 variants; win rate barely moved either way.
**Test 2 (open_momentum + six_concepts, real ~70-day OI window):** win 68.8%→64.3%, net ₹21,994→₹8,869, SQN 4.78→2.81. Trades that FAILED the gate did BETTER (90.3% win) than trades that passed it (83.3% win) for open_momentum.
**Verdict:** FAILED, twice, on two different strategy sets. Also structurally limited — real historical OI only exists ~70-120 days back (contract expiry), so it can never be tested on a large sample.

## Market Structure Engine (swing points, trend structure, BOS/CHoCH)

**What it was:** real, price-only, causal swing/trend detection (fixed N=5, not swept). Full architecture in `Research/MARKET_STRUCTURE_ML_ARCHITECTURE.md`.
**Test (6mo real data, "structure agrees with direction" filter, per the roadmap's gate before any ML):** trend-structure filter: 239→63 trades, net ₹60,097→₹17,847, SQN 5.61→3.22. BOS-direction filter: 239→179 trades, net ₹60,097→₹38,779, SQN 5.61→4.37. Both worse than baseline. For open_momentum, "disagreeing" trades still won 76.8% — structure isn't separating good from bad trades.
**Verdict:** FAILED the gate. Per the agreed roadmap, ML layer was correctly NOT built on top of this.

## Sample AI Exit Engine (logistic regression, meta-labeling style)

**What it was:** real trained model (not a heuristic) predicting hold-vs-exit per bar, walk-forward split (train on 85 earlier days, test on 37 later days never seen), open_momentum only.
**Test:** baseline (fixed sl20/trail1): win 86.5%, net ₹14,163, SQN 3.93. AI exit (hard stop + model-decides-profit-taking): win 24.3%, net -₹15,372, SQN -1.39. 27 of 37 trades hit the hard stop.
**Root cause identified:** removing the tight 1pt trailing stop (replaced with a static hard stop + model) let profitable trades give back their whole gain before the model decided to exit. This tested "what happens without trailing protection," not really "is the model's judgment good."
**Verdict:** FAILED, but with a known fixable flaw (keep the trail as a floor, let the model only choose the exit timing within it) — not retried yet.

## Synthetic candle bootstrap test (day-block resampling of real patterns)

**What it was:** 30 independent synthetic 2-month sequences, each built by randomly drawing real trading days (with replacement) from the 122-day pool and stitching them together with real intraday shapes intact, rebased to remove fake overnight gaps. Not a random walk — real candle patterns, reshuffled combinations.
**Test (real 6mo data pool):** 0/30 trials lost money. Win rate range 58.2%-74.4% (mean 64.9%). SQN range 2.02-5.97 (mean 3.42), never near zero. Net profit range ₹14,165-₹33,657, all positive. Max DD contained (₹1,490-₹5,149).
**Verdict:** PASSED. The edge doesn't appear to depend on the one specific historical ordering of the original 6-month backtest. Genuinely reassuring, though still built from the same 122 real days -- not equivalent to real out-of-sample data.

## Position sizing / compounding (not a filter — this changes trade size, not trade selection)

**Fixed 1-lot (original):** 5yr net ₹296,066, maxDD ₹4,853, SQN 12.67.
**Uncapped compounding (lots scale 1:1 with capital growth):** produced ₹32.6 BILLION — mathematically consistent but not real; ends at 65M+ lots/trade, far beyond any real market liquidity or NSE/SEBI position limits. Discarded as not achievable.
**Compounding capped at 20x base lot:** 5yr net ₹6,172,072, maxDD ₹91,748, SQN 14.84. Still reaches 1,300-1,500 lots/trade by year 4-5 — likely still optimistic vs. real slippage at that size.
**+1 lot every 6 months (time-based, not P&L-based):** 5yr net ₹2,588,803, maxDD ₹46,014, SQN 14.35. Ends at ~650 lots/trade. More conservative than capped compounding since it doesn't accelerate right after a hot streak.
**Verdict:** all three are mathematically real (same trade sequence, verified sizing logic) but position size realism decreases as the cap loosens. The fixed 1-lot number is the only one close to fully achievable without slippage concern; the capped versions show upside potential but likely overstate what's actually fillable at scale.

---

## Data infrastructure notes (not a strategy test, but load-bearing facts)

- Dhan intraday history: documented 5 years, confirmed for real via `sirius_5yr_data_probe.py` (all 22 chunks from 2021-08-06 returned real data, no gaps).
- Dhan futures OI history: only ~70-120 days back in practice (contract expiry rolls off the instrument master) — hard ceiling on any OI-based backtest sample size.
- Dhan access token expires roughly daily and must be refreshed via `update_dhan_token.py` (writes to `Research/logs/dhan_token.txt` + `dhan_client_id.txt`, shared by every script in the project).
- No historical option-chain OI dataset exists anywhere — only live/current snapshots. Any option-chain-OI-based feature can only be collected prospectively going forward, never backtested on the past.

---

## Standing pattern across all filter tests

Four different confirmation-style filters (step6, DNA, OI, Market Structure) have now all failed honest out-of-sample testing on this strategy pair. The consistent finding: open_momentum's and six_concepts' entry signals already contain most of the real information available; every add-on tested so far has removed good trades along with bad ones rather than separating them. Any NEW filter proposed should be judged against this track record, not treated as a fresh idea with even odds.
