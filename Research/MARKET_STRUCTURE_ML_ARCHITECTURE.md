# Market Structure Engine + Real ML Project — Architecture (Draft for Review)

Status: **Revised after external review.** Incorporates feedback that reordered the plan and added two concrete pieces (feature importance report, per-prediction explainability). Structure engine build is now in progress; ML remains gated on structure proving useful alone, per Section 7.

This design is built directly on what's already been proven and disproven in this project — it doesn't start from a blank page.

---

## 1. Why these two, and why together

Three earlier attempts to add a "confirmation layer" on top of the two live strategies all failed honestly-tested validation:

| Attempt | Result |
|---|---|
| step6 "AI" Strategy Selector | Hand-tuned constants, backtested with/without — no improvement |
| Market DNA trend filter | Looked strong on 6 months, failed completely on 18 months out-of-sample |
| OI gate (futures OI, two strategies + Concepts 2&4) | Hurt performance in both real tests |

The common failure mode: a **fixed rule** ("score ≥ 70," "regime must match direction") applied to a **hand-picked threshold**, tested once, not re-validated out-of-sample before being trusted.

Market Structure and real ML are different in two specific ways that address this:

1. **Market structure uses only price data** — no dependency on option-chain OI (which can't be backtested at all, only collected going forward) or futures OI (only ~90-120 days of real history exists, ever). Structure can be tested on the full 2-year dataset already fetchable, the same size sample the actual strategies were validated on.
2. **A real ML model learns its own thresholds from data** instead of a human guessing "70" or "agrees with direction." That doesn't guarantee it works — it still has to pass the same walk-forward out-of-sample bar the DNA filter failed — but it removes the specific failure mode of an arbitrary hand-picked cutoff.

---

## 2. Market Structure Engine — architecture

**Goal:** describe the price structure around each bar objectively — trend state, key levels, and structural breaks — as a defined, testable feature set. Not a black box, not "AI," just clearly defined price geometry.

### 2.1 Core components

**Swing point detection.** A bar is a swing high if its high is greater than the highs of N bars on each side (fractal method); swing low is the mirror. N is a tunable parameter (start with N=3 to 5 on the 2-min bars, needs its own sensitivity check, not just one arbitrary pick).

**Trend structure state.** Read off the sequence of recent swing highs/lows:
- Higher-High + Higher-Low → **Uptrend structure**
- Lower-High + Lower-Low → **Downtrend structure**
- Mixed / overlapping → **Range structure**

**Break of Structure (BOS) / Change of Character (CHoCH).** Standard price-action concepts:
- BOS: price breaks past the most recent swing high in an uptrend (or swing low in a downtrend) → trend continuation signal.
- CHoCH: price breaks the *opposite* direction from the current structure (e.g., breaks below the last swing low while structure was still reading "uptrend") → possible reversal signal.

**Key levels.** Reuses what's already built and correct in this project (`build_previous_day_pivot.py` — already verified no-lookahead): previous day high/low/close, today's opening-range high/low, plus the swing highs/lows from above. Distance-to-nearest-level becomes a feature (e.g., "currently within 0.1% of yesterday's high").

**Multi-timeframe context (optional, phase 2).** Compute structure on a higher timeframe (e.g., 15-min bars) as context for the 2-min entries — same idea `oi_gated_concepts_backtest.py`'s `attach_htf_trend` already tried for trend, but built independently and tested on its own merit, not assumed to work because a similarly-named thing existed before.

### 2.2 Data requirement

Only OHLC price data — already have 2+ years cached/fetchable via `om.fetch_index_chunked`. No OI dependency at all. This is the engine's single biggest practical advantage over anything OI-based.

### 2.3 Output

A per-bar structured state (not a single "score" like DNA — several defined fields):
```
{trend_structure: "Uptrend"/"Downtrend"/"Range",
 last_bos_direction: "UP"/"DOWN"/None,
 last_choch: True/False,
 distance_to_prev_day_high_pct: float,
 distance_to_prev_day_low_pct: float,
 distance_to_nearest_swing_pct: float}
```

### 2.4 Swing-length parameter — fixed, not swept

N is set to a conventional value (5 bars each side) chosen before testing, not tuned by trying several values and keeping whichever looks best. Sweeping N until performance improves is the same curve-fitting mechanism that already produced a false positive with the DNA filter (which passed on 6 months and failed on 18). If N=5 doesn't show anything, that's a real result, not a reason to try N=7.

### 2.5 Mandatory validation before this is trusted for anything

Same standard already applied to DNA and OI: does trading *only when structure agrees with the trade's direction* actually improve open_momentum/six_concepts on a real backtest, tested on one window and then re-checked on a separate, later window it wasn't tuned on? If it fails that test the same way DNA did, it gets reported honestly as a failure too — this isn't guaranteed to work just because it's a different idea.

---

## 3. Real ML Project — architecture

**Goal:** a genuinely trained, validated model — not a rule-based scorecard mislabeled as AI (the mistake in `step6` and `Strategy_Selector_AI`).

### 3.1 What it should predict (scope matters — pick ONE, not "everything")

Recommended target: **meta-labeling** — a well-established real quant technique. The base strategies (open_momentum, six_concepts) already decide *when* to enter; the ML model's only job is to predict, at the moment of entry, whether *this specific signal* is likely to be a winner or a loser, using context features (structure, DNA, OI if available). If the model says "low confidence," the trade can be skipped or sized down — the base strategy's entry logic is never replaced, only filtered, and it can always fall back to "trade everything" if the model is unavailable or unsure.

This is deliberately narrower than trying to build a model that invents its own entries from scratch — that's a much bigger, riskier project, and this project's own history (Strategy_Selector_AI, ORB_ML_OPTIMIZER) shows ambitious ML scope without real labels is exactly how you end up with fabricated-looking output.

### 3.2 Labels

Real, not synthetic (the specific mistake in `Strategy_Selector_AI`, which used `np.random.normal()` for its labels). Labels come directly from the actual backtested outcome of each real trade already generated by `om.run_for_combo` / `sc.run_module`: win = 1, loss = 0 (or regression target = actual rupee P&L, if enough data to support that).

### 3.3 Features

- Market Structure Engine output (Section 2), only if it independently passes its own validation first.
- Price-only Market DNA (trend_score, momentum_score) — already proven leak-free, even though it failed as a fixed-threshold rule, it may still carry real information a model can learn a better threshold from. Worth including as a raw feature and letting the model decide, rather than assuming "already failed once, therefore useless."
- OI features, if available — include cautiously; explicitly test the model with and without OI to see honestly whether it adds anything, rather than assuming yes or no in advance.
- Time-of-day, day-of-week (cheap, already used informally elsewhere in the project).

### 3.4 Model choice

Start simple: logistic regression or a shallow decision tree first. Something you can inspect and explain, not a black box, consistent with how this project already prefers transparent heuristics over opaque ones. Only escalate to something like gradient boosting if the simple version shows a real, validated signal worth refining.

### 3.5 Validation discipline (non-negotiable, built into the architecture itself)

- **Walk-forward only.** Train on an expanding window of earlier trades, test only on strictly later trades the model never saw. No shuffled train/test splits — that was the exact leakage bug already found and fixed in `Market_Regime_Classifier` and `ORB_ML_OPTIMIZER`.
- **One true holdout.** Reserve a final slice of the most recent data that is touched exactly once, at the very end, after all tuning is done elsewhere. This is what protects against the "looked great on 6 months, failed on 18 months" pattern happening again.
- **Evaluate on trading metrics, not just accuracy.** Same SQN / win-rate / max-drawdown comparison already used for every other test in this project (baseline vs. with-model), not classifier accuracy in isolation.
- **Small-sample honesty.** The current combined trade log is roughly 1,000 trades over 2 years. That's a small-data problem for ML — needs strong regularization, realistic expectations, and probably more trades (either a longer history or more strategies contributing labeled examples) before a complex model is justified.

### 3.6 Fail-safe behavior

If the model is unavailable, unsure, or hasn't been validated for the current market conditions, the system must fall back to the base strategy trading normally — never fail silently into "no trades" or an unvalidated guess. Mirrors the existing project-wide rule of never fabricating a number when live data isn't available.

### 3.7 Feature Importance Report (added per external review)

Generated after every training run, not just once: which features actually moved the model's predictions (Market Structure, DNA, ATR, OI, gap, time-of-day, etc.), ranked by contribution. This is what tells you honestly whether Structure or OI or anything else is pulling weight, instead of assuming. Cheap to produce from a simple model (logistic regression coefficients, or a shallow tree's built-in importances) — another reason to keep the model simple.

### 3.8 Per-prediction explainability (added per external review)

Every prediction should be able to state which inputs drove it, e.g. "Trade: yes — uptrend structure, DNA momentum positive, ATR normal, small gap, similar historical setups won 74% of the time" — not just a bare confidence number. This stays honest and cheap as long as the model stays simple (Section 3.4); it gets harder to trust the moment the model gets more complex, which is one more reason not to escalate past logistic regression/shallow trees without a clear justification.

---

## 4. How the pieces connect (proposed layering)

```
Data layer (existing)
  step1_config_and_connector.py -- unchanged

Feature layer (new)
  Market Structure Engine  -- new, price-only, testable on full 2yr history
  Market DNA (price-only)  -- existing, reused as a raw feature, not a fixed rule
  OI features               -- existing step2/step3 engine, included cautiously

Meta-labeling ML layer (new)
  Trained on REAL trade outcomes from om_trades / sc_trades
  Walk-forward validated, one true holdout at the end
  Outputs: confidence score per signal, not a new entry signal

Decision layer (new, thin)
  base strategy signal + ML confidence -> take trade / skip / fall back to base-only

Execution layer (existing pattern, reused)
  Same separate-DB, no-interference pattern as step10_live_paper_best2.py

Monitoring layer (existing pattern, reused)
  Same dashboard pattern as step11_dashboard_best2.py, extended to show ML confidence
```

Nothing in this layering touches `step9_live_engine.py`, `step10_live_paper_best2.py`, or their databases — same isolation principle already used for every add-on in this project.

---

## 5. Honest risks to weigh before building

- **Curve-fitting risk from a small sample.** ~1,000 trades is not a lot for ML. Real discipline (holdout set touched once) is what prevents this becoming DNA-filter-again.
- **Structure detection has its own hidden parameters** (swing lookback N, BOS confirmation rules) that could themselves be overfit — needs its own honest out-of-sample check, independent of whether the ML layer on top of it works.
- **OI remains a proven-weak input class so far** (two failed tests) — worth including as a feature to test fairly, but going in with the expectation it may add nothing, based on the actual evidence so far.
- **This is a real multi-week effort**, not a quick add-on — building the structure engine properly, generating enough labeled trade history, and doing walk-forward ML validation correctly all take real time. Worth setting that expectation before starting.

---

## 6. What's NOT in scope for this design (kept simple deliberately)

- No new entry strategies — this only re-scores signals from the two you already trust.
- No liquidity-sweep / advanced smart-money-concepts layers yet — flagged as a possible phase 2, not needed to get a first honest test running.
- No live option-chain-based ML features yet — those can only be collected prospectively (no historical option-chain data exists to backtest with), so they're a future addition once enough live history accumulates, not part of the first build.

## 7. Revised roadmap (per external review — structure-first gate)

The original draft presented Structure and ML as one architecture. The stronger sequencing: treat them as two separate, honest decisions, with ML strictly gated on the first one succeeding. ML can only re-weight signal that already exists — if Structure alone shows nothing, no amount of meta-labeling manufactures an edge from noise.

1. Build the Market Structure Engine (Section 2) — swing points (fixed N, not swept), trend structure, BOS/CHoCH, key levels.
2. **Gate check:** does Structure *alone*, used the same simple way DNA/OI were tested (trade only when structure agrees with direction), improve open_momentum/six_concepts on real data?
   - If **no** — stop here, report it honestly, do not build the ML layer. Matches how DNA and OI were reported when they failed.
   - If **yes** — re-check on a separate, later out-of-sample window before trusting it (same discipline that caught the DNA filter's false positive).
3. Only if Structure passes both checks: build the meta-labeling ML layer (Section 3) on top of it, with walk-forward validation, one true holdout, feature importance report, and per-prediction explainability.
4. Paper trade the validated result (same isolated-DB pattern as `step10_live_paper_best2.py`) before ever considering live capital.
