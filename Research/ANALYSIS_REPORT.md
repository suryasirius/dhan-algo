# AI Quant Platform — Deep Review & Rating

Scope: full `Research/` tree (step1–step9, Market_DNA, Market_Regime_Classifier,
OI_Intelligence_Engine, Strategy_Selector_AI, AI_Trade_Decision_Engine,
ORB_ML_OPTIMIZER, Previous_Day_Pivot, Live_Paper_Trading) plus a sample of the
150+ top-level backtest scripts in `C:\dhan algo`.

## Overall rating: 5/10

A genuinely ambitious architecture with real live infrastructure now wired up
(today's work), and some individual scripts with excellent backtest discipline —
but the modules actually branded "AI"/"ML" are not trustworthy as implemented, and
security/ops hygiene is weak for something heading toward live capital. Breakdown:

| Area | Rating | Why |
|---|---|---|
| Live engineering / pipeline architecture | 7/10 | Now genuinely live end-to-end (step9). Clean separation of concerns across step1–7. Some brittleness (hardcoded paths, duplicate config). |
| ML / "AI" methodology | 3/10 | Two of the three ML-branded modules have serious leakage or fabricated outputs (see below). This is the weakest part of the project relative to its own claims. |
| Rule-based backtest rigor | 7/10 | Wide quality spread — a few scripts (`orb_upside_breakout_csv.py`, `candle_filter_top3_6month.py`, ORB_ML_OPTIMIZER's cost model) are excellent; most of the 150+ scripts have no walk-forward validation. |
| Security / ops hygiene | 2/10 | Live Dhan token + client ID hardcoded in plaintext in multiple files, no `.gitignore`, no version control found anywhere. |
| Test coverage | 0/10 | Zero automated tests found anywhere in the project. |

---

## Top 10 issues, ranked by severity

1. **Live Dhan access token and client ID are hardcoded in plaintext** in `dhan_config.py`, `nifty_paper_trade_new_strategy.py`, and `Live_Paper_Trading/config.py`, plus stored in plaintext in `logs/dhan_token.txt`. `dhan_config.py`'s own docstring claims "never shared, printed, or read by anyone but you" — that's not true, it's plaintext source. This is a real financial-account credential exposure risk, independent of anything else in this report.
2. **`Strategy_Selector_AI`'s "best strategy" and all its win-rate/profit-factor/drawdown numbers are fabricated** — hand-built formulas plus `np.random.normal(...)` noise in `feature_matrix.py`, not derived from any real backtest. The module *does* train a real RF/XGB/LGBM/CatBoost ensemble (`model_trainer.py`), but `run.py` never uses it — `selector.py` reads straight from the synthetic columns instead. It produces confident-sounding numbers that are noise dressed as ML output.
3. **`Market_Regime_Classifier` has target leakage**: the `Market_Type` label is a deterministic function of the same values used as model features, features are normalized using full-dataset (not rolling) statistics, and a "class balance" patch can leak a test-set row directly into training. Any accuracy this module reports is meaningless.
4. **No walk-forward / out-of-sample validation anywhere in the project**, despite 150+ strategy variants and multiple grid searches all tested against the same static historical dataset. High multiple-comparisons risk — whichever variant looks best is partly luck.
5. **`ORB_ML_OPTIMIZER`'s grid search results are computed and then discarded** — the ML training and Monte Carlo step run on one hardcoded parameter set with no stated justification. Its ML training also uses a shuffled train/test split on time-ordered trade data (temporal leakage), not a walk-forward split.
6. **Zero automated tests anywhere in the repo.** For a system this size that's meant to eventually route real capital, there's no regression safety net — every change is verified by manually rerunning a script and eyeballing output.
7. **No version control detected anywhere** (`dhan algo` and `Research` both have no `.git`, no `.gitignore`). Beyond the token issue, this means no change history, no rollback, no diff review as the project evolves.
8. **Dead/duplicate code**: `dhan_integration_layer.py` vs `dhan_integration_master.py` are near-duplicate unused stubs; `AI_Trade_Decision_Engine` generates its numbers from `np.random` entirely; the trained ensemble in Strategy_Selector_AI is computed but unused. Risk that a stale/fake module gets wired into the live engine by mistake later.
9. **16 files hardcode absolute `C:\dhan algo\...` paths** in source rather than a relative root or env var — brittle to any move, clone, or second machine.
10. **`logging.basicConfig()` is called independently in step1/step7/step9** — only the first call in a process actually takes effect, so log formatting becomes nondeterministic depending on import order once steps are combined into a live pipeline (as step9 now does).

## What's genuinely good

- The core `step1`→`step7` pipeline (config, OI features, OI classifier, confidence/risk, strategy selector, paper trading) is clean, has consistent error handling (no bare `except: pass` anywhere), and the OI feature engineering correctly uses lagged/rolling computation, not future data.
- `orb_upside_breakout_csv.py` and `candle_filter_top3_6month.py` (and the ORB_ML_OPTIMIZER cost model) are honestly and carefully built: explicit no-lookahead fills on the *next* bar's open, conservative stop-before-trail intrabar ordering, disclosed synthetic option-premium proxy assumptions (labeled as such, not hidden), and a realistic multi-component cost model (STT, GST, exchange charges, stamp duty, slippage, lot-size schedule changes over time).
- `Previous_Day_Pivot`'s core pivot logic correctly shifts by one day — no lookahead there.
- Today's build gives the platform real live infrastructure for the first time: genuine Dhan candle + option-chain fetch, a transparent (non-random) live DNA/regime scoring heuristic, and a paper-trading loop that only acts on real data or explicitly waits.

## Recommended next steps, in priority order

1. **Rotate your Dhan token immediately** and move every hardcoded credential (in `dhan_config.py`, `nifty_paper_trade_new_strategy.py`, `Live_Paper_Trading/config.py`) to read from `token_manager.py` instead, the way `step1_config_and_connector.py` now does. Add a `.gitignore` covering `logs/`, `*.txt` secrets, and any `dhan_config.py`-style file before this project ever touches git.
2. **Stop trusting `Strategy_Selector_AI` and `Market_Regime_Classifier` outputs as-is.** Either fix the leakage (rolling/expanding normalization, a label that isn't a function of the input features, a real walk-forward split) or don't feed their numbers into live decisions until fixed. Right now `step9_live_engine.py` deliberately does NOT use either module — it computes its own transparent heuristic DNA/regime instead — which was the right call given what this review found.
3. **Pick 3–5 of your best-performing backtest scripts and add one real out-of-sample test**: hold out the most recent 1–2 months, tune only on earlier data, then check performance on the held-out period once. That single check would tell you more about real edge than the other 145 scripts combined.
4. **Add a bare-minimum test suite**: even 10-15 unit tests on `OIBuilder`, `OIClassifier`, `ConfidenceAndRiskEngine`, and `StrategySelectorAI` (pure functions, easy to test) would catch regressions before they hit the live engine.
5. **Delete or clearly quarantine dead code** (`dhan_integration_layer.py`/`dhan_integration_master.py`, `AI_Trade_Decision_Engine`) so nothing gets accidentally wired into `step9_live_engine.py` later.
