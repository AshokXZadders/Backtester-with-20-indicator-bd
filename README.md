# wto.py — Multi-Indicator Swing Backtester

Terminal-based, multi-stage backtesting engine for swing trading on daily
bars. Screens 20 individual indicators, searches indicator combinations,
Bayesian-tunes the winning combo's parameters, then stress-tests it with
bootstrap Monte Carlo. Optionally sizes positions off a GARCH volatility
forecast instead of ATR.

**Always run this through the CLI flags below — never edit the constants
at the top of the file (`MIN_TRADES_HARD`, `RR_GRID`, indicator grids,
etc.) directly to change behavior.** Every parameter that affects a run
is exposed as a flag so results stay reproducible and comparable across
runs (the whole point of `--param`, `--sizes`, `--signal-mode`, and the
sizing/GARCH flags below). If you find yourself wanting to hand-edit a
value in the script to try something, check first whether a flag already
covers it — most do.

---

## 1. What it actually does (pipeline order)

Running with `--pipeline` executes four stages in sequence:

1. **Stage 1 — Solo Indicator Screen**
   Every one of the 20 indicators is tested alone (first grid value each)
   with a lenient scorer (return doesn't have to be positive). Anything
   clearing `MIN_TRADES_HARD` trades survives to Stage 2. This is a
   coarse filter, not a real evaluation.

2. **Stage 2 — Combo Search**
   Builds every 2- and 3-indicator combination (configurable via
   `--sizes`) from the top-N survivors of Stage 1 (`--top-n`), then runs
   a full `bt.optimize()` grid search per combo across worker processes.
   Best combo by `score_stats()` wins.

3. **Stage 3 — Bayesian Fine-Tune**
   Uses `scikit-optimize` (`gp_minimize`) to fine-tune the winning
   combo's indicator parameters plus `rr_ratio`, `atr_sl_mult`, and
   `max_hold_bars` around the Stage 2 best point. Falls back to the raw
   Stage 2 params (with a clear warning) if `scikit-optimize` isn't
   installed.

4. **Stage 4 — Monte Carlo Validation**
   Bootstrap-resamples the final strategy's per-trade returns (with
   replacement) 1,000 times to build a distribution of possible
   outcomes, flags whether the actual backtest was lucky/robust/
   conservative, and reports ruin probability (>30% loss).

If you chose the "6+3 walk-forward" data split at startup, a final
**out-of-sample test** runs the tuned combo/params (no re-optimization)
against the untouched 3-year test window.

## 2. Fixes baked into this version (context, not action items)

- **FIX-1**: Entries fill at next-bar open (no lookahead).
- **FIX-2**: Compounded Sharpe uses √252 annualization.
- **FIX-3**: Monte Carlo uses bootstrap resampling with replacement.
- **FIX-4**: SL/TP stored as *distances* and re-anchored to the actual
  fill price at execution (not the signal-bar close), so gaps don't
  distort risk.
- **FIX-5**: Pending-order state is instance-level only — no class-level
  leakage across optimizer worker runs.
- **SIZING-1**: `sizing_mode` explicitly controls both stop distance and
  position size (`legacy` = old full-equity sizing behavior kept for
  backward compatibility; `atr` = real `risk_pct`-based sizing off ATR;
  `garch` = risk_pct-based sizing off a GARCH volatility forecast).

## 3. Data requirements

CSV with columns `open, high, low, close, volume` (case-insensitive),
a date/time index in column 0. Rows with non-positive values are dropped.

At startup you'll be prompted interactively to choose a data range:

| Choice | Mode | Behavior |
|---|---|---|
| 1 | Full data | Uses the entire file |
| 2 | Last 2 years | + 200-bar warmup before the trading window |
| 3 | 6+3 walk-forward | Optimize on 6y (+200-bar warmup), validate untouched on the final 3y |

---

## 4. CLI Reference

Run everything as `python wto.py [flags]`. Below is every flag defined
in `main()`.

### Core / data

| Flag | Default | Meaning |
|---|---|---|
| `--file PATH` | `DLF-EQ.csv` | Input OHLCV CSV |
| `--cash N` | `100000` | Starting capital |
| `--commission N` | `0.07` | Commission, in **percent** (e.g. `0.07` = 0.07%) |
| `--start-date YYYY-MM-DD` | none | Filter data to on/after this date, applied before the range prompt |
| `--end-date YYYY-MM-DD` | none | Filter data to on/before this date |

### Run mode (pick one)

| Flag | Meaning |
|---|---|
| `--pipeline` | Run the full 4-stage pipeline (Stage 1→2→3→4) |
| `--single-run --combo "ind1,ind2"` | Run one specific indicator combination once, no search |
| *(neither flag)* | Default: brute-force full combinatorial search over **all** indicators at the given `--sizes`, no staged screening |

### Single-run specific

| Flag | Meaning |
|---|---|
| `--combo "use_rsi,use_macd"` | Comma-separated indicator flag names to enable (required with `--single-run`) |
| `--param key=value` | Override any strategy parameter; repeatable, e.g. `--param rsi_period=21 --param rr_ratio=2.5` |

### Search / pipeline tuning

| Flag | Default | Meaning |
|---|---|---|
| `--sizes N [N ...]` | `2 3` | Combo sizes to test, e.g. `--sizes 2` for pairs only, `--sizes 2 3 4` |
| `--top-n N` | `14` | How many Stage-1 survivors carry into Stage 2 combo search |
| `--signal-mode {0,1,2}` | `2` | Vote aggregation: `0`=any vote, `1`=unanimous, `2`=majority |
| `--min-trades N` | `50` | Sets `MIN_TRADES_SOFT`; `MIN_TRADES_HARD` derives as `max(20, N-30)` |
| `--n-jobs N` | `-1` | Worker processes for combo search; `-1` = `cpu_count()-1` |
| `--bayes-calls N` | `150` | Number of Stage 3 Bayesian optimization evaluations |
| `--reinvest-rate N` | `0.5` | Fraction of each winning trade's PnL reinvested when computing compounded stats |

### Sizing / GARCH (Option 4)

| Flag | Default | Meaning |
|---|---|---|
| `--sizing-mode {legacy,atr,garch}` | `legacy` | `legacy`=old full-equity sizing; `atr`=real risk_pct sizing off ATR; `garch`=risk_pct sizing off a GARCH forecast |
| `--garch-file PATH` | none | **Required** when `--sizing-mode garch`. Output CSV from `garch_risk.py` with a `Date` index and `GarchVol` column |
| `--garch-sl-mult N` | `1.5` | Stop-loss multiple applied to `GarchVol` (mirrors `atr_sl_mult`) |
| `--dual-compare` | off | Only meaningful with `--sizing-mode garch --pipeline`: after the GARCH run, re-runs the *same* final combo/params under `sizing_mode='atr'` and prints a side-by-side comparison table |

### Diagnostics

| Flag | Meaning |
|---|---|
| `--fib-debug` | Prints swing-high/low counts for the loaded data and exits (no backtest run) |

---

## 5. Every command, spelled out

Run these from the terminal in the project directory — **do not** call
functions from this file directly in a Python session or notebook;
the checkpoint/recovery system, worker pool initialization, and
interactive data-range prompt all assume a normal CLI invocation.

```bash
# 1. Full pipeline, defaults everywhere (interactive prompt for data range)
python wto.py --file DLF-EQ.csv --pipeline

# 2. Full pipeline, only pairs (no triples), tighter trade floor
python wto.py --file DLF-EQ.csv --pipeline --sizes 2 --min-trades 40

# 3. Full pipeline with more Stage-1 survivors and more Bayesian calls
python wto.py --file DLF-EQ.csv --pipeline --top-n 18 --bayes-calls 300

# 4. Full pipeline, custom cash/commission, unanimous signal mode
python wto.py --file DLF-EQ.csv --pipeline --cash 500000 --commission 0.05 --signal-mode 1

# 5. Full pipeline restricted to a date window
python wto.py --file DLF-EQ.csv --pipeline --start-date 2018-01-01 --end-date 2024-12-31

# 6. Single run — one specific combo, default params
python wto.py --file DLF-EQ.csv --single-run --combo "use_rsi,use_macd,use_adx"

# 7. Single run — combo with parameter overrides
python wto.py --file DLF-EQ.csv --single-run --combo "use_rsi,use_ema" \
    --param rsi_period=21 --param rsi_lower=25 --param rr_ratio=2.5 \
    --param atr_sl_mult=2.0 --param max_hold_bars=20

# 8. Default mode — brute-force ALL indicator combos (no staged screen)
python wto.py --file DLF-EQ.csv --sizes 2 3 --n-jobs 6

# 9. Diagnostic only — check swing detection counts, no backtest
python wto.py --file DLF-EQ.csv --fib-debug

# 10. Real ATR-based risk sizing (SIZING-1), instead of legacy full-equity
python wto.py --file DLF-EQ.csv --pipeline --sizing-mode atr

# 11. GARCH-based position sizing (Option 4) — requires a precomputed
#     garch_risk.py output CSV first
python wto.py --file DLF-EQ.csv --pipeline \
    --sizing-mode garch --garch-file garch_risk_output.csv --garch-sl-mult 1.5

# 12. GARCH sizing + automatic side-by-side comparison vs ATR sizing
#     on the exact same tuned combo/params
python wto.py --file DLF-EQ.csv --pipeline \
    --sizing-mode garch --garch-file garch_risk_output.csv --dual-compare

# 13. Walk-forward pipeline (choose option 3 at the data-range prompt),
#     more workers for the combo search stage
python wto.py --file DLF-EQ.csv --pipeline --n-jobs -1 --top-n 12
```

---

## 6. Checkpointing & interruption

- Every stage writes a checkpoint (`datacan5_checkpoint.json`, next to
  the script) after each unit of work — a Stage 1 indicator, a Stage 2
  combo — so you can `Ctrl+C` at any point without losing progress
  visibility.
- On the next run, if a checkpoint file exists, its summary (stage,
  progress, best combo/score, top results) prints automatically before
  the run starts.
- The checkpoint is only deleted on a **successful** full completion of
  `--pipeline` or the default full-search mode. A crashed or interrupted
  run leaves it in place intentionally, for inspection.
- The checkpoint is informational only — it is not currently
  auto-resumed into a partially-completed run; treat it as a progress
  log to read, not a `--resume` flag.

## 7. Reading the output

- **Score** (`score_stats()`) is not raw return — it blends Sharpe
  (30%), return (30%), profit factor (15%), peak/final equity gain
  (10% each), trade-count bonus (5%), plus a small compounded-stats
  bonus. Anything under `MIN_TRADES_HARD` trades scores `-1000` outright.
- **NORMAL vs COMPOUNDED** columns in the final results: NORMAL is
  `backtesting.py`'s native stats; COMPOUNDED simulates partial
  reinvestment of winning trades (`--reinvest-rate`) on top of that,
  which is closer to how compounding would behave in live trading.
- Monte Carlo's P5/P25/P50/P75/P95 bands tell you whether the single
  historical backtest result was a favorable draw (`LUCKY`, above P75),
  an unfavorable one (`CONSERVATIVE`, below P25), or typical (`ROBUST`).

## 8. Required packages

```bash
pip install pandas numpy scipy talib-binary backtesting scikit-optimize
```
(`scikit-optimize` is optional — Stage 3 degrades gracefully without it,
using Stage 2's best params unrefined, but install it if you want actual
Bayesian fine-tuning.)
