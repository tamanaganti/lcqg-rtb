# Latency-Constrained Quality Gating for Generative Ad Creatives in Real-Time Bidding Systems

Simulation code for the paper:
> **Latency-Constrained Quality Gating for Generative Ad Creatives in Real-Time Bidding Systems**

---

## Overview

LCQG is a lightweight framework that evaluates generative ad creative quality **inside** the RTB bid-processing pipeline — before impressions are served — while staying within the 10 ms bidder-side latency budget.

It combines three signals into a composite **Creative Robustness Index (CRI)**:
- **PRS** — Perceptual Relevance Score
- **BCI** — Brand Coherence Index
- **ACS** — Artifact Contamination Score

All experiments run on a fully reproducible **synthetic** RTB environment (50,000 creatives). No real campaign data, CTR, or revenue figures are used.

---

## Key Results

| Metric | Value |
|---|---|
| Low-quality exposure reduction | 81.3% [95% CI: 80.5, 82.1] |
| Engagement proxy lift | +14.1% [95% CI: +13.8, +14.4] |
| LCQG Expert F1 | 0.650 [95% CI: 0.643, 0.657] |
| Full LCQG P95 latency | **2.46 ms** (budget: 10 ms) |
| Context false-accept reduction | 18.6 pp (pre-deployment mode) |
| Threshold stability under shift | stable (τ* = 0.50) |

---

## Requirements

- Python 3.10+
- ~4 GB RAM
- ~5 min runtime (final mode, 50k creatives)

```bash
pip install -r requirements.txt
```

---

## Usage

### Full reproduction from scratch

```bash
python main.py --mode final
```

Generates 50k synthetic creatives, trains all models, runs all 6 experiments, saves results to `results/`. All random seeds are fixed — output is fully deterministic.

### Quick verification (skip data generation and training)

```bash
python main.py --mode final --skip-data --skip-models
```

Uses the pre-computed data and models already in `results/`. Re-runs all experiments and reproduces the paper's numbers exactly.

### Fast smoke test (5k creatives, ~1 min)

```bash
python main.py --mode dev
```

Numbers differ from the paper (5k vs 50k) but the full pipeline runs identically.

---

## Project Structure

```
main.py                          ← entry point
requirements.txt
autoeval_rtb/
├── data/
│   └── synthetic_creative_generator.py   ← 50k synthetic RTB creatives
├── scoring/
│   ├── prs.py                            ← Perceptual Relevance Score
│   ├── bci.py                            ← Brand Coherence Index
│   ├── acs.py                            ← Artifact Contamination Score
│   └── cri.py                            ← Creative Robustness Index + gating
├── gating/
│   ├── lcqg.py                           ← main LCQG pipeline
│   └── latency_profiler.py               ← 10k-iteration latency benchmark
├── baselines/
│   ├── no_gate.py
│   ├── offline_gate.py
│   ├── clip_gate.py
│   └── supervised_gate.py
├── experiments/
│   ├── exp1_latency.py                   ← table1_latency.csv
│   ├── exp2_cri_sweep.py                 ← threshold optimization + exp2_test_results.csv
│   ├── exp3_ablation.py                  ← table2_ablation.csv
│   ├── exp4_baselines.py                 ← table3_baselines.csv
│   ├── exp5_context_robustness.py        ← table4_context.csv
│   └── exp6_distribution_shift.py        ← table5_shift.csv
└── utils/
    ├── metrics.py
    ├── bootstrap.py                      ← 95% CIs via bootstrap resampling
    └── plots.py
results/                         ← pre-computed outputs (CSVs, PNGs, models)
```

---

## Experiments

| # | Script | Output file | What it measures |
|---|---|---|---|
| E1 | `exp1_latency.py` | `table1_latency.csv` | P50/P95/P99 latency per component |
| E2 | `exp2_cri_sweep.py` | `exp2_test_results.csv` | Threshold sweep + constrained optimum |
| E3 | `exp3_ablation.py` | `table2_ablation.csv` | Signal contribution (PRS / BCI / ACS) |
| E4 | `exp4_baselines.py` | `table3_baselines.csv` | LCQG vs no-gate / offline / CLIP / supervised |
| E5 | `exp5_context_robustness.py` | `table4_context.csv` | Context-aware pre-deployment screening |
| E6 | `exp6_distribution_shift.py` | `table5_shift.csv` | Artifact + context + brand distribution shift |

---

## Pre-computed Results

The `results/` directory contains the exact outputs used in the paper:

- `table0_data_sanity.csv` — dataset statistics
- `table1_latency.csv` through `table5_shift.csv` — all paper tables
- `fig2_cri_sweep.png` through `fig6_shift.png` — all figures
- `optimal_threshold.json` — τ* = 0.50
- `cri_calibrated_weights.json` — Ridge-learned CRI weights
- `acs_model.pkl` — trained IsolationForest + scalers
- `supervised_gate.pkl` — trained logistic regression baseline
- `synthetic_creatives.parquet` — full 50k dataset
- `data_splits.pkl` — deterministic train/dev/test indices

---

## Reproducibility Notes

- Global seed: `numpy.random.seed(42)` set before all imports in `main.py`
- Data splits: stratified by quality tier, `random_state=42`
- Bootstrap CIs: `n=1000`, `seed=42`
- Latency: measured with `time.perf_counter()` over 10,000 iterations
- Never use `--skip-data` across different `--mode` settings (dev vs final) — this reuses stale data from the wrong run size

---

## Disclaimer

This is a **simulation only**. All creatives, embeddings, quality labels, and engagement metrics are synthetic. No real advertising data, click-through rates, or revenue figures are used or implied.
