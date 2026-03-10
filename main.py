import numpy as np
np.random.seed(42)

import argparse
import os
import pickle
import sys

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Parse arguments before importing project modules.
parser = argparse.ArgumentParser(description='LCQG RTB Simulation')
parser.add_argument('--mode',        choices=['dev', 'final'], default='dev',
                    help='dev=5k creatives, final=50k creatives')
parser.add_argument('--skip-data',   action='store_true',
                    help='Skip data generation if parquet exists')
parser.add_argument('--skip-models', action='store_true',
                    help='Skip model training if pkl/json exist')
args = parser.parse_args()

N_CREATIVES = 5_000 if args.mode == 'dev' else 50_000

# Import project modules after the global seed is set.
import pandas as pd

from autoeval_rtb.data.synthetic_creative_generator import generate_data, generate_table0
from autoeval_rtb.scoring.acs import train_acs
from autoeval_rtb.scoring.cri import train_cri_calibrated
from autoeval_rtb.baselines.supervised_gate import train_supervised_gate

from autoeval_rtb.experiments import (
    exp1_latency,
    exp2_cri_sweep,
    exp3_ablation,
    exp4_baselines,
    exp5_context_robustness,
    exp6_distribution_shift,
)

from sklearn.model_selection import train_test_split

# Data generation
parquet_path = os.path.join(RESULTS_DIR, 'synthetic_creatives.parquet')
splits_path  = os.path.join(RESULTS_DIR, 'data_splits.pkl')

if args.skip_data and os.path.exists(parquet_path):
    print(f"Skipping data generation — loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df['embedding']       = df['embedding'].apply(list)
    df['brand_embedding'] = df['brand_embedding'].apply(list)
else:
    print(f"\nGenerating {N_CREATIVES} synthetic creatives (mode={args.mode})...")
    df, brand_embs = generate_data(N_CREATIVES, results_dir=RESULTS_DIR)

generate_table0(df, results_dir=RESULTS_DIR)

# Data splits
if args.skip_data and os.path.exists(splits_path):
    print(f"Loading data splits from {splits_path}")
    with open(splits_path, 'rb') as f:
        split_data = pickle.load(f)
    train_idx = split_data['train_idx']
    dev_idx   = split_data['dev_idx']
    test_idx  = split_data['test_idx']
else:
    print("\nCreating data splits (train=20%, dev=10%, test=70%)...")
    all_idx = np.arange(len(df))
    # Split 70% to test, then split the remainder 2:1 into train and dev.
    traindev_idx, test_idx = train_test_split(
        all_idx, test_size=0.70, random_state=42, stratify=df['quality_tier'].values
    )
    train_idx, dev_idx = train_test_split(
        traindev_idx, test_size=1/3, random_state=42,
        stratify=df['quality_tier'].iloc[traindev_idx].values
    )
    split_data = {'train_idx': train_idx, 'dev_idx': dev_idx, 'test_idx': test_idx}
    with open(splits_path, 'wb') as f:
        pickle.dump(split_data, f)
    print(f"  Train: {len(train_idx)} | Dev: {len(dev_idx)} | Test: {len(test_idx)}")

train_df = df.iloc[train_idx].reset_index(drop=True)
dev_df   = df.iloc[dev_idx].reset_index(drop=True)
test_df  = df.iloc[test_idx].reset_index(drop=True)

# Train ACS on HIGH-tier non-artifact creatives from the train split.
acs_model_path = os.path.join(RESULTS_DIR, 'acs_model.pkl')
if args.skip_models and os.path.exists(acs_model_path):
    print(f"\nSkipping ACS training — model exists at {acs_model_path}")
else:
    print("\nTraining ACS model...")
    high_non_art = train_df[(train_df['quality_tier'] == 2) & (train_df['artifact_flag'] == 0)]
    if len(high_non_art) == 0:
        print("WARNING: No HIGH-tier non-artifact in train_split. Using all HIGH-tier.")
        high_non_art = train_df[train_df['quality_tier'] == 2]
    train_embeddings = np.stack(high_non_art['embedding'].apply(np.asarray).values)
    train_acs(train_embeddings, results_dir=RESULTS_DIR)

# Train calibrated CRI weights
cal_weights_path = os.path.join(RESULTS_DIR, 'cri_calibrated_weights.json')
if args.skip_models and os.path.exists(cal_weights_path):
    print(f"\nSkipping CRI calibration — weights exist at {cal_weights_path}")
else:
    print("\nTraining CRI calibrated weights...")
    train_cri_calibrated(train_df, results_dir=RESULTS_DIR)

# Train supervised gate
sup_model_path = os.path.join(RESULTS_DIR, 'supervised_gate.pkl')
if args.skip_models and os.path.exists(sup_model_path):
    print(f"\nSkipping supervised gate training — model exists at {sup_model_path}")
else:
    print("\nTraining supervised gate...")
    train_supervised_gate(train_df, dev_df, results_dir=RESULTS_DIR)

# Exp 1 — Latency profiling
print("\nRunning Exp 1: Latency profiling...")
latency_table, lcqg_exp_p = exp1_latency.run(results_dir=RESULTS_DIR)

# Exp 2 — CRI threshold sweep
print("\nRunning Exp 2: CRI sweep...")
optimal_threshold, test_merged, test_results = exp2_cri_sweep.run(
    dev_df, test_df, results_dir=RESULTS_DIR
)

# Exp 3 — Ablation
print("\nRunning Exp 3: Ablation...")
ablation_df = exp3_ablation.run(test_df, results_dir=RESULTS_DIR)

# Exp 4 — Baselines
print("\nRunning Exp 4: Baselines...")
baselines_df, le_metrics, lc_metrics, sg_metrics = exp4_baselines.run(
    train_df, dev_df, test_df, latency_table, results_dir=RESULTS_DIR
)

# Exp 5 — Context robustness
print("\nRunning Exp 5: Context robustness...")
(table4, context_unstable_rate,
 fa_adverse_single, fa_adverse_context,
 latency_delta) = exp5_context_robustness.run(test_df, latency_table, results_dir=RESULTS_DIR)

# Exp 6 — Distribution shift
print("\nRunning Exp 6: Distribution shift...")
(table5, orig_f1, shift_f1,
 orig_lift, shift_lift, stability_str) = exp6_distribution_shift.run(test_df, results_dir=RESULTS_DIR)

# Final summary
def _get_metric(test_results_df, metric, field='point'):
    row = test_results_df[test_results_df['metric'] == metric]
    if len(row) == 0: return 0.0
    return float(row[field].iloc[0])

lowred_pt  = _get_metric(test_results, 'low_quality_exposure_reduction', 'point')
lowred_lo  = _get_metric(test_results, 'low_quality_exposure_reduction', 'ci_lo')
lowred_hi  = _get_metric(test_results, 'low_quality_exposure_reduction', 'ci_hi')
lift_pt    = _get_metric(test_results, 'engagement_proxy_lift', 'point')
lift_lo    = _get_metric(test_results, 'engagement_proxy_lift', 'ci_lo')
lift_hi    = _get_metric(test_results, 'engagement_proxy_lift', 'ci_hi')

le_f1     = le_metrics['f1']
le_f1_lo  = le_metrics.get('f1_ci_lo', le_f1)
le_f1_hi  = le_metrics.get('f1_ci_hi', le_f1)
lc_f1     = lc_metrics['f1']
lc_f1_lo  = lc_metrics.get('f1_ci_lo', lc_f1)
lc_f1_hi  = lc_metrics.get('f1_ci_hi', lc_f1)
sg_f1     = sg_metrics['f1']

p50_exp = lcqg_exp_p['P50']
p95_exp = lcqg_exp_p['P95']
p99_exp = lcqg_exp_p['P99']
budget_ok = "YES" if p95_exp < 10.0 else "NO"

n_h = int((df['quality_tier'] == 2).sum())
n_m = int((df['quality_tier'] == 1).sum())
n_l = int((df['quality_tier'] == 0).sum())
art_prev = float(df[df['quality_tier'] == 0]['artifact_flag'].mean() * 100.0)

fa_reduction = fa_adverse_single - fa_adverse_context

print("""
================================================================
LCQG SIMULATION RESULTS — FINAL SUMMARY
================================================================""")
print(f"Mode / N creatives:                {args.mode} / {N_CREATIVES}")
print("----------------------------------------------------------------")
print("DATA")
print(f"Quality distribution (H/M/L):     {n_h} / {n_m} / {n_l}")
print(f"Artifact prevalence (LOW tier):    {art_prev:.1f}%")
print("----------------------------------------------------------------")
print("LATENCY (full_LCQG_expert)")
print(f"P50 / P95 / P99:                   {p50_exp:.1f} / {p95_exp:.1f} / {p99_exp:.1f} ms")
print(f"10ms budget satisfied:             {budget_ok}")
print("----------------------------------------------------------------")
print(f"MAIN RESULT (test_split, optimal threshold={optimal_threshold:.2f})")
print(f"Low-quality exposure reduction:    {lowred_pt:.1f}% [CI: {lowred_lo:.1f}, {lowred_hi:.1f}]")
print(f"Engagement proxy lift:             +{lift_pt:.1f}% [CI: +{lift_lo:.1f}, +{lift_hi:.1f}]")
print(f"LCQG expert    F1:                 {le_f1:.2f}  [CI: {le_f1_lo:.2f}, {le_f1_hi:.2f}]")
print(f"LCQG calibrated F1:                {lc_f1:.2f}  [CI: {lc_f1_lo:.2f}, {lc_f1_hi:.2f}]")
print(f"Supervised gate F1:                {sg_f1:.2f}")
print("No gate        F1:                 0.00")
print("----------------------------------------------------------------")
print("CONTEXT ROBUSTNESS")
print(f"Context-unstable rate:             {context_unstable_rate:.1f}%")
print(f"False accept reduction (adverse):  {fa_reduction:.1f}%")
print(f"Context-aware latency overhead:    +{latency_delta:.1f}ms (pre-deploy only)")
print("----------------------------------------------------------------")
print("DISTRIBUTION SHIFT")
print(f"F1 drop under shift:               {orig_f1:.2f} → {shift_f1:.2f} (delta={shift_f1-orig_f1:.2f})")
print(f"Threshold stability:               [{stability_str}]")
print("================================================================")
