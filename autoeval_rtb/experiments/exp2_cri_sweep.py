import os
import json
import numpy as np
import pandas as pd

from autoeval_rtb.gating.lcqg import LCQG
from autoeval_rtb.utils.metrics import (
    compute_prf1, compute_engagement_lift,
    compute_low_quality_exposure_reduction, compute_utility
)
from autoeval_rtb.utils.bootstrap import bootstrap_ci
from autoeval_rtb.utils.plots import plot_cri_sweep

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
THRESHOLD_RANGE = np.arange(0.30, 0.90, 0.05)


def _sweep_metrics(df, threshold, results_dir):
    lcqg = LCQG(threshold=threshold, fusion_mode='expert', results_dir=results_dir)
    result_df = lcqg.process_batch(df)

    merged = df.copy().reset_index(drop=True)
    merged['gate_decision'] = result_df['gate_decision'].values
    merged['cri']           = result_df['cri'].values

    passed_df = merged[merged['gate_decision'] == 1]
    y_true = (merged['quality_tier'] == 0).astype(int).values
    y_pred = (merged['gate_decision'] == 0).astype(int).values

    prf      = compute_prf1(y_true, y_pred)
    eng_lift = compute_engagement_lift(passed_df, merged)
    low_red  = compute_low_quality_exposure_reduction(passed_df, merged)
    rej_rate = float((merged['gate_decision'] == 0).mean())

    low_passed_no_gate  = int((merged['quality_tier'] == 0).sum())
    low_passed_with_gate = int(((merged['quality_tier'] == 0) & (merged['gate_decision'] == 1)).sum())
    low_exp_red = (low_passed_no_gate - low_passed_with_gate) / low_passed_no_gate if low_passed_no_gate > 0 else 0.0

    return {
        'threshold':               float(threshold),
        'rejection_rate':          rej_rate,
        'engagement_proxy_lift':   eng_lift,
        'precision':               prf['precision'],
        'recall':                  prf['recall'],
        'f1':                      prf['f1'],
        'low_quality_exposure_reduction': low_exp_red * 100.0,
    }


def run(dev_df, test_df, results_dir=None):
    if results_dir is None:
        results_dir = RESULTS_DIR

    print("\n" + "="*60)
    print("EXP 2: CRI THRESHOLD SWEEP")
    print("="*60)

    dev_rows = []
    for thr in THRESHOLD_RANGE:
        row = _sweep_metrics(dev_df, round(float(thr), 4), results_dir)
        dev_rows.append(row)
        print(f"  thr={thr:.2f}  rej={row['rejection_rate']:.3f}  "
              f"recall_low={row['recall']:.3f}  lift={row['engagement_proxy_lift']:.2f}%  "
              f"F1={row['f1']:.4f}")

    sweep_df = pd.DataFrame(dev_rows)

    utility_vals = compute_utility(
        sweep_df['engagement_proxy_lift'].values,
        sweep_df['recall'].values,
        sweep_df['precision'].values,
    )
    sweep_df['utility'] = utility_vals

    eligible = sweep_df[
        (sweep_df['rejection_rate'] <= 0.40) &
        (sweep_df['recall'] >= 0.65)
    ]

    if len(eligible) == 0:
        print("WARNING: No threshold satisfies constraints — using argmax utility")
        best_row = sweep_df.loc[sweep_df['utility'].idxmax()]
    else:
        best_row = eligible.loc[eligible['utility'].idxmax()]

    optimal_threshold = float(best_row['threshold'])
    print(f"\nOptimal CRI threshold (constrained): {optimal_threshold:.2f}")

    sweep_df.to_csv(os.path.join(results_dir, 'exp2_sweep_dev.csv'), index=False)
    with open(os.path.join(results_dir, 'optimal_threshold.json'), 'w') as f:
        json.dump({'optimal_threshold': optimal_threshold}, f, indent=2)

    lcqg = LCQG(threshold=optimal_threshold, fusion_mode='expert', results_dir=results_dir)
    test_result_df = lcqg.process_batch(test_df)

    test_merged = test_df.copy().reset_index(drop=True)
    test_merged['gate_decision'] = test_result_df['gate_decision'].values
    test_merged['cri']           = test_result_df['cri'].values

    passed_test = test_merged[test_merged['gate_decision'] == 1]
    y_true_t    = (test_merged['quality_tier'] == 0).astype(int).values
    y_pred_t    = (test_merged['gate_decision'] == 0).astype(int).values

    def _metric_fn_lift(df_sample):
        passed_s = df_sample[df_sample['gate_decision'] == 1]
        return compute_engagement_lift(passed_s, df_sample)

    def _metric_fn_prf(df_sample):
        yt = (df_sample['quality_tier'] == 0).astype(int).values
        yp = (df_sample['gate_decision'] == 0).astype(int).values
        return compute_prf1(yt, yp)

    def _metric_fn_lowred(df_sample):
        passed_s = df_sample[df_sample['gate_decision'] == 1]
        return compute_low_quality_exposure_reduction(passed_s, df_sample)

    lift_pt, lift_lo, lift_hi  = bootstrap_ci(test_merged, _metric_fn_lift,  seed=42)
    prf_pt, prf_lo, prf_hi     = bootstrap_ci(test_merged, _metric_fn_prf,   seed=42)
    lowred_pt, lowred_lo, lowred_hi = bootstrap_ci(test_merged, _metric_fn_lowred, seed=42)

    test_results = pd.DataFrame([{
        'metric': 'engagement_proxy_lift',
        'point':  lift_pt,
        'ci_lo':  lift_lo,
        'ci_hi':  lift_hi,
    }, {
        'metric': 'precision',
        'point':  prf_pt['precision'],
        'ci_lo':  prf_lo['precision'],
        'ci_hi':  prf_hi['precision'],
    }, {
        'metric': 'recall',
        'point':  prf_pt['recall'],
        'ci_lo':  prf_lo['recall'],
        'ci_hi':  prf_hi['recall'],
    }, {
        'metric': 'f1',
        'point':  prf_pt['f1'],
        'ci_lo':  prf_lo['f1'],
        'ci_hi':  prf_hi['f1'],
    }, {
        'metric': 'low_quality_exposure_reduction',
        'point':  lowred_pt,
        'ci_lo':  lowred_lo,
        'ci_hi':  lowred_hi,
    }])
    test_results.to_csv(os.path.join(results_dir, 'exp2_test_results.csv'), index=False)

    print(f"\nTest-set results at optimal threshold {optimal_threshold:.2f}:")
    print(f"  Low-quality exposure reduction: {lowred_pt:.1f}% [CI: {lowred_lo:.1f}, {lowred_hi:.1f}]")
    print(f"  Engagement proxy lift:          +{lift_pt:.1f}%  [CI: +{lift_lo:.1f}, +{lift_hi:.1f}]")
    print(f"  Precision: {prf_pt['precision']:.2f}  Recall: {prf_pt['recall']:.2f}  F1: {prf_pt['f1']:.2f} [CI shown]")

    plot_cri_sweep(sweep_df, optimal_threshold, results_dir)

    return optimal_threshold, test_merged, test_results
