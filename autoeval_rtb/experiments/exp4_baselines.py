import os
import json
import numpy as np
import pandas as pd

from autoeval_rtb.baselines.no_gate        import run_no_gate
from autoeval_rtb.baselines.offline_gate   import run_offline_gate
from autoeval_rtb.baselines.clip_gate      import run_clip_gate
from autoeval_rtb.baselines.supervised_gate import run_supervised_gate
from autoeval_rtb.gating.lcqg             import LCQG
from autoeval_rtb.utils.metrics           import compute_prf1, compute_engagement_lift, compute_low_quality_exposure_reduction
from autoeval_rtb.utils.bootstrap         import bootstrap_ci
from autoeval_rtb.utils.plots             import plot_baselines

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')


def _eval_lcqg(test_df, mode, threshold, results_dir, latency_p50):
    lcqg = LCQG(threshold=threshold, fusion_mode=mode, results_dir=results_dir)
    result_df = lcqg.process_batch(test_df)

    merged = test_df.copy().reset_index(drop=True)
    merged['gate_decision'] = result_df['gate_decision'].values
    merged['cri']           = result_df['cri'].values

    passed = merged[merged['gate_decision'] == 1]
    yt     = (merged['quality_tier'] == 0).astype(int).values
    yp     = (merged['gate_decision'] == 0).astype(int).values

    def _prf_fn(df_s):
        y_t = (df_s['quality_tier'] == 0).astype(int).values
        y_p = (df_s['gate_decision'] == 0).astype(int).values
        return compute_prf1(y_t, y_p)

    def _lift_fn(df_s):
        p = df_s[df_s['gate_decision'] == 1]
        return compute_engagement_lift(p, df_s)

    prf_pt, prf_lo, prf_hi = bootstrap_ci(merged, _prf_fn, seed=42)
    lift_pt, lift_lo, lift_hi = bootstrap_ci(merged, _lift_fn, seed=42)
    low_red = compute_low_quality_exposure_reduction(passed, merged)

    return {
        'method':                    f'lcqg_{mode}',
        'precision':                 prf_pt['precision'],
        'recall':                    prf_pt['recall'],
        'f1':                        prf_pt['f1'],
        'f1_ci_lo':                  prf_lo['f1'],
        'f1_ci_hi':                  prf_hi['f1'],
        'engagement_proxy_lift':     lift_pt,
        'lift_ci_lo':                lift_lo,
        'lift_ci_hi':                lift_hi,
        'low_quality_exposure_reduction': low_red,
        'latency_overhead_ms':       latency_p50,
        'rejection_rate':            float((merged['gate_decision'] == 0).mean()),
    }


def run(train_df, dev_df, test_df, latency_table, results_dir=None):
    if results_dir is None:
        results_dir = RESULTS_DIR

    print("\n" + "="*60)
    print("EXP 4: BASELINE COMPARISON")
    print("="*60)

    threshold_path = os.path.join(results_dir, 'optimal_threshold.json')
    with open(threshold_path, 'r') as f:
        threshold = json.load(f)['optimal_threshold']

    def _get_p50(component):
        row = latency_table[latency_table['component'] == component]
        if len(row) > 0:
            return float(row['P50'].iloc[0])
        return 0.0

    p50_exp  = _get_p50('full_LCQG_expert')
    p50_cal  = _get_p50('full_LCQG_calibrated')

    # Use CRI_expert latency as a proxy for the embedding-similarity baseline.
    p50_clip = _get_p50('CRI_expert')

    all_metrics = []

    ng = run_no_gate(test_df)
    ng['f1_ci_lo'] = ng['f1']; ng['f1_ci_hi'] = ng['f1']
    ng['lift_ci_lo'] = ng['engagement_proxy_lift']; ng['lift_ci_hi'] = ng['engagement_proxy_lift']
    all_metrics.append(ng)
    print(f"  No gate:          F1={ng['f1']:.4f}  lift={ng['engagement_proxy_lift']:.2f}%  lat=0ms")

    og = run_offline_gate(test_df, results_dir)
    og['f1_ci_lo'] = og['f1']; og['f1_ci_hi'] = og['f1']
    og['lift_ci_lo'] = og['engagement_proxy_lift']; og['lift_ci_hi'] = og['engagement_proxy_lift']
    all_metrics.append(og)
    print(f"  Offline gate:     F1={og['f1']:.4f}  lift={og['engagement_proxy_lift']:.2f}%  lat=200ms")

    clip_metrics, _ = run_clip_gate(train_df, test_df, results_dir=results_dir)
    clip_metrics['latency_overhead_ms'] = p50_clip
    clip_metrics['f1_ci_lo'] = clip_metrics['f1']
    clip_metrics['f1_ci_hi'] = clip_metrics['f1']
    clip_metrics['lift_ci_lo'] = clip_metrics['engagement_proxy_lift']
    clip_metrics['lift_ci_hi'] = clip_metrics['engagement_proxy_lift']
    all_metrics.append(clip_metrics)
    print(f"  CLIP gate:        F1={clip_metrics['f1']:.4f}  lift={clip_metrics['engagement_proxy_lift']:.2f}%  lat={p50_clip:.2f}ms")

    sg = run_supervised_gate(test_df, results_dir)
    sg['f1_ci_lo'] = sg['f1']; sg['f1_ci_hi'] = sg['f1']
    sg['lift_ci_lo'] = sg['engagement_proxy_lift']; sg['lift_ci_hi'] = sg['engagement_proxy_lift']
    all_metrics.append(sg)
    print(f"  Supervised gate:  F1={sg['f1']:.4f}  lift={sg['engagement_proxy_lift']:.2f}%  lat={sg['latency_overhead_ms']:.2f}ms")

    le = _eval_lcqg(test_df, 'expert', threshold, results_dir, p50_exp)
    all_metrics.append(le)
    print(f"  LCQG expert:      F1={le['f1']:.4f}  lift={le['engagement_proxy_lift']:.2f}%  lat={p50_exp:.2f}ms")

    lc = _eval_lcqg(test_df, 'calibrated', threshold, results_dir, p50_cal)
    all_metrics.append(lc)
    print(f"  LCQG calibrated:  F1={lc['f1']:.4f}  lift={lc['engagement_proxy_lift']:.2f}%  lat={p50_cal:.2f}ms")

    baselines_df = pd.DataFrame(all_metrics)
    baselines_df.to_csv(os.path.join(results_dir, 'table3_baselines.csv'), index=False)

    print("\n=== TABLE 3: BASELINE COMPARISON ===")
    print(baselines_df[['method', 'precision', 'recall', 'f1',
                         'engagement_proxy_lift', 'latency_overhead_ms']].to_string(index=False))

    if sg['f1'] > le['f1']:
        print("\nNOTE: Supervised gate achieves higher F1.")
        print("      LCQG advantage is interpretability, modularity, and no label dependency.")
    else:
        print("\nLCQG expert achieves competitive or superior F1 vs supervised baseline.")

    plot_baselines(baselines_df, results_dir)

    return baselines_df, le, lc, sg
