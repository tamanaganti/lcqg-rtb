import os
import json
import numpy as np
import pandas as pd

from autoeval_rtb.scoring.prs import score_prs
from autoeval_rtb.scoring.bci import score_bci
from autoeval_rtb.scoring.acs import score_acs
from autoeval_rtb.gating.lcqg import LCQG
from autoeval_rtb.utils.metrics import compute_prf1, compute_engagement_lift
from autoeval_rtb.utils.bootstrap import bootstrap_ci
from autoeval_rtb.utils.plots import plot_ablation

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')


def _score_config(df, weights, threshold, results_dir):
    """Apply arbitrary {prs, bci, acs} weights and return gate decisions + cri."""
    embeddings       = np.stack(df['embedding'].apply(np.asarray).values)
    brand_embeddings = np.stack(df['brand_embedding'].apply(np.asarray).values)

    prs_vals = np.array([
        score_prs(embeddings[i], df['prs_noise'].iloc[i], df['slot_type'].iloc[i])
        for i in range(len(df))
    ])
    bci_vals = np.array([
        score_bci(embeddings[i], brand_embeddings[i], df['device_type'].iloc[i])
        for i in range(len(df))
    ])
    acs_vals = np.array([
        score_acs(embeddings[i], df['slot_type'].iloc[i], results_dir)
        for i in range(len(df))
    ])

    wp = weights.get('prs', 0.0)
    wb = weights.get('bci', 0.0)
    wa = weights.get('acs', 0.0)
    total = wp + wb + wa
    if total > 0:
        wp /= total; wb /= total; wa /= total

    cri_vals = np.clip(wp * prs_vals + wb * bci_vals + wa * acs_vals, 0, 1)
    gate_decisions = (cri_vals >= threshold).astype(int)

    out = df.copy().reset_index(drop=True)
    out['cri']          = cri_vals
    out['gate_decision'] = gate_decisions
    return out


def run(test_df, results_dir=None):
    if results_dir is None:
        results_dir = RESULTS_DIR

    print("\n" + "="*60)
    print("EXP 3: ABLATION STUDY")
    print("="*60)

    threshold_path = os.path.join(results_dir, 'optimal_threshold.json')
    with open(threshold_path, 'r') as f:
        threshold = json.load(f)['optimal_threshold']

    weights_path = os.path.join(results_dir, 'cri_calibrated_weights.json')
    with open(weights_path, 'r') as f:
        cal_weights = json.load(f)

    print(f"Learned CRI weights: PRS={cal_weights['prs']:.4f} BCI={cal_weights['bci']:.4f} ACS={cal_weights['acs']:.4f}")

    configs = [
        ('LCQG_expert (0.45/0.35/0.20)',  {'prs': 0.45, 'bci': 0.35, 'acs': 0.20}),
        ('LCQG_calibrated',                cal_weights),
        ('PRS_only',                       {'prs': 1.00, 'bci': 0.00, 'acs': 0.00}),
        ('BCI_only',                       {'prs': 0.00, 'bci': 1.00, 'acs': 0.00}),
        ('ACS_only',                       {'prs': 0.00, 'bci': 0.00, 'acs': 1.00}),
        ('PRS+BCI',                        {'prs': 0.50, 'bci': 0.50, 'acs': 0.00}),
        ('PRS+ACS',                        {'prs': 0.50, 'bci': 0.00, 'acs': 0.50}),
        ('BCI+ACS',                        {'prs': 0.00, 'bci': 0.50, 'acs': 0.50}),
    ]

    rows = []
    for config_name, weights in configs:
        scored_df = _score_config(test_df, weights, threshold, results_dir)

        def _prf_fn(df_s):
            yt = (df_s['quality_tier'] == 0).astype(int).values
            yp = (df_s['gate_decision'] == 0).astype(int).values
            return compute_prf1(yt, yp)

        def _lift_fn(df_s):
            passed = df_s[df_s['gate_decision'] == 1]
            return compute_engagement_lift(passed, df_s)

        prf_pt, prf_lo, prf_hi = bootstrap_ci(scored_df, _prf_fn, seed=42)
        lift_pt, lift_lo, lift_hi = bootstrap_ci(scored_df, _lift_fn, seed=42)

        rows.append({
            'config':             config_name,
            'precision':          prf_pt['precision'],
            'recall':             prf_pt['recall'],
            'f1':                 prf_pt['f1'],
            'f1_ci_lo':           prf_lo['f1'],
            'f1_ci_hi':           prf_hi['f1'],
            'engagement_lift':    lift_pt,
            'lift_ci_lo':         lift_lo,
            'lift_ci_hi':         lift_hi,
        })

        print(f"  {config_name:40s}  F1={prf_pt['f1']:.4f} "
              f"[{prf_lo['f1']:.4f},{prf_hi['f1']:.4f}]  "
              f"lift={lift_pt:.2f}%")

    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv(os.path.join(results_dir, 'table2_ablation.csv'), index=False)

    plot_ablation(ablation_df, results_dir)
    print(f"\nAblation table saved to {os.path.join(results_dir, 'table2_ablation.csv')}")

    return ablation_df
