import os
import json
import numpy as np
import pandas as pd

from autoeval_rtb.scoring.cri import score_cri
from autoeval_rtb.utils.metrics import compute_prf1, compute_engagement_lift, compute_low_quality_exposure_reduction

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')


def run_offline_gate(test_df, results_dir=None):
    """
    Offline gate baseline:
      - Same CRI threshold as LCQG (loaded from results/optimal_threshold.json)
      - 200ms constant latency
      - 15% of LOW-tier creatives already served before filter applied
        (effective recall_low reduced by 0.15 multiplicatively)

    Returns
    -------
    dict of metrics
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    threshold_path = os.path.join(results_dir, 'optimal_threshold.json')
    with open(threshold_path, 'r') as f:
        threshold_data = json.load(f)
    threshold = threshold_data['optimal_threshold']

    df = test_df.copy()
    embeddings       = np.stack(df['embedding'].apply(np.asarray).values)
    brand_embeddings = np.stack(df['brand_embedding'].apply(np.asarray).values)

    gate_decisions = []
    for i in range(len(df)):
        row = df.iloc[i]
        result = score_cri(
            embeddings[i], brand_embeddings[i], row['prs_noise'],
            row['device_type'], row['slot_type'],
            mode='expert', threshold=threshold, results_dir=results_dir
        )
        gate_decisions.append(result['gate'])

    df['gate_decision'] = gate_decisions
    passed_df = df[df['gate_decision'] == 1].copy()

    y_true = (df['quality_tier'] == 0).astype(int).values
    y_pred = (df['gate_decision'] == 0).astype(int).values

    prf = compute_prf1(y_true, y_pred)

    # Delayed filtering reduces effective LOW-tier recall.
    adjusted_recall = prf['recall'] * (1.0 - 0.15)

    p = prf['precision']
    r = adjusted_recall
    f1_adj = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    eng_lift = compute_engagement_lift(passed_df, df)
    low_red  = compute_low_quality_exposure_reduction(passed_df, df)

    # Apply the same delayed-filtering penalty to exposure reduction.
    adjusted_low_red = low_red * (1.0 - 0.15)

    return {
        'method':                    'offline_gate',
        'precision':                 prf['precision'],
        'recall':                    adjusted_recall,
        'f1':                        f1_adj,
        'engagement_proxy_lift':     eng_lift,
        'low_quality_exposure_reduction': adjusted_low_red,
        'latency_overhead_ms':       200.0,
        'rejection_rate':            float((df['gate_decision'] == 0).mean()),
    }
