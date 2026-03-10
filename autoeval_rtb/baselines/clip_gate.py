import os
import numpy as np
import pandas as pd

from autoeval_rtb.utils.metrics import compute_prf1, compute_engagement_lift, compute_low_quality_exposure_reduction

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')


def run_clip_gate(train_df, test_df, threshold_range=None, results_dir=None):
    """
    CLIP-style gate baseline:
      Gate = cosine_similarity(embedding, HIGH_centroid) >= threshold
      HIGH_centroid = mean of train_split HIGH-tier embeddings
      Sweep same threshold range as exp2. No context adjustment, no fusion.

    Returns
    -------
    dict with optimal metrics and per-threshold sweep DataFrame
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    if threshold_range is None:
        threshold_range = np.arange(0.30, 0.90, 0.05)

    high_train = train_df[train_df['quality_tier'] == 2]
    high_embs  = np.stack(high_train['embedding'].apply(np.asarray).values)
    centroid   = high_embs.mean(axis=0)
    norm_c     = np.linalg.norm(centroid)
    if norm_c > 0:
        centroid /= norm_c

    test_embs = np.stack(test_df['embedding'].apply(np.asarray).values)
    norms     = np.linalg.norm(test_embs, axis=1, keepdims=True)
    norms     = np.where(norms == 0, 1.0, norms)
    cos_sims  = (test_embs / norms) @ centroid

    y_true = (test_df['quality_tier'] == 0).astype(int).values

    best_f1 = -1.0
    best_threshold = threshold_range[0]
    best_metrics   = {}

    sweep_rows = []
    for thr in threshold_range:
        gate_pass = (cos_sims >= thr).astype(int)
        gate_rej  = 1 - gate_pass

        prf = compute_prf1(y_true, gate_rej)
        passed_mask = gate_pass == 1
        passed_df   = test_df.iloc[np.where(passed_mask)[0]]
        eng_lift    = compute_engagement_lift(passed_df, test_df)

        sweep_rows.append({
            'threshold':      round(float(thr), 4),
            'precision':      prf['precision'],
            'recall':         prf['recall'],
            'f1':             prf['f1'],
            'engagement_proxy_lift': eng_lift,
            'rejection_rate': float(gate_rej.mean()),
        })

        if prf['f1'] > best_f1:
            best_f1 = prf['f1']
            best_threshold = float(thr)
            passed_df_best = passed_df
            best_metrics = {
                'method':                    'clip_gate',
                'precision':                 prf['precision'],
                'recall':                    prf['recall'],
                'f1':                        prf['f1'],
                'engagement_proxy_lift':     eng_lift,
                'low_quality_exposure_reduction': compute_low_quality_exposure_reduction(passed_df, test_df),
                'latency_overhead_ms':       None,
                'rejection_rate':            float(gate_rej.mean()),
                'threshold':                 best_threshold,
            }

    sweep_df = pd.DataFrame(sweep_rows)
    return best_metrics, sweep_df
