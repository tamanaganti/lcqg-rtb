import numpy as np
import pandas as pd

from autoeval_rtb.utils.metrics import compute_prf1, compute_engagement_lift, compute_low_quality_exposure_reduction
from autoeval_rtb.utils.bootstrap import bootstrap_ci


def run_no_gate(test_df):
    """
    Baseline: pass all creatives (no gating).
    Gate decision = 1 for all creatives.

    Returns
    -------
    dict of metrics with bootstrap CIs
    """
    df = test_df.copy()
    df['gate_decision'] = 1

    passed_df = df

    y_true = (df['quality_tier'] == 0).astype(int).values
    y_pred = np.zeros(len(df), dtype=int)

    prf = compute_prf1(y_true, y_pred)
    eng_lift = compute_engagement_lift(passed_df, df)
    low_red  = compute_low_quality_exposure_reduction(passed_df, df)

    return {
        'method':                    'no_gate',
        'precision':                 prf['precision'],
        'recall':                    prf['recall'],
        'f1':                        prf['f1'],
        'engagement_proxy_lift':     eng_lift,
        'low_quality_exposure_reduction': low_red,
        'latency_overhead_ms':       0.0,
        'rejection_rate':            0.0,
    }
