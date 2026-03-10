import os
import json
import numpy as np
import pandas as pd

from autoeval_rtb.gating.lcqg import LCQG
from autoeval_rtb.utils.metrics import (
    compute_prf1, compute_engagement_lift,
    compute_low_quality_exposure_reduction, compute_utility
)
from autoeval_rtb.utils.plots import plot_distribution_shift

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
THRESHOLD_RANGE = np.arange(0.30, 0.90, 0.05)


def _apply_shift(test_df, results_dir):
    """Apply combined distribution shifts to the test split."""
    df = test_df.copy().reset_index(drop=True)
    rng = np.random.RandomState(42)

    # Shift 1 — increase artifact prevalence in the LOW tier.
    low_idx = np.where(df['quality_tier'] == 0)[0]
    n_extra = int(round(0.10 * len(low_idx)))
    no_artifact_low = low_idx[df['artifact_flag'].iloc[low_idx].values == 0]
    if len(no_artifact_low) >= n_extra:
        flip_idx = rng.choice(no_artifact_low, size=n_extra, replace=False)
    else:
        flip_idx = no_artifact_low

    for idx in flip_idx:
        emb = np.asarray(df.at[idx, 'embedding'], dtype=float)
        noise = rng.normal(0, 1, 16)
        emb[0:16] += noise
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        df.at[idx, 'embedding']       = emb.tolist()
        df.at[idx, 'artifact_flag']   = 1
        df.at[idx, 'artifact_type']   = 'A'

    # Shift 2 — swap mobile and desktop contexts for 30% of creatives.
    swap_n = int(round(0.30 * len(df)))
    swap_idx = rng.choice(len(df), size=swap_n, replace=False)
    for idx in swap_idx:
        dt = df.at[idx, 'device_type']
        if dt == 'mobile':
            df.at[idx, 'device_type'] = 'desktop'
        elif dt == 'desktop':
            df.at[idx, 'device_type'] = 'mobile'

    # Shift 3 — perturb the engagement proxy with Gaussian noise.
    noise_eng = rng.normal(0, 0.02, len(df))
    df['engagement_proxy'] = np.clip(df['engagement_proxy'].values + noise_eng, 0.0, None)

    return df


def _eval_lcqg(df, threshold, results_dir):
    lcqg = LCQG(threshold=threshold, fusion_mode='expert', results_dir=results_dir)
    result_df = lcqg.process_batch(df)

    merged = df.copy().reset_index(drop=True)
    merged['gate_decision'] = result_df['gate_decision'].values
    merged['cri']           = result_df['cri'].values

    passed = merged[merged['gate_decision'] == 1]
    yt     = (merged['quality_tier'] == 0).astype(int).values
    yp     = (merged['gate_decision'] == 0).astype(int).values
    prf    = compute_prf1(yt, yp)
    lift   = compute_engagement_lift(passed, merged)
    return prf['f1'], lift


def _find_constrained_optimal(df, results_dir):
    """Find constrained-optimal threshold on shifted data."""
    rows = []
    for thr in THRESHOLD_RANGE:
        lcqg = LCQG(threshold=float(thr), fusion_mode='expert', results_dir=results_dir)
        rdf  = lcqg.process_batch(df)
        merged = df.copy().reset_index(drop=True)
        merged['gate_decision'] = rdf['gate_decision'].values

        passed = merged[merged['gate_decision'] == 1]
        yt = (merged['quality_tier'] == 0).astype(int).values
        yp = (merged['gate_decision'] == 0).astype(int).values
        prf      = compute_prf1(yt, yp)
        rej_rate = float((merged['gate_decision'] == 0).mean())
        lift     = compute_engagement_lift(passed, merged)
        rows.append({
            'threshold':      round(float(thr), 4),
            'rejection_rate': rej_rate,
            'recall':         prf['recall'],
            'precision':      prf['precision'],
            'f1':             prf['f1'],
            'engagement_proxy_lift': lift,
        })

    sweep_df = pd.DataFrame(rows)
    util_vals = compute_utility(
        sweep_df['engagement_proxy_lift'].values,
        sweep_df['recall'].values,
        sweep_df['precision'].values,
    )
    sweep_df['utility'] = util_vals

    eligible = sweep_df[
        (sweep_df['rejection_rate'] <= 0.40) &
        (sweep_df['recall'] >= 0.65)
    ]
    if len(eligible) == 0:
        best_row = sweep_df.loc[sweep_df['utility'].idxmax()]
    else:
        best_row = eligible.loc[eligible['utility'].idxmax()]
    return float(best_row['threshold'])


def run(test_df, results_dir=None):
    if results_dir is None:
        results_dir = RESULTS_DIR

    print("\n" + "="*60)
    print("EXP 6: DISTRIBUTION SHIFT")
    print("="*60)

    threshold_path = os.path.join(results_dir, 'optimal_threshold.json')
    with open(threshold_path, 'r') as f:
        original_threshold = json.load(f)['optimal_threshold']

    orig_f1, orig_lift = _eval_lcqg(test_df, original_threshold, results_dir)
    print(f"  Original:  F1={orig_f1:.4f}  lift={orig_lift:.2f}%  threshold={original_threshold:.2f}")

    shifted_df = _apply_shift(test_df, results_dir)

    shift_f1, shift_lift = _eval_lcqg(shifted_df, original_threshold, results_dir)
    print(f"  Shifted:   F1={shift_f1:.4f}  lift={shift_lift:.2f}%  threshold={original_threshold:.2f}")

    shifted_optimal = _find_constrained_optimal(shifted_df, results_dir)
    threshold_stable = abs(shifted_optimal - original_threshold) < 1e-6
    stability_str = "stable" if threshold_stable else "unstable"

    print(f"\n  F1 under distribution shift:   {shift_f1:.4f} (vs {orig_f1:.4f} original, "
          f"delta={shift_f1-orig_f1:.4f})")
    print(f"  Lift under distribution shift: {shift_lift:.1f}% (vs {orig_lift:.1f}% original)")
    print(f"  Threshold stability:           [{stability_str}]")

    table5 = pd.DataFrame([{
        'scenario':           'original',
        'threshold':          original_threshold,
        'f1':                 orig_f1,
        'engagement_lift':    orig_lift,
    }, {
        'scenario':           'shifted',
        'threshold':          original_threshold,
        'f1':                 shift_f1,
        'engagement_lift':    shift_lift,
    }, {
        'scenario':           'shifted_optimal_threshold',
        'threshold':          shifted_optimal,
        'f1':                 shift_f1,
        'engagement_lift':    shift_lift,
    }])
    table5.to_csv(os.path.join(results_dir, 'table5_shift.csv'), index=False)

    plot_distribution_shift(
        {'f1': orig_f1,  'engagement_lift': orig_lift},
        {'f1': shift_f1, 'engagement_lift': shift_lift},
        results_dir
    )

    return table5, orig_f1, shift_f1, orig_lift, shift_lift, stability_str
