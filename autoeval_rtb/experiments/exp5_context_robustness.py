import os
import json
import numpy as np
import pandas as pd

from autoeval_rtb.scoring.cri import score_cri
from autoeval_rtb.gating.lcqg import LCQG
from autoeval_rtb.utils.metrics import compute_prf1
from autoeval_rtb.utils.bootstrap import bootstrap_ci
from autoeval_rtb.utils.plots import plot_context_robustness

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')


def _gate_under_context(df, device_type_override, slot_type_override, threshold, results_dir):
    """Compute gate decisions for a specific context override."""
    embeddings       = np.stack(df['embedding'].apply(np.asarray).values)
    brand_embeddings = np.stack(df['brand_embedding'].apply(np.asarray).values)

    gates = []
    cris  = []
    for i in range(len(df)):
        dt = device_type_override if device_type_override else df['device_type'].iloc[i]
        st = slot_type_override   if slot_type_override   else df['slot_type'].iloc[i]
        result = score_cri(
            embeddings[i], brand_embeddings[i], df['prs_noise'].iloc[i],
            dt, st, mode='expert', threshold=threshold, results_dir=results_dir
        )
        gates.append(result['gate'])
        cris.append(result['cri'])
    return np.array(gates), np.array(cris)


def _gate_context_aware(df, threshold, results_dir):
    """Compute context-aware gate decisions."""
    embeddings       = np.stack(df['embedding'].apply(np.asarray).values)
    brand_embeddings = np.stack(df['brand_embedding'].apply(np.asarray).values)

    gates = []
    cris  = []
    for i in range(len(df)):
        result = score_cri(
            embeddings[i], brand_embeddings[i], df['prs_noise'].iloc[i],
            df['device_type'].iloc[i], df['slot_type'].iloc[i],
            mode='expert', context_aware=True, threshold=threshold, results_dir=results_dir
        )
        gates.append(result['gate'])
        cris.append(result['cri'])
    return np.array(gates), np.array(cris)


def run(test_df, latency_table, results_dir=None):
    if results_dir is None:
        results_dir = RESULTS_DIR

    print("\n" + "="*60)
    print("EXP 5: CONTEXT ROBUSTNESS")
    print("="*60)

    threshold_path = os.path.join(results_dir, 'optimal_threshold.json')
    with open(threshold_path, 'r') as f:
        threshold = json.load(f)['optimal_threshold']

    df = test_df.copy().reset_index(drop=True)
    n  = len(df)

    # Single-context evaluations
    gate_orig, cri_orig = _gate_under_context(df, None, None, threshold, results_dir)
    gate_worst, cri_worst = _gate_under_context(df, 'mobile', 'banner', threshold, results_dir)
    gate_best, cri_best = _gate_under_context(df, 'desktop', 'instream', threshold, results_dir)

    # Context-aware evaluation
    gate_ctx_aware, cri_ctx_aware = _gate_context_aware(df, threshold, results_dir)

    # Gate changes across original, worst, and best single-context evaluations.
    unstable_mask = ~(
        (gate_orig == gate_worst) & (gate_worst == gate_best)
    )
    context_unstable_rate = float(unstable_mask.mean() * 100.0)

    # False-accept rates for LOW-tier creatives that pass the gate.
    low_mask = (df['quality_tier'] == 0).values

    fa_orig_single     = float((gate_orig[low_mask]  == 1).mean() * 100.0)
    fa_adverse_single  = float((gate_worst[low_mask] == 1).mean() * 100.0)
    fa_best_single     = float((gate_best[low_mask]  == 1).mean() * 100.0)

    fa_adverse_context = float((gate_ctx_aware[low_mask] == 1).mean() * 100.0)

    # Robustness F1 on the unstable subset.
    unstable_idx = np.where(unstable_mask)[0]
    if len(unstable_idx) > 0:
        yt_unstable = (df['quality_tier'].iloc[unstable_idx] == 0).astype(int).values
        yp_unstable = (gate_orig[unstable_idx] == 0).astype(int)
        robustness_prf = compute_prf1(yt_unstable, yp_unstable)
    else:
        robustness_prf = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    def _get_p50(component):
        row = latency_table[latency_table['component'] == component]
        if len(row) > 0:
            return float(row['P50'].iloc[0])
        return 0.0

    p50_single  = _get_p50('full_LCQG_expert')
    p50_context = _get_p50('context_aware (pre-deploy only, not auction-path)')
    latency_delta = p50_context - p50_single

    # Bootstrap CIs for context instability and original-context false accepts.
    df_aug = df.copy()
    df_aug['gate_orig']       = gate_orig
    df_aug['gate_worst']      = gate_worst
    df_aug['gate_best']       = gate_best
    df_aug['gate_ctx_aware']  = gate_ctx_aware
    df_aug['cri_single']      = cri_orig
    df_aug['cri_context']     = cri_ctx_aware

    def _fn_unstable_rate(df_s):
        u = ~((df_s['gate_orig'] == df_s['gate_worst']) & (df_s['gate_worst'] == df_s['gate_best']))
        return float(u.mean() * 100.0)

    def _fn_fa_orig(df_s):
        low = df_s['quality_tier'] == 0
        if low.sum() == 0: return 0.0
        return float((df_s.loc[low, 'gate_orig'] == 1).mean() * 100.0)

    ur_pt, ur_lo, ur_hi = bootstrap_ci(df_aug, _fn_unstable_rate, seed=42)
    fa_pt, fa_lo, fa_hi = bootstrap_ci(df_aug, _fn_fa_orig,       seed=42)

    table4 = pd.DataFrame([{
        'metric':           'context_unstable_rate',
        'point':            context_unstable_rate,
        'ci_lo':            ur_lo,
        'ci_hi':            ur_hi,
    }, {
        'metric':           'false_accept_original_single',
        'point':            fa_orig_single,
        'ci_lo':            fa_lo,
        'ci_hi':            fa_hi,
    }, {
        'metric':           'false_accept_context_aware_gate',
        'point':            fa_adverse_context,
        'ci_lo':            fa_adverse_context,
        'ci_hi':            fa_adverse_context,
    }, {
        'metric':           'false_accept_adverse_single',
        'point':            fa_adverse_single,
        'ci_lo':            fa_adverse_single,
        'ci_hi':            fa_adverse_single,
    }, {
        'metric':           'false_accept_best',
        'point':            fa_best_single,
        'ci_lo':            fa_best_single,
        'ci_hi':            fa_best_single,
    }, {
        'metric':           'robustness_f1',
        'point':            robustness_prf['f1'],
        'ci_lo':            robustness_prf['f1'],
        'ci_hi':            robustness_prf['f1'],
    }, {
        'metric':           'latency_delta_ms',
        'point':            latency_delta,
        'ci_lo':            latency_delta,
        'ci_hi':            latency_delta,
    }])
    table4.to_csv(os.path.join(results_dir, 'table4_context.csv'), index=False)

    print(f"  Context-unstable creative rate:         {context_unstable_rate:.1f}%")
    print(f"  False accept (original context):")
    print(f"    Single-context gate: {fa_orig_single:.1f}%")
    print(f"    Context-aware gate:  {fa_adverse_context:.1f}%")
    reduction = fa_orig_single - fa_adverse_context
    print(f"    Reduction:           {reduction:.1f}pp")
    print(f"  Context-aware latency overhead:         +{latency_delta:.1f}ms (pre-deployment mode only)")

    gated_single_df  = df_aug[['quality_tier', 'cri_single', 'gate_orig']].copy()
    gated_single_df.rename(columns={'cri_single': 'cri', 'gate_orig': 'gate_decision'}, inplace=True)
    gated_context_df = df_aug[['quality_tier', 'cri_context', 'gate_ctx_aware']].copy()
    gated_context_df.rename(columns={'cri_context': 'cri', 'gate_ctx_aware': 'gate_decision'}, inplace=True)

    fa_vals = {
        'fa_adverse_single': fa_adverse_single,
        'fa_orig_single':    fa_orig_single,
        'fa_best_single':    fa_best_single,
        'fa_context_aware':  fa_adverse_context,
    }
    plot_context_robustness(df, gated_single_df, gated_context_df, results_dir,
                            fa_vals=fa_vals)

    return table4, context_unstable_rate, fa_orig_single, fa_adverse_context, latency_delta
