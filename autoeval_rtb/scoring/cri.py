import os
import json
import numpy as np
from sklearn.linear_model import Ridge

from autoeval_rtb.scoring.prs import score_prs
from autoeval_rtb.scoring.bci import score_bci
from autoeval_rtb.scoring.acs import score_acs

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')

EXPERT_WEIGHTS = {'prs': 0.45, 'bci': 0.35, 'acs': 0.20}

_calibrated_weights = None


def _load_calibrated_weights(results_dir=None):
    global _calibrated_weights
    if _calibrated_weights is not None:
        return _calibrated_weights
    if results_dir is None:
        results_dir = RESULTS_DIR
    path = os.path.join(results_dir, 'cri_calibrated_weights.json')
    with open(path, 'r') as f:
        _calibrated_weights = json.load(f)
    return _calibrated_weights


def train_cri_calibrated(train_df, results_dir=None):
    """
    Train Ridge regression on train_split to learn CRI weights.
    Features: [PRS, BCI, ACS], target: engagement_proxy.
    Saves normalized weights to results/cri_calibrated_weights.json.
    """
    global _calibrated_weights
    if results_dir is None:
        results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    import pandas as pd
    embeddings      = np.stack(train_df['embedding'].apply(np.asarray).values)
    brand_embeddings = np.stack(train_df['brand_embedding'].apply(np.asarray).values)
    prs_noises      = train_df['prs_noise'].values
    device_types    = train_df['device_type'].values
    slot_types      = train_df['slot_type'].values
    targets         = train_df['engagement_proxy'].values

    prs_scores = np.array([
        score_prs(embeddings[i], prs_noises[i], slot_types[i])
        for i in range(len(train_df))
    ])
    bci_scores = np.array([
        score_bci(embeddings[i], brand_embeddings[i], device_types[i])
        for i in range(len(train_df))
    ])
    acs_scores = np.array([
        score_acs(embeddings[i], slot_types[i], results_dir)
        for i in range(len(train_df))
    ])

    X = np.column_stack([prs_scores, bci_scores, acs_scores])
    y = targets

    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)

    coefs = np.abs(ridge.coef_)
    coefs_sum = coefs.sum()
    if coefs_sum == 0:
        coefs = np.array([1/3, 1/3, 1/3])
    else:
        coefs = coefs / coefs_sum

    weights = {'prs': float(coefs[0]), 'bci': float(coefs[1]), 'acs': float(coefs[2])}
    path = os.path.join(results_dir, 'cri_calibrated_weights.json')
    with open(path, 'w') as f:
        json.dump(weights, f, indent=2)

    _calibrated_weights = weights
    print(f"Calibrated CRI weights: PRS={weights['prs']:.4f} BCI={weights['bci']:.4f} ACS={weights['acs']:.4f}")
    print(f"Saved to {path}")
    return weights


def score_cri(embedding, brand_embedding, prs_noise,
              device_type, slot_type,
              mode='expert', context_aware=False,
              threshold=0.50, results_dir=None):
    """
    Compute Composite Relevance Index.

    Parameters
    ----------
    embedding       : array-like (128,)
    brand_embedding : array-like (128,)
    prs_noise       : float
    device_type     : str
    slot_type       : str
    mode            : 'expert' | 'calibrated'
    context_aware   : bool — if True, uses context_aware mode (pre-deployment only)
    threshold       : float — gate threshold
    results_dir     : str or None

    Returns
    -------
    dict: {prs, bci, acs, cri, gate, fusion_mode, context_aware}
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    def _compute_once(emb, bemb, pn, dt, st):
        prs = score_prs(emb, pn, st)
        bci = score_bci(emb, bemb, dt)
        acs_threshold_adj = -0.05 if st == 'instream' else 0.0
        acs = score_acs(emb, st, results_dir)
        return prs, bci, acs

    if context_aware:
        emb  = np.asarray(embedding, dtype=float)
        bemb = np.asarray(brand_embedding, dtype=float)

        ctx_original = (device_type, slot_type)
        ctx_worst    = ('mobile', 'banner')
        ctx_best     = ('desktop', 'instream')

        cri_vals = []
        for dt, st in [ctx_original, ctx_worst, ctx_best]:
            prs_k, bci_k, acs_k = _compute_once(emb, bemb, prs_noise, dt, st)
            if mode == 'expert':
                w = EXPERT_WEIGHTS
            else:
                w = _load_calibrated_weights(results_dir)
            cri_k = w['prs'] * prs_k + w['bci'] * bci_k + w['acs'] * acs_k
            cri_vals.append(cri_k)

        cri_mean = float(np.mean(cri_vals))
        cri_std  = float(np.std(cri_vals))
        cri_final = float(np.clip(cri_mean - 3.00 * cri_std, 0.0, 1.0))

        # Report component scores under the original context.
        prs_r, bci_r, acs_r = _compute_once(emb, bemb, prs_noise, device_type, slot_type)
        gate = 1 if cri_final >= threshold else 0

        return {
            'prs':           prs_r,
            'bci':           bci_r,
            'acs':           acs_r,
            'cri':           cri_final,
            'gate':          gate,
            'fusion_mode':   mode,
            'context_aware': True,
        }

    else:
        emb  = np.asarray(embedding, dtype=float)
        bemb = np.asarray(brand_embedding, dtype=float)

        prs, bci, acs = _compute_once(emb, bemb, prs_noise, device_type, slot_type)

        if mode == 'expert':
            w = EXPERT_WEIGHTS
        else:
            w = _load_calibrated_weights(results_dir)

        cri = float(np.clip(w['prs'] * prs + w['bci'] * bci + w['acs'] * acs, 0.0, 1.0))
        gate = 1 if cri >= threshold else 0

        return {
            'prs':           prs,
            'bci':           bci,
            'acs':           acs,
            'cri':           cri,
            'gate':          gate,
            'fusion_mode':   mode,
            'context_aware': False,
        }
