import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from autoeval_rtb.scoring.prs import score_prs
from autoeval_rtb.scoring.bci import score_bci
from autoeval_rtb.scoring.acs import score_acs
from autoeval_rtb.utils.metrics import compute_prf1, compute_engagement_lift, compute_low_quality_exposure_reduction

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')


def _build_features(df, device_enc, slot_enc, results_dir):
    embeddings       = np.stack(df['embedding'].apply(np.asarray).values)
    brand_embeddings = np.stack(df['brand_embedding'].apply(np.asarray).values)

    prs_scores = np.array([
        score_prs(embeddings[i], df['prs_noise'].iloc[i], df['slot_type'].iloc[i])
        for i in range(len(df))
    ])
    bci_scores = np.array([
        score_bci(embeddings[i], brand_embeddings[i], df['device_type'].iloc[i])
        for i in range(len(df))
    ])
    acs_scores = np.array([
        score_acs(embeddings[i], df['slot_type'].iloc[i], results_dir)
        for i in range(len(df))
    ])

    dev_enc  = device_enc.transform(df['device_type'].values)
    slot_enc_vals = slot_enc.transform(df['slot_type'].values)

    X = np.column_stack([prs_scores, bci_scores, acs_scores,
                         dev_enc.astype(float), slot_enc_vals.astype(float)])
    return X


def train_supervised_gate(train_df, dev_df, results_dir=None):
    """
    Train LogisticRegression gate. Select threshold on dev_split.
    Saves model to results/supervised_gate.pkl.

    Returns
    -------
    (model_bundle, optimal_threshold, dev_metrics)
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    device_enc = LabelEncoder()
    slot_enc   = LabelEncoder()

    device_enc.fit(train_df['device_type'].values)
    slot_enc.fit(train_df['slot_type'].values)

    X_train = _build_features(train_df, device_enc, slot_enc, results_dir)
    y_train = (train_df['quality_tier'] == 0).astype(int).values

    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)

    X_dev  = _build_features(dev_df, device_enc, slot_enc, results_dir)
    y_dev  = (dev_df['quality_tier'] == 0).astype(int).values
    proba  = lr.predict_proba(X_dev)[:, 1]

    best_f1 = -1.0
    best_thr = 0.5
    for thr in np.arange(0.10, 0.95, 0.05):
        y_pred_rej = (proba >= thr).astype(int)
        prf = compute_prf1(y_dev, y_pred_rej)
        if prf['f1'] > best_f1:
            best_f1  = prf['f1']
            best_thr = float(thr)

    model_bundle = {
        'lr':         lr,
        'device_enc': device_enc,
        'slot_enc':   slot_enc,
        'threshold':  best_thr,
    }
    model_path = os.path.join(results_dir, 'supervised_gate.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_bundle, f)

    print(f"Supervised gate saved to {model_path}. Dev threshold={best_thr:.2f} Dev F1={best_f1:.4f}")
    return model_bundle, best_thr


def run_supervised_gate(test_df, results_dir=None):
    """
    Evaluate supervised gate on test_split.

    Returns
    -------
    dict of metrics
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    model_path = os.path.join(results_dir, 'supervised_gate.pkl')
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)

    lr         = bundle['lr']
    device_enc = bundle['device_enc']
    slot_enc   = bundle['slot_enc']
    threshold  = bundle['threshold']

    X_test = _build_features(test_df, device_enc, slot_enc, results_dir)
    proba  = lr.predict_proba(X_test)[:, 1]

    n_sample = min(1000, len(test_df))
    lat_times = []
    for i in range(n_sample):
        t0 = time.perf_counter()
        lr.predict_proba(X_test[i:i+1])
        t1 = time.perf_counter()
        lat_times.append((t1 - t0) * 1000.0)
    p50_lat = float(np.percentile(lat_times, 50))

    y_pred_rej = (proba >= threshold).astype(int)
    y_true     = (test_df['quality_tier'] == 0).astype(int).values

    prf = compute_prf1(y_true, y_pred_rej)
    passed_df = test_df.iloc[np.where(y_pred_rej == 0)[0]]
    eng_lift  = compute_engagement_lift(passed_df, test_df)
    low_red   = compute_low_quality_exposure_reduction(passed_df, test_df)

    print(f"Supervised gate: precision={prf['precision']:.4f} recall={prf['recall']:.4f} "
          f"F1={prf['f1']:.4f} engagement_lift={eng_lift:.2f}% latency_P50={p50_lat:.3f}ms")

    return {
        'method':                    'supervised_gate',
        'precision':                 prf['precision'],
        'recall':                    prf['recall'],
        'f1':                        prf['f1'],
        'engagement_proxy_lift':     eng_lift,
        'low_quality_exposure_reduction': low_red,
        'latency_overhead_ms':       p50_lat,
        'rejection_rate':            float(y_pred_rej.mean()),
    }
