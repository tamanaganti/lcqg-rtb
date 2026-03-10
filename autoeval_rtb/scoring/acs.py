import os
import pickle
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
MODEL_PATH  = os.path.join(RESULTS_DIR, 'acs_model.pkl')

_acs_model = None


def _extract_artifact_features(embeddings):
    """
    Compute 3 artifact-detection features for a batch of embeddings.
    embeddings: np.ndarray, shape (N, 128)
    Returns: np.ndarray, shape (N, 3)
    """
    emb = np.asarray(embeddings, dtype=float)
    if emb.ndim == 1:
        emb = emb[np.newaxis, :]

    feature_a = np.mean(np.abs(emb[:, 0:16]), axis=1)                         # Type A detector
    feature_b = np.max(np.abs(emb), axis=1) / (np.mean(np.abs(emb), axis=1) + 1e-9)  # Type B spikes
    feature_c = np.var(emb[:, 48:80], axis=1)                                  # Type C collapse

    return np.column_stack([feature_a, feature_b, feature_c])


def train_acs(train_embeddings, results_dir=None):
    """
    Train ACS on HIGH-tier, non-artifact embeddings from train_split.
    Saves model to results/acs_model.pkl.

    Parameters
    ----------
    train_embeddings : np.ndarray, shape (N, 128)
    """
    global _acs_model
    if results_dir is None:
        results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, 'acs_model.pkl')

    train_emb = np.asarray(train_embeddings, dtype=float)

    iso_forest = IsolationForest(contamination=0.05, n_estimators=100, random_state=42)
    iso_forest.fit(train_emb)

    art_feats = _extract_artifact_features(train_emb)
    art_scaler = MinMaxScaler()
    art_scaler.fit(art_feats)

    iso_raw_train = iso_forest.decision_function(train_emb)
    iso_scaler = MinMaxScaler()
    iso_scaler.fit(iso_raw_train.reshape(-1, 1))

    model_bundle = {
        'iso_forest':  iso_forest,
        'art_scaler':  art_scaler,
        'iso_scaler':  iso_scaler,
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_bundle, f)

    _acs_model = model_bundle
    print(f"ACS model saved to {model_path}")


def _load_model(results_dir=None):
    global _acs_model
    if _acs_model is not None:
        return _acs_model
    if results_dir is None:
        results_dir = RESULTS_DIR
    model_path = os.path.join(results_dir, 'acs_model.pkl')
    with open(model_path, 'rb') as f:
        _acs_model = pickle.load(f)
    return _acs_model


def score_acs(embedding, slot_type, results_dir=None):
    """
    Compute Artifact Cleanliness Score.

    Parameters
    ----------
    embedding : array-like, shape (128,)
    slot_type : str  ('banner', 'native', 'sidebar', 'instream')
                     For instream, the ACS threshold is treated as -0.05 stricter (caller handles).

    Returns
    -------
    float in [0, 1]
    """
    model = _load_model(results_dir)
    emb = np.asarray(embedding, dtype=float).reshape(1, -1)

    iso_raw = model['iso_forest'].decision_function(emb)  # shape (1,)
    iso_norm = float(model['iso_scaler'].transform(iso_raw.reshape(-1, 1))[0, 0])

    art_feats = _extract_artifact_features(emb)  # shape (1, 3)
    art_scaled = model['art_scaler'].transform(art_feats)  # shape (1, 3)
    pattern_score = float(1.0 - np.mean(art_scaled))

    acs = float(np.clip(0.6 * iso_norm + 0.4 * pattern_score, 0.0, 1.0))
    return acs
