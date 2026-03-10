import numpy as np

_rng_prs = np.random.RandomState(200)
_W_prs_raw = _rng_prs.randn(128)
W_PRS = _W_prs_raw / np.linalg.norm(_W_prs_raw)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def score_prs(embedding, prs_noise, slot_type):
    """
    Compute Perceptual Relevance Score.

    Parameters
    ----------
    embedding : array-like, shape (128,)
    prs_noise : float  (pre-generated per-creative noise from dataset)
    slot_type : str    one of 'banner', 'native', 'sidebar', 'instream'

    Returns
    -------
    float in [0, 1]
    """
    emb = np.asarray(embedding, dtype=float)
    raw = float(np.dot(emb, W_PRS))
    score = _sigmoid((raw + float(prs_noise)) * 3.0)

    if slot_type == 'banner':
        score *= 0.88
    elif slot_type == 'instream':
        score *= 1.12

    return float(np.clip(score, 0.0, 1.0))
