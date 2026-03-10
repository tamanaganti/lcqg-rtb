import numpy as np


def score_bci(embedding, brand_embedding, device_type):
    """
    Compute Brand Consistency Index.

    Parameters
    ----------
    embedding       : array-like, shape (128,)
    brand_embedding : array-like, shape (128,)
    device_type     : str  one of 'mobile', 'desktop', 'tablet'

    Returns
    -------
    float in [0, 1]
    """
    emb  = np.asarray(embedding, dtype=float)
    bemb = np.asarray(brand_embedding, dtype=float)

    norm_e = np.linalg.norm(emb)
    norm_b = np.linalg.norm(bemb)

    if norm_e == 0 or norm_b == 0:
        cos_sim = 0.0
    else:
        cos_sim = float(np.dot(emb, bemb) / (norm_e * norm_b))

    score = (1.0 + cos_sim) / 2.0

    if device_type == 'mobile':
        score *= 0.90
    elif device_type == 'tablet':
        score *= 0.95

    return float(np.clip(score, 0.0, 1.0))
