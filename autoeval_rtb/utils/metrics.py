import numpy as np


def compute_prf1(y_true_low_binary, y_pred_rejected):
    """
    Compute precision, recall, F1 for identifying LOW-tier creatives.

    Parameters
    ----------
    y_true_low_binary : array-like  1 if creative is LOW tier, 0 otherwise
    y_pred_rejected   : array-like  1 if gate rejects, 0 if passes

    Returns
    -------
    dict: {precision, recall, f1}
    """
    yt = np.asarray(y_true_low_binary, dtype=int)
    yp = np.asarray(y_pred_rejected,   dtype=int)

    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {'precision': precision, 'recall': recall, 'f1': f1}


def compute_engagement_lift(passed_df, all_df):
    """
    Engagement proxy lift of passed creatives vs all creatives.

    Returns
    -------
    float  (percentage, e.g. 5.0 means +5%)
    """
    if len(passed_df) == 0:
        return 0.0
    mean_passed = passed_df['engagement_proxy'].mean()
    mean_all    = all_df['engagement_proxy'].mean()
    if mean_all == 0:
        return 0.0
    return float((mean_passed / mean_all - 1.0) * 100.0)


def compute_low_quality_exposure_reduction(passed_df, all_df):
    """
    Reduction in low-quality creatives served.

    low_passed_no_gate = number of LOW-tier creatives in all_df
    low_passed_with_gate = number of LOW-tier creatives in passed_df

    Returns
    -------
    float  (percentage)
    """
    low_no_gate   = int((all_df['quality_tier'] == 0).sum())
    low_with_gate = int((passed_df['quality_tier'] == 0).sum())
    if low_no_gate == 0:
        return 0.0
    return float((low_no_gate - low_with_gate) / low_no_gate * 100.0)


def latency_percentiles(times_list):
    """
    Compute P50, P95, P99 from a list of latency values (ms).

    Returns
    -------
    dict: {P50, P95, P99}
    """
    arr = np.asarray(times_list, dtype=float)
    return {
        'P50': float(np.percentile(arr, 50)),
        'P95': float(np.percentile(arr, 95)),
        'P99': float(np.percentile(arr, 99)),
    }


def compute_utility(engagement_lift_arr, recall_low_arr, precision_arr):
    """
    Compute utility score for threshold selection.
    utility = 0.5*norm(engagement_lift) + 0.3*recall_low + 0.2*precision
    norm = (x - min) / (max - min) across sweep.

    Parameters
    ----------
    engagement_lift_arr : array-like  engagement lift values across thresholds
    recall_low_arr      : array-like  recall of LOW-tier across thresholds
    precision_arr       : array-like  precision across thresholds

    Returns
    -------
    np.ndarray of utility scores (same length)
    """
    el = np.asarray(engagement_lift_arr, dtype=float)
    rl = np.asarray(recall_low_arr,      dtype=float)
    pr = np.asarray(precision_arr,       dtype=float)

    el_min, el_max = el.min(), el.max()
    if el_max > el_min:
        el_norm = (el - el_min) / (el_max - el_min)
    else:
        el_norm = np.zeros_like(el)

    utility = 0.5 * el_norm + 0.3 * rl + 0.2 * pr
    return utility
