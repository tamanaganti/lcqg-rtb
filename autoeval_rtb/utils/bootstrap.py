import numpy as np


def bootstrap_ci(data_df, metric_fn, n_bootstrap=1000, ci=0.95, seed=42):
    """
    Compute bootstrap confidence interval for a metric.

    Parameters
    ----------
    data_df    : pd.DataFrame  input data
    metric_fn  : callable(df) -> float or dict
    n_bootstrap: int
    ci         : float  confidence level (e.g. 0.95)
    seed       : int

    Returns
    -------
    (point_estimate, lower_bound, upper_bound)
    If metric_fn returns a dict, returns (dict_point, dict_lower, dict_upper).
    """
    rng = np.random.RandomState(seed)
    point_estimate = metric_fn(data_df)

    n = len(data_df)
    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        sample = data_df.iloc[idx].reset_index(drop=True)
        est = metric_fn(sample)
        bootstrap_estimates.append(est)

    alpha = 1.0 - ci
    lo_pct = alpha / 2.0 * 100.0
    hi_pct = (1.0 - alpha / 2.0) * 100.0

    if isinstance(point_estimate, dict):
        lower = {}
        upper = {}
        for k in point_estimate:
            vals = np.array([e[k] for e in bootstrap_estimates])
            lower[k] = float(np.percentile(vals, lo_pct))
            upper[k] = float(np.percentile(vals, hi_pct))
        return point_estimate, lower, upper
    else:
        vals = np.array(bootstrap_estimates, dtype=float)
        lower = float(np.percentile(vals, lo_pct))
        upper = float(np.percentile(vals, hi_pct))
        return float(point_estimate), lower, upper
