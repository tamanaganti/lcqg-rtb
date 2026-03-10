import os
import time
import numpy as np
import pandas as pd

from autoeval_rtb.scoring.prs import score_prs
from autoeval_rtb.scoring.bci import score_bci
from autoeval_rtb.scoring.acs import score_acs
from autoeval_rtb.scoring.cri import score_cri
from autoeval_rtb.gating.lcqg import LCQG

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')


def _random_unit_embedding(rng, dim=128):
    v = rng.randn(dim)
    return v / (np.linalg.norm(v) + 1e-12)


def profile_latency(n_iter=10_000, results_dir=None):
    """
    Profile P50/P95/P99 latency for all scoring components.

    Returns
    -------
    pd.DataFrame saved to results/table1_latency.csv
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    rng = np.random.RandomState(42)
    brand_rng = np.random.RandomState(43)

    embeddings       = [_random_unit_embedding(rng)       for _ in range(n_iter)]
    brand_embeddings = [_random_unit_embedding(brand_rng) for _ in range(n_iter)]
    prs_noises       = rng.normal(0, 0.05, n_iter)
    device_types     = rng.choice(['mobile', 'desktop', 'tablet'], n_iter, p=[0.55, 0.35, 0.10])
    slot_types       = rng.choice(['banner', 'native', 'sidebar', 'instream'], n_iter,
                                  p=[0.40, 0.25, 0.20, 0.15])

    def _time_fn(fn, args_list):
        times = []
        for args in args_list:
            t0 = time.perf_counter()
            fn(*args)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        return np.array(times)

    def _pct(arr):
        return {
            'P50': float(np.percentile(arr, 50)),
            'P95': float(np.percentile(arr, 95)),
            'P99': float(np.percentile(arr, 99)),
        }

    print(f"Profiling latency over {n_iter} iterations...")

    prs_args = [(embeddings[i], prs_noises[i], slot_types[i]) for i in range(n_iter)]
    prs_times = _time_fn(score_prs, prs_args)
    prs_p = _pct(prs_times)

    bci_args = [(embeddings[i], brand_embeddings[i], device_types[i]) for i in range(n_iter)]
    bci_times = _time_fn(score_bci, bci_args)
    bci_p = _pct(bci_times)

    acs_args = [(embeddings[i], slot_types[i]) for i in range(n_iter)]
    acs_times = _time_fn(score_acs, acs_args)
    acs_p = _pct(acs_times)

    cri_exp_args = [(embeddings[i], brand_embeddings[i], prs_noises[i],
                     device_types[i], slot_types[i]) for i in range(n_iter)]
    cri_exp_times = _time_fn(lambda *a: score_cri(*a, mode='expert'), cri_exp_args)
    cri_exp_p = _pct(cri_exp_times)

    cri_cal_times = _time_fn(lambda *a: score_cri(*a, mode='calibrated'), cri_exp_args)
    cri_cal_p = _pct(cri_cal_times)

    lcqg_exp = LCQG(threshold=0.50, fusion_mode='expert', results_dir=results_dir)
    lcqg_exp_args = [(i, embeddings[i], brand_embeddings[i],
                      prs_noises[i], device_types[i], slot_types[i])
                     for i in range(n_iter)]
    lcqg_exp_times = _time_fn(lcqg_exp.process_bid_request, lcqg_exp_args)
    lcqg_exp_p = _pct(lcqg_exp_times)

    lcqg_cal = LCQG(threshold=0.50, fusion_mode='calibrated', results_dir=results_dir)
    lcqg_cal_times = _time_fn(lcqg_cal.process_bid_request, lcqg_exp_args)
    lcqg_cal_p = _pct(lcqg_cal_times)

    ctx_args = [(embeddings[i], brand_embeddings[i], prs_noises[i],
                 device_types[i], slot_types[i]) for i in range(n_iter)]
    ctx_times = _time_fn(
        lambda *a: score_cri(*a, mode='expert', context_aware=True, results_dir=results_dir),
        ctx_args
    )
    ctx_p = _pct(ctx_times)

    rows = [
        {'component': 'PRS',                                           **prs_p},
        {'component': 'BCI',                                           **bci_p},
        {'component': 'ACS',                                           **acs_p},
        {'component': 'CRI_expert',                                    **cri_exp_p},
        {'component': 'CRI_calibrated',                                **cri_cal_p},
        {'component': 'full_LCQG_expert',                              **lcqg_exp_p},
        {'component': 'full_LCQG_calibrated',                          **lcqg_cal_p},
        {'component': 'context_aware (pre-deploy only, not auction-path)', **ctx_p},
    ]

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(results_dir, 'table1_latency.csv'), index=False)

    print("\n=== TABLE 1: LATENCY PROFILE (ms) ===")
    print(table.to_string(index=False))

    p95_expert = lcqg_exp_p['P95']
    if p95_expert >= 10.0:
        raise AssertionError(
            f"full_LCQG_expert P95 latency {p95_expert:.3f}ms >= 10ms budget! "
            "Latency constraint violated."
        )
    else:
        print(f"\nfull_LCQG_expert P95={p95_expert:.3f}ms < 10ms budget: PASS")

    print("\nNote: context_aware mode is pre-deployment only, not compared to 10ms budget")

    return table, lcqg_exp_p
