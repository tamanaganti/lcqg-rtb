import os
from autoeval_rtb.gating.latency_profiler import profile_latency

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')


def run(results_dir=None):
    if results_dir is None:
        results_dir = RESULTS_DIR
    print("\n" + "="*60)
    print("EXP 1: LATENCY PROFILING")
    print("="*60)
    table, lcqg_exp_p = profile_latency(n_iter=10_000, results_dir=results_dir)
    return table, lcqg_exp_p
