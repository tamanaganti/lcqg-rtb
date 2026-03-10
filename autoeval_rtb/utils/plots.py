import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

STYLE = 'seaborn-v0_8-whitegrid'
FIGSIZE = (10, 5)
DPI = 150

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')


def _save(fig, fname, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, fname)
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def plot_cri_sweep(sweep_df, optimal_threshold, results_dir=None):
    """
    fig2_cri_sweep.png:
      Panel 1: rejection_rate and engagement_proxy_lift vs threshold (dual y-axis)
      Panel 2: precision, recall, F1 vs threshold
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    with plt.style.context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)

        ax1_r = ax1.twinx()
        ax1.plot(sweep_df['threshold'], sweep_df['rejection_rate'],
                 color='steelblue', label='Rejection Rate', linewidth=2)
        ax1_r.plot(sweep_df['threshold'], sweep_df['engagement_proxy_lift'],
                   color='darkorange', label='Engagement Lift (%)', linewidth=2, linestyle='--')
        ax1.axvline(optimal_threshold, color='red', linestyle=':', linewidth=1.5,
                    label=f'Optimal θ={optimal_threshold:.2f}')
        ax1.set_xlabel('CRI Threshold')
        ax1.set_ylabel('Rejection Rate', color='steelblue')
        ax1_r.set_ylabel('Engagement Proxy Lift (%)', color='darkorange')
        ax1.set_title('Rejection Rate & Engagement Lift vs Threshold')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_r.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

        ax2.plot(sweep_df['threshold'], sweep_df['precision'], label='Precision',
                 color='green', linewidth=2)
        ax2.plot(sweep_df['threshold'], sweep_df['recall'],    label='Recall',
                 color='blue',  linewidth=2)
        ax2.plot(sweep_df['threshold'], sweep_df['f1'],        label='F1',
                 color='purple', linewidth=2)
        ax2.axvline(optimal_threshold, color='red', linestyle=':', linewidth=1.5,
                    label=f'Optimal θ={optimal_threshold:.2f}')
        ax2.set_xlabel('CRI Threshold')
        ax2.set_ylabel('Score')
        ax2.set_title('Precision / Recall / F1 vs Threshold')
        ax2.legend(fontsize=8)

        fig.tight_layout()
        _save(fig, 'fig2_cri_sweep.png', results_dir)


def plot_ablation(ablation_df, results_dir=None):
    """
    fig3_ablation.png: Grouped bar chart of F1 per config with CI error bars.
    Expert (config 1) and calibrated (config 2) highlighted in different color.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    configs   = ablation_df['config'].tolist()
    f1_vals   = ablation_df['f1'].tolist()
    f1_lo     = ablation_df['f1_ci_lo'].tolist()
    f1_hi     = ablation_df['f1_ci_hi'].tolist()

    colors = []
    for c in configs:
        if 'expert' in c.lower():
            colors.append('steelblue')
        elif 'calibrated' in c.lower():
            colors.append('darkorange')
        else:
            colors.append('gray')

    yerr_lo = [f1_vals[i] - f1_lo[i] for i in range(len(f1_vals))]
    yerr_hi = [f1_hi[i] - f1_vals[i] for i in range(len(f1_vals))]
    yerr = [yerr_lo, yerr_hi]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        xs = range(len(configs))
        ax.bar(xs, f1_vals, color=colors, yerr=yerr, capsize=4,
               error_kw={'elinewidth': 1.5})
        ax.set_xticks(list(xs))
        ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('F1 Score')
        ax.set_title('Ablation Study: F1 per Configuration')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='steelblue',  label='LCQG expert'),
            Patch(facecolor='darkorange', label='LCQG calibrated'),
            Patch(facecolor='gray',       label='Other configs'),
        ]
        ax.legend(handles=legend_elements, fontsize=8)

        fig.tight_layout()
        _save(fig, 'fig3_ablation.png', results_dir)


def plot_baselines(baselines_df, results_dir=None):
    """
    fig4_baselines.png:
      Left panel:  F1 comparison with CI bars
      Right panel: engagement_proxy_lift comparison with CI bars
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    methods = baselines_df['method'].tolist()
    colors  = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    with plt.style.context(STYLE):
        fig, (ax_f1, ax_eng) = plt.subplots(1, 2, figsize=FIGSIZE)

        xs = np.arange(len(methods))

        f1_vals = baselines_df['f1'].tolist()
        f1_lo   = [baselines_df['f1'].iloc[i] - baselines_df['f1_ci_lo'].iloc[i]
                   for i in range(len(methods))]
        f1_hi   = [baselines_df['f1_ci_hi'].iloc[i] - baselines_df['f1'].iloc[i]
                   for i in range(len(methods))]
        ax_f1.bar(xs, f1_vals, color=colors,
                  yerr=[f1_lo, f1_hi], capsize=4, error_kw={'elinewidth': 1.5})
        ax_f1.set_xticks(xs)
        ax_f1.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        ax_f1.set_ylabel('F1 Score')
        ax_f1.set_title('F1 Comparison')

        el_vals = baselines_df['engagement_proxy_lift'].tolist()
        el_lo   = [baselines_df['engagement_proxy_lift'].iloc[i] - baselines_df['lift_ci_lo'].iloc[i]
                   for i in range(len(methods))]
        el_hi   = [baselines_df['lift_ci_hi'].iloc[i] - baselines_df['engagement_proxy_lift'].iloc[i]
                   for i in range(len(methods))]
        ax_eng.bar(xs, el_vals, color=colors,
                   yerr=[el_lo, el_hi], capsize=4, error_kw={'elinewidth': 1.5})
        ax_eng.set_xticks(xs)
        ax_eng.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        ax_eng.set_ylabel('Engagement Proxy Lift (%)')
        ax_eng.set_title('Engagement Lift Comparison')

        fig.tight_layout()
        _save(fig, 'fig4_baselines.png', results_dir)


def plot_context_robustness(test_df, gated_single_df, gated_context_df,
                            results_dir=None, fa_vals=None):
    """
    fig5_context.png:
      Panel 1: KDE of CRI distribution — single-context vs context-aware, per tier
      Panel 2: bar chart of false_accept rate under adverse/original/best context
                for single-context gate vs context-aware gate.

    fa_vals dict keys:
      fa_adverse_single, fa_orig_single, fa_best_single  — single-context gate FA rates
      fa_context_aware                                    — context-aware gate FA rate
                                                            (same value shown for all contexts)
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    tier_colors = {2: 'green', 1: 'orange', 0: 'red'}
    tier_names  = {2: 'HIGH', 1: 'MED', 0: 'LOW'}

    with plt.style.context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)

        for tier_val in [2, 1, 0]:
            mask = test_df['quality_tier'] == tier_val
            if mask.sum() == 0:
                continue
            cri_single  = gated_single_df.loc[mask, 'cri']
            cri_context = gated_context_df.loc[mask, 'cri']
            color = tier_colors[tier_val]
            cri_single.plot.kde(ax=ax1, label=f'{tier_names[tier_val]} single',
                                color=color, linestyle='-')
            cri_context.plot.kde(ax=ax1, label=f'{tier_names[tier_val]} context-aware',
                                 color=color, linestyle='--')
        ax1.set_xlabel('CRI')
        ax1.set_ylabel('Density')
        ax1.set_title('CRI Distribution: Single vs Context-Aware')
        ax1.legend(fontsize=7)

        categories = ['Adverse\n(mobile+banner)', 'Original', 'Best\n(desktop+instream)']

        if fa_vals is not None:
            fa_adv_s  = fa_vals['fa_adverse_single']
            fa_orig_s = fa_vals['fa_orig_single']
            fa_best_s = fa_vals['fa_best_single']
            # Use the same context-aware false-accept rate for each display category.
            fa_ctx    = fa_vals['fa_context_aware']
            fa_adv_c = fa_orig_c = fa_best_c = fa_ctx
        else:
            fa_adv_s = fa_orig_s = fa_best_s = 0.0
            fa_adv_c = fa_orig_c = fa_best_c = 0.0

        x_cats = np.arange(len(categories))
        width  = 0.35
        bars_s = ax2.bar(x_cats - width/2, [fa_adv_s, fa_orig_s, fa_best_s],
                         width, label='Single-context', color='steelblue')
        bars_c = ax2.bar(x_cats + width/2, [fa_adv_c, fa_orig_c, fa_best_c],
                         width, label='Context-aware',  color='darkorange')

        # Annotate bars so near-zero context-aware values remain visible.
        y_max = max(fa_adv_s, fa_orig_s, fa_best_s, fa_adv_c, fa_orig_c, fa_best_c, 1.0)
        for bar in bars_s:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + y_max * 0.01,
                     f'{h:.1f}%', ha='center', va='bottom', fontsize=7)
        for bar in bars_c:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + y_max * 0.01,
                     f'{h:.2f}%', ha='center', va='bottom', fontsize=7, color='darkorange')

        ax2.set_xticks(x_cats)
        ax2.set_xticklabels(categories, fontsize=8)
        ax2.set_ylabel('False Accept Rate (%)')
        ax2.set_title('False Accept Rate by Context')
        ax2.legend(fontsize=8)

        fig.tight_layout()
        _save(fig, 'fig5_context.png', results_dir)


def plot_distribution_shift(original_metrics, shifted_metrics, results_dir=None):
    """
    fig6_shift.png: Bar chart of F1 and engagement_lift — original vs shifted
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    metrics = ['F1', 'Engagement Lift (%)']
    orig_vals    = [original_metrics['f1'],  original_metrics['engagement_lift']]
    shifted_vals = [shifted_metrics['f1'],   shifted_metrics['engagement_lift']]

    x = np.arange(len(metrics))
    width = 0.35

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.bar(x - width/2, orig_vals,    width, label='Original',  color='steelblue')
        ax.bar(x + width/2, shifted_vals, width, label='Shifted',   color='darkorange')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylabel('Score / %')
        ax.set_title('Performance under Distribution Shift')
        ax.legend(fontsize=9)
        fig.tight_layout()
        _save(fig, 'fig6_shift.png', results_dir)
