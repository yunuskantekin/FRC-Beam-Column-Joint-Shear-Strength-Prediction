import math
import os

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from config import (
    SELECTED_FEATURES, CATEGORICAL_FEATURES,
    SHAP_LABELS, _SHAP_TREE_MODELS,
)

_CMAP_BR = LinearSegmentedColormap.from_list('shap_br', ['#0000ff', '#ff0000'])


def _nice_bounds(lo, hi):
    # rounds axis limits to clean numbers based on the data range
    if hi == lo:
        return lo, hi
    rng = hi - lo
    mag = 10 ** math.floor(math.log10(rng / 5))
    return math.floor(lo / mag) * mag, math.ceil(hi / mag) * mag


def _fmt_dep_tick(val, feat_name, is_max):
    """Format a colorbar min/max tick label for the dependence plot grid."""
    if feat_name in ('ρb', 'ρc', 'ρv', 'Vf'):
        return f'{val * 100:.2f}%'
    elif feat_name == 'fyv':
        mult = 10
        r = math.ceil(val / mult) * mult if is_max else math.floor(val / mult) * mult
        return str(int(r))
    elif feat_name == 'ftf':
        mult = 50
        r = math.ceil(val / mult) * mult if is_max else math.floor(val / mult) * mult
        return str(int(r))
    elif feat_name == 'Lf/Df':
        r = math.ceil(val) if is_max else math.floor(val)
        return str(int(r))
    else:
        return f'{val:.3g}'


def _shap_select_model(models_rep, aggregated):
    available = [a for a in _SHAP_TREE_MODELS if a in models_rep]
    if not available:
        return None
    return min(
        available,
        key=lambda a: aggregated.get(a, {}).get('test_rmse_mean', float('inf')),
    )


# sum one-hot encoded SHAP columns back to their original feature
def _shap_aggregate_1d(sv, oh_cols):
    oh_list = list(oh_cols)
    out = np.zeros((sv.shape[0], len(SELECTED_FEATURES)))
    for j, feat in enumerate(SELECTED_FEATURES):
        if feat in CATEGORICAL_FEATURES:
            idxs = [i for i, c in enumerate(oh_list)
                    if c.startswith(feat + '_')]
            if idxs:
                out[:, j] = sv[:, idxs].sum(axis=1)
        else:
            if feat in oh_list:
                out[:, j] = sv[:, oh_list.index(feat)]
    return out


# same but for interaction values (feature x feature matrix)
def _shap_aggregate_2d(sv_int, oh_cols):
    oh_list = list(oh_cols)
    n_orig = len(SELECTED_FEATURES)
    feat_oh = {}
    for j, feat in enumerate(SELECTED_FEATURES):
        if feat in CATEGORICAL_FEATURES:
            feat_oh[j] = [i for i, c in enumerate(oh_list) if c.startswith(feat + '_')]
        else:
            feat_oh[j] = [oh_list.index(feat)] if feat in oh_list else []
    out = np.zeros((sv_int.shape[0], n_orig, n_orig))
    for i in range(n_orig):
        for j in range(n_orig):
            for ii in feat_oh[i]:
                for jj in feat_oh[j]:
                    out[:, i, j] += sv_int[:, ii, jj]
    return out


def _make_X_display(splits_rep):
    X_display = splits_rep['X_raw_test'].rename(columns=SHAP_LABELS).copy()
    if 'Joint Type' in X_display.columns:
        jt_raw = X_display['Joint Type'].astype(str).str.lower().str.strip()
        X_display['Joint Type'] = pd.to_numeric(
            jt_raw.map({'exterior': 0.0, 'interior': 1.0}), errors='coerce'
        )
    return X_display


def _style_beeswarm(fig_bee, ax_bee):
    # Some styling for beeswarm figures
    ax_bee.spines['bottom'].set_color('black')
    ax_bee.spines['bottom'].set_linewidth(1.5)
    for col in ax_bee.collections:
        col.set_sizes([s * 1.4 for s in col.get_sizes()] if col.get_sizes() else [14])
    ax_bee.set_xlabel('')
    ax_bee.set_yticklabels(ax_bee.get_yticklabels(), fontsize=9, ha='right')
    ax_bee.yaxis.set_tick_params(pad=2)
    # Symmetric x-axis
    lo, hi = ax_bee.get_xlim()
    max_abs = max(abs(lo), abs(hi))
    ax_bee.set_xlim(-max_abs, max_abs)


def run_shap_analysis(models_rep, splits_rep, aggregated, output_prefix, timestamp):
    try:
        import shap
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
    except ImportError:
        print("Shap/Matplotlib not available: Skipping SHAP analysis.")
        return None, None

    alg = _shap_select_model(models_rep, aggregated)
    if alg is None:
        print("No tree model available for SHAP: Skipping.")
        return None, None

    print(f"\nSHAP analysis for {alg}:")

    if alg == 'LGBM':
        X_shap = splits_rep['X_test_lgb']
        needs_aggregation = False
    elif alg == 'CB':
        X_shap = splits_rep['X_test_cb']
        needs_aggregation = False
    else:
        X_shap = splits_rep['X_test_oh']
        needs_aggregation = True

    feat_labels = [SHAP_LABELS.get(f, f) for f in SELECTED_FEATURES]
    explainer = shap.TreeExplainer(models_rep[alg])

    print("Computing SHAP values.")
    sv = np.array(explainer.shap_values(X_shap))

    print("Computing SHAP interaction values.")
    sv_int = np.array(explainer.shap_interaction_values(X_shap))

    if needs_aggregation:
        oh_cols = list(X_shap.columns)
        sv = _shap_aggregate_1d(sv, oh_cols)
        sv_int = _shap_aggregate_2d(sv_int, oh_cols)

    X_display = _make_X_display(splits_rep)
    fiber_label = SHAP_LABELS.get('fbrtype', 'Fiber Type')
    bee_feat_idx = [i for i, l in enumerate(feat_labels) if l != fiber_label]
    sv_bee = sv[:, bee_feat_idx]
    X_bee = X_display[[feat_labels[i] for i in bee_feat_idx]]
    feat_bee = [feat_labels[i] for i in bee_feat_idx]
    cm_to_in = 1 / 2.54

    # Plot 1: Beeswarm
    with plt.rc_context({
        'font.family': 'Times New Roman', 'font.size': 10,
        'axes.titlesize': 10, 'axes.labelsize': 10,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
    }):
        shap.summary_plot(sv_bee, X_bee, feature_names=feat_bee,
                          show=False, plot_size=None)
        fig_bee = plt.gcf()
        fig_bee.set_size_inches(12 * cm_to_in, 8 * cm_to_in)
        ax_bee = fig_bee.axes[0]
        _style_beeswarm(fig_bee, ax_bee)
        ax_bee.set_title(f"{alg} (best model)", fontsize=10)
        plt.tight_layout()
        fig_bee.subplots_adjust(left=0.18)
        path_bee = os.path.join(os.path.dirname(output_prefix), f'SHAP_Beeswarm_{alg}_{timestamp}.tiff')
        fig_bee.savefig(path_bee, dpi=500, bbox_inches='tight')
        plt.close(fig_bee)

    # Plot 2: Bar
    shap.summary_plot(sv, X_display, feature_names=feat_labels,
                      plot_type='bar', show=False)
    plt.gcf().set_size_inches(8, 6)
    plt.tight_layout()
    path_bar = os.path.join(os.path.dirname(output_prefix), f'SHAP_Bar_{alg}_{timestamp}.tiff')
    plt.savefig(path_bar, dpi=100, bbox_inches='tight')
    plt.close()

    # Plot 3: f'c centered dependence plots
    fc_label = SHAP_LABELS.get('fc', "f'c")
    fc_idx = feat_labels.index(fc_label)
    shap_fc = sv[:, fc_idx]
    fc_vals = X_display[fc_label].values

    # specimens with zero stirrups are excluded as colorbar coloring breaks down
    ZERO_EXCLUDE_FEATS = {
        SHAP_LABELS.get('rhov', 'ρv'),
        SHAP_LABELS.get('fyv', 'fyv'),
    }

    dep_color_feats = [
        SHAP_LABELS.get('jtype', 'Joint Type'),
        SHAP_LABELS.get('ftf', 'ftf'),
        SHAP_LABELS.get('vf', 'Vf'),
        SHAP_LABELS.get('rhob', 'ρb'),
        SHAP_LABELS.get('rhoc', 'ρc'),
        SHAP_LABELS.get('rhov', 'ρv'),
        SHAP_LABELS.get('ar', 'Lf/Df'),
        SHAP_LABELS.get('fyv', 'fyv'),
        SHAP_LABELS.get('n', 'n'),
    ]
    dep_color_feats = [f for f in dep_color_feats if f in X_display.columns]

    _ROTATE_LABELS = {'Joint Type', 'Lf/Df'}

    n_dep = len(dep_color_feats)
    n_cols = 3
    n_rows = (n_dep + n_cols - 1) // n_cols
    fig_w = 22 * cm_to_in
    fig_h = 16 * cm_to_in

    fc_xmin = math.floor(float(fc_vals.min()) / 10) * 10
    fc_xmax = math.ceil(float(fc_vals.max()) / 10) * 10
    shap_ymin, shap_ymax = _nice_bounds(float(shap_fc.min()), float(shap_fc.max()))

    with plt.rc_context({
        'font.family': 'Times New Roman', 'font.size': 10,
        'axes.titlesize': 10, 'axes.labelsize': 10,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
    }):
        fig_dep, axes_dep = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                                         sharex=True, sharey=True)
        axes_dep = np.array(axes_dep).ravel()

        for k, color_feat in enumerate(dep_color_feats):
            ax = axes_dep[k]
            row = k // n_cols
            col = k % n_cols
            c_vals = X_display[color_feat].values

            if color_feat in ZERO_EXCLUDE_FEATS:
                valid = c_vals.astype(float) != 0.0
                c_vals = c_vals[valid]
                fc_plot = fc_vals[valid]
                shap_plot = shap_fc[valid]
            else:
                fc_plot = fc_vals
                shap_plot = shap_fc

            if color_feat == SHAP_LABELS.get('jtype', 'Joint Type'):
                col_ext = shap.plots.colors.red_blue(0.0)
                col_int = shap.plots.colors.red_blue(1.0)
                ext_mask = (c_vals == 0.0)
                int_mask = (c_vals == 1.0)
                ax.scatter(fc_plot[ext_mask], shap_plot[ext_mask],
                           color=col_ext, alpha=0.75, edgecolors='#aaaaaa', linewidths=0.3, s=40,
                           zorder=2)
                ax.scatter(fc_plot[int_mask], shap_plot[int_mask],
                           color=col_int, alpha=0.85, edgecolors='#aaaaaa', linewidths=0.3, s=40,
                           zorder=3)
                # Invisible dummy colorbar to keep panel size identical to others
                sc_dummy = ax.scatter([], [], c=[], cmap='gray', vmin=0, vmax=1)
                cbar_dummy = fig_dep.colorbar(sc_dummy, ax=ax, pad=0.02)
                cbar_dummy.ax.set_visible(False)
                legend_elems = [
                    Line2D([0], [0], marker='o', color='none', markerfacecolor=col_int,
                           markeredgecolor='#aaaaaa', markeredgewidth=0.3,
                           markersize=6, label='Interior'),
                    Line2D([0], [0], marker='o', color='none', markerfacecolor=col_ext,
                           markeredgecolor='#aaaaaa', markeredgewidth=0.3,
                           markersize=6, label='Exterior'),
                ]
                ax.legend(handles=legend_elems, title=color_feat, title_fontsize=8,
                          fontsize=8, frameon=False, loc='lower right', handletextpad=0.3)
            else:
                c_float = c_vals.astype(float)
                data_min = float(c_float.min())
                data_max = float(c_float.max())
                sort_idx = np.argsort(c_float)
                sc = ax.scatter(fc_plot[sort_idx], shap_plot[sort_idx],
                                c=c_float[sort_idx],
                                cmap=shap.plots.colors.red_blue, alpha=0.75,
                                edgecolors='#aaaaaa', linewidths=0.3, s=40,
                                vmin=data_min, vmax=data_max)
                cbar_dep = fig_dep.colorbar(sc, ax=ax, pad=0.02)
                cbar_dep.set_ticks([data_min, (data_min + data_max) / 2, data_max])
                cbar_dep.set_ticklabels([
                    _fmt_dep_tick(data_min, color_feat, False),
                    color_feat,
                    _fmt_dep_tick(data_max, color_feat, True),
                ])
                for tk in cbar_dep.ax.yaxis.get_major_ticks():
                    if tk.label1.get_text() == color_feat:
                        tk.tick1line.set_visible(False)
                        tk.tick2line.set_visible(False)
                        if color_feat in _ROTATE_LABELS:
                            tk.label1.set_rotation(90)
                            tk.label1.set_va('center')

            if col == 0:
                ax.set_ylabel(f"SHAP value for {fc_label}")
            if row == n_rows - 1:
                ax.set_xlabel(fc_label)
            ax.axhline(0, color='grey', linewidth=0.6, linestyle='--')

        axes_dep[0].set_xlim(fc_xmin, fc_xmax)
        axes_dep[0].set_ylim(shap_ymin, shap_ymax)
        axes_dep[0].set_xticks([fc_xmin, fc_xmax])
        axes_dep[0].set_yticks([shap_ymin, 0, shap_ymax])

        for k in range(n_dep, len(axes_dep)):
            axes_dep[k].set_visible(False)
        fig_dep.tight_layout()
        path_dep = os.path.join(os.path.dirname(output_prefix), f'SHAP_Dependence_{alg}_{timestamp}.tiff')
        fig_dep.savefig(path_dep, dpi=500, bbox_inches='tight')
        plt.close(fig_dep)

    # Plot 4: SHAP Interaction Matrix
    n_feat = len(feat_labels)
    interact_mat = np.abs(sv_int).mean(axis=0)
    np.fill_diagonal(interact_mat, 0)
    vmax_int = float(interact_mat.max())
    norm_int = Normalize(vmin=0, vmax=vmax_int)
    cmap_int = _CMAP_BR

    with plt.rc_context({
        'font.family': 'Times New Roman', 'font.size': 9,
        'axes.titlesize': 9, 'axes.labelsize': 9,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
    }):
        fig_int, ax_int = plt.subplots(figsize=(16 * cm_to_in, 16 * cm_to_in))

        for i in range(n_feat):
            for j in range(n_feat):
                if i == j:
                    continue
                val = interact_mat[i, j]
                half = (val / vmax_int) * 0.48 if vmax_int > 0 else 0
                rect = Rectangle((j - half, i - half), 2 * half, 2 * half,
                                  color=cmap_int(norm_int(val)), zorder=2)
                ax_int.add_patch(rect)

        ax_int.set_xlim(-0.5, n_feat - 0.5)
        ax_int.set_ylim(n_feat - 0.5, -0.5)
        ax_int.set_xticks(range(n_feat))
        ax_int.set_yticks(range(n_feat))
        ax_int.set_xticklabels(feat_labels, rotation=45, ha='right')
        ax_int.set_yticklabels(feat_labels)
        ax_int.set_aspect('equal')
        ax_int.tick_params(length=0)
        for spine in ax_int.spines.values():
            spine.set_visible(False)

        sm = ScalarMappable(cmap=cmap_int, norm=norm_int)
        sm.set_array([])

        fig_int.tight_layout(rect=[0, 0, 0.88, 1])

        # Compute colorbar height to match the actual plotted square.
        # With aspect='equal', the square fits inside the axes bounding box
        # constrained by the shorter dimension.
        pos = ax_int.get_position()
        fig_w = fig_int.get_figwidth()
        fig_h = fig_int.get_figheight()
        sq_in = min(pos.width * fig_w, pos.height * fig_h)
        sq_h = sq_in / fig_h  # square height in fig fraction
        sq_y0 = pos.y0 + (pos.height - sq_h) / 2  # vertically centred in bbox

        cbar_ax = fig_int.add_axes([pos.x1 + 0.01, sq_y0, 0.025, sq_h])
        cb = fig_int.colorbar(sm, cax=cbar_ax)
        cb.set_ticks([0, vmax_int])
        cb.set_ticklabels(['Low', 'High'])
        cb.set_label('Interaction')

        path_int = os.path.join(os.path.dirname(output_prefix), f'SHAP_Interaction_{alg}_{timestamp}.tiff')
        fig_int.savefig(path_int, dpi=500, bbox_inches='tight')
        plt.close(fig_int)

    print(f"Saved at: {path_bee}.")
    print(f"Saved at: {path_bar}.")
    print(f"Saved at: {path_dep}.")
    print(f"Saved at: {path_int}.")

    shap_df = pd.DataFrame(sv, columns=feat_labels)
    return shap_df, alg


def run_shap_comparison(models_rep, splits_rep, output_prefix, timestamp, primary_alg=None):
    try:
        import shap
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Shap/Matplotlib not available: Skipping SHAP comparison.")
        return

    comparison_algs = ['XGB', 'RF', 'GB', 'LGBM', 'CB']
    available = [a for a in comparison_algs if a in models_rep and a != primary_alg]
    if not available:
        return

    print("\nSHAP comparison beeswarms:")
    feat_labels = [SHAP_LABELS.get(f, f) for f in SELECTED_FEATURES]
    fiber_label = SHAP_LABELS.get('fbrtype', 'Fiber Type')
    bee_feat_idx = [i for i, l in enumerate(feat_labels) if l != fiber_label]
    feat_bee = [feat_labels[i] for i in bee_feat_idx]
    cm_to_in = 1 / 2.54
    X_display = _make_X_display(splits_rep)

    for alg in available:
        print(f"Computing for {alg}:")
        if alg == 'LGBM':
            X_shap = splits_rep['X_test_lgb']
            needs_agg = False
        elif alg == 'CB':
            X_shap = splits_rep['X_test_cb']
            needs_agg = False
        else:
            X_shap = splits_rep['X_test_oh']
            needs_agg = True

        try:
            explainer = shap.TreeExplainer(models_rep[alg])
            sv = np.array(explainer.shap_values(X_shap))
            if needs_agg:
                sv = _shap_aggregate_1d(sv, list(X_shap.columns))
        except Exception as e:
            print(f"SHAP failed for {alg}: {e} : Skipping.")
            continue

        sv_bee = sv[:, bee_feat_idx]
        X_bee = X_display[[feat_labels[i] for i in bee_feat_idx]]

        with plt.rc_context({
            'font.family': 'Times New Roman', 'font.size': 10,
            'axes.titlesize': 10, 'axes.labelsize': 10,
            'xtick.labelsize': 10, 'ytick.labelsize': 10,
        }):
            shap.summary_plot(sv_bee, X_bee, feature_names=feat_bee,
                              show=False, plot_size=None)
            fig = plt.gcf()
            fig.set_size_inches(12 * cm_to_in, 8 * cm_to_in)
            ax = fig.axes[0]
            _style_beeswarm(fig, ax)
            ax.set_title(alg, fontsize=10)
            plt.tight_layout()
            fig.subplots_adjust(left=0.18)
            path = os.path.join(os.path.dirname(output_prefix), f'SHAP_Compare_{alg}_{timestamp}.tiff')
            fig.savefig(path, dpi=500, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved at: {path}.")