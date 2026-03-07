"""Generacion de figuras estadisticas."""

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = ['#3182ce', '#e53e3e', '#38a169', '#d69e2e', '#805ad5', '#dd6b20']


def generate_figure(fig_type, df, var_dep, var_group, groups=None, result=None):
    """Genera una figura segun el tipo seleccionado."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    fig, ax = plt.subplots(figsize=(8, 6))

    if groups:
        plot_df = df[df[var_group].isin(groups)].copy()
    else:
        plot_df = df.copy()

    try:
        if fig_type == 'boxplot':
            _plot_boxplot(ax, plot_df, var_dep, var_group)
        elif fig_type == 'violin':
            _plot_violin(ax, plot_df, var_dep, var_group)
        elif fig_type == 'bar_error':
            _plot_bar_error(ax, plot_df, var_dep, var_group, groups)
        elif fig_type == 'scatter':
            _plot_scatter(ax, plot_df, var_dep, var_group)
        elif fig_type == 'paired':
            _plot_paired(ax, plot_df, var_dep, var_group, groups)
        elif fig_type == 'histogram':
            _plot_histogram(ax, plot_df, var_dep, var_group, groups)

        _add_significance(ax, result)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center',
                transform=ax.transAxes)

    return fig


def _plot_boxplot(ax, df, var_dep, var_group):
    sns.boxplot(data=df, x=var_group, y=var_dep, palette=PALETTE,
                width=0.6, ax=ax, showfliers=True)
    sns.stripplot(data=df, x=var_group, y=var_dep, color='black',
                  alpha=0.4, size=4, jitter=True, ax=ax)
    ax.set_title(f'{var_dep} por {var_group}', fontweight='bold', pad=15)


def _plot_violin(ax, df, var_dep, var_group):
    sns.violinplot(data=df, x=var_group, y=var_dep, palette=PALETTE,
                   inner='box', ax=ax)
    sns.stripplot(data=df, x=var_group, y=var_dep, color='black',
                  alpha=0.3, size=3, jitter=True, ax=ax)
    ax.set_title(f'{var_dep} por {var_group}', fontweight='bold', pad=15)


def _plot_bar_error(ax, df, var_dep, var_group, groups):
    grouped = df.groupby(var_group)[var_dep]
    means = grouped.mean()
    sems = grouped.sem()
    order = groups if groups else means.index
    x_pos = range(len(order))
    ax.bar(x_pos, [means[g] for g in order],
           yerr=[sems[g] for g in order],
           capsize=5, color=PALETTE[:len(order)],
           edgecolor='black', linewidth=0.5, alpha=0.85, width=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(order)
    ax.set_ylabel(var_dep)
    ax.set_xlabel(var_group)
    ax.set_title(f'{var_dep} por {var_group} (media +/- SEM)', fontweight='bold', pad=15)


def _plot_scatter(ax, df, var_dep, var_group):
    ax.scatter(df[var_group], df[var_dep], alpha=0.6,
               color=PALETTE[0], edgecolors='white', s=60)
    clean = df[[var_dep, var_group]].dropna()
    if len(clean) > 2:
        z = np.polyfit(clean[var_group], clean[var_dep], 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(clean[var_group].min(), clean[var_group].max(), 100)
        ax.plot(x_line, p_line(x_line), '--', color=PALETTE[1], linewidth=2)
        r, pval = stats.pearsonr(clean[var_group], clean[var_dep])
        ax.text(0.05, 0.95, f'r = {r:.3f}, p = {pval:.4f}',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel(var_group)
    ax.set_ylabel(var_dep)
    ax.set_title(f'{var_dep} vs {var_group}', fontweight='bold', pad=15)


def _plot_paired(ax, df, var_dep, var_group, groups):
    unique_groups = groups if groups else sorted(df[var_group].dropna().unique())[:2]
    if len(unique_groups) >= 2:
        g1_data = df[df[var_group] == unique_groups[0]][var_dep].dropna().values
        g2_data = df[df[var_group] == unique_groups[1]][var_dep].dropna().values
        min_n = min(len(g1_data), len(g2_data))
        for i in range(min_n):
            color = PALETTE[1] if g2_data[i] > g1_data[i] else PALETTE[0]
            ax.plot([0, 1], [g1_data[i], g2_data[i]], '-o', color=color,
                    alpha=0.4, markersize=6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(unique_groups[:2])
        ax.set_ylabel(var_dep)
        ax.set_title(f'{var_dep}: {unique_groups[0]} -> {unique_groups[1]}',
                     fontweight='bold', pad=15)


def _plot_histogram(ax, df, var_dep, var_group, groups):
    if var_group and df[var_group].dtype == 'object':
        unique_groups = groups if groups else sorted(df[var_group].dropna().unique())
        for i, g in enumerate(unique_groups):
            gd = df[df[var_group] == g][var_dep].dropna()
            ax.hist(gd, bins='auto', alpha=0.6, color=PALETTE[i % len(PALETTE)],
                    label=str(g), edgecolor='white')
        ax.legend()
    else:
        ax.hist(df[var_dep].dropna(), bins='auto', alpha=0.7,
                color=PALETTE[0], edgecolor='white')
    ax.set_xlabel(var_dep)
    ax.set_ylabel('Frecuencia')
    ax.set_title(f'Distribucion de {var_dep}', fontweight='bold', pad=15)


def _add_significance(ax, result):
    if result and result.get('p_value') is not None:
        p = result['p_value']
        if p < 0.001:
            sig_text = '***'
        elif p < 0.01:
            sig_text = '**'
        elif p < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'
        ax.text(0.95, 0.95, f'p = {p:.4f} ({sig_text})', transform=ax.transAxes,
                fontsize=11, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
