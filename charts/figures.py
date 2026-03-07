"""Generacion de figuras estadisticas."""

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

PALETTE = ['#3182ce', '#e53e3e', '#38a169', '#d69e2e', '#805ad5', '#dd6b20']

# G7: Fuente profesional con fallback
_PREFERRED_FONTS = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']
_FONT_FAMILY = 'DejaVu Sans'
for font in _PREFERRED_FONTS:
    if any(font.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        _FONT_FAMILY = font
        break

# G3: Tamanos adaptativos por tipo de grafico
_FIGSIZE = {
    'boxplot': (8, 6),
    'violin': (8, 6),
    'bar_error': (8, 6),
    'scatter': (7, 7),
    'paired': (6, 7),
    'histogram': (8, 5.5),
}


def _figsize_for(fig_type, n_groups=None):
    """Ajusta figsize segun tipo y numero de grupos."""
    w, h = _FIGSIZE.get(fig_type, (8, 6))
    if n_groups and n_groups > 4 and fig_type in ('boxplot', 'violin', 'bar_error'):
        w = max(w, n_groups * 1.8)
    return (w, h)


def _apply_style():
    """G2/G7: Configura estilo global para figuras de calidad publicacion."""
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': [_FONT_FAMILY],
        'axes.titleweight': 'bold',
        'axes.titlepad': 15,
        'figure.dpi': 150,
        'savefig.dpi': 300,
    })


def _get_palette(options=None):
    """Retorna la paleta a usar segun opciones del usuario."""
    if options and options.get('palette'):
        return options['palette']
    return PALETTE


def generate_figure(fig_type, df, var_dep, var_group, groups=None, result=None, options=None):
    """Genera una figura segun el tipo seleccionado."""
    _apply_style()
    palette = _get_palette(options)

    n_groups = len(groups) if groups else None
    figsize = _figsize_for(fig_type, n_groups)
    fig, ax = plt.subplots(figsize=figsize)

    if groups:
        plot_df = df[df[var_group].isin(groups)].copy()
    else:
        plot_df = df.copy()

    try:
        if fig_type == 'boxplot':
            _plot_boxplot(ax, plot_df, var_dep, var_group, palette)
        elif fig_type == 'violin':
            _plot_violin(ax, plot_df, var_dep, var_group, palette)
        elif fig_type == 'bar_error':
            _plot_bar_error(ax, plot_df, var_dep, var_group, groups, palette)
        elif fig_type == 'scatter':
            _plot_scatter(ax, plot_df, var_dep, var_group, palette)
        elif fig_type == 'paired':
            _plot_paired(ax, plot_df, var_dep, var_group, groups, palette)
        elif fig_type == 'histogram':
            _plot_histogram(ax, plot_df, var_dep, var_group, groups, palette)

        # G8: Aplicar personalizaciones del usuario
        if options:
            if options.get('title'):
                ax.set_title(options['title'], fontweight='bold', pad=15)
            if options.get('xlabel'):
                ax.set_xlabel(options['xlabel'])
            if options.get('ylabel'):
                ax.set_ylabel(options['ylabel'])

        # G4: Brackets de significancia para comparacion de 2 grupos
        if result and result.get('p_value') is not None:
            if fig_type in ('boxplot', 'violin', 'bar_error') and n_groups == 2:
                _add_bracket(ax, result)
            else:
                _add_significance_text(ax, result)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()

    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center',
                transform=ax.transAxes)

    return fig


def _plot_boxplot(ax, df, var_dep, var_group, palette=PALETTE):
    n = df[var_group].nunique()
    pal = palette[:n] if isinstance(palette, list) else palette
    sns.boxplot(data=df, x=var_group, y=var_dep, hue=var_group,
                palette=pal, width=0.6, ax=ax, showfliers=False, legend=False)
    # G5: Puntos mas visibles
    sns.stripplot(data=df, x=var_group, y=var_dep, color='#2d3748',
                  alpha=0.6, size=5, jitter=True, ax=ax)
    ax.set_title(f'{var_dep} por {var_group}')


def _plot_violin(ax, df, var_dep, var_group, palette=PALETTE):
    n = df[var_group].nunique()
    pal = palette[:n] if isinstance(palette, list) else palette
    sns.violinplot(data=df, x=var_group, y=var_dep, hue=var_group,
                   palette=pal, inner='box', ax=ax, legend=False)
    sns.stripplot(data=df, x=var_group, y=var_dep, color='#2d3748',
                  alpha=0.5, size=4, jitter=True, ax=ax)
    ax.set_title(f'{var_dep} por {var_group}')


def _plot_bar_error(ax, df, var_dep, var_group, groups, palette=PALETTE):
    grouped = df.groupby(var_group)[var_dep]
    means = grouped.mean()
    sems = grouped.sem()
    order = groups if groups else means.index
    x_pos = list(range(len(order)))
    if isinstance(palette, list):
        colors = palette[:len(order)]
    else:
        colors = sns.color_palette(palette, len(order))
    ax.bar(x_pos, [means[g] for g in order],
           yerr=[sems[g] for g in order],
           capsize=5, color=colors,
           edgecolor='black', linewidth=0.5, alpha=0.85, width=0.6)
    # G5: Puntos individuales sobre barras
    for i, g in enumerate(order):
        vals = df[df[var_group] == g][var_dep].dropna()
        ax.scatter([i] * len(vals), vals, color='#2d3748', alpha=0.5,
                   s=20, zorder=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(order)
    ax.set_ylabel(var_dep)
    ax.set_xlabel(var_group)
    ax.set_title(f'{var_dep} por {var_group} (media +/- SEM)')


def _plot_scatter(ax, df, var_dep, var_group, palette=PALETTE):
    color = palette[0] if isinstance(palette, list) else sns.color_palette(palette, 1)[0]
    ax.scatter(df[var_group], df[var_dep], alpha=0.65,
               color=color, edgecolors='#2d3748', linewidths=0.5, s=60)
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
    ax.set_title(f'{var_dep} vs {var_group}')


def _plot_paired(ax, df, var_dep, var_group, groups, palette=PALETTE):
    unique_groups = groups if groups else sorted(df[var_group].dropna().unique())[:2]
    if len(unique_groups) >= 2:
        g1_data = df[df[var_group] == unique_groups[0]][var_dep].dropna().values
        g2_data = df[df[var_group] == unique_groups[1]][var_dep].dropna().values
        min_n = min(len(g1_data), len(g2_data))
        if isinstance(palette, list):
            c_up, c_down = palette[1], palette[0]
        else:
            pal = sns.color_palette(palette, 2)
            c_up, c_down = pal[1], pal[0]
        for i in range(min_n):
            color = c_up if g2_data[i] > g1_data[i] else c_down
            ax.plot([0, 1], [g1_data[i], g2_data[i]], '-o', color=color,
                    alpha=0.55, markersize=7, linewidth=1.2)
        # G6: Marcadores de media por grupo
        mean1 = np.mean(g1_data[:min_n])
        mean2 = np.mean(g2_data[:min_n])
        ax.plot([0, 1], [mean1, mean2], '-s', color='black',
                markersize=10, linewidth=2.5, zorder=5, label='Media')
        ax.legend(loc='upper right', framealpha=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(unique_groups[:2])
        ax.set_ylabel(var_dep)
        ax.set_title(f'{var_dep}: {unique_groups[0]} -> {unique_groups[1]}')


def _plot_histogram(ax, df, var_dep, var_group, groups, palette=PALETTE):
    if var_group and df[var_group].dtype == 'object':
        unique_groups = groups if groups else sorted(df[var_group].dropna().unique())
        if isinstance(palette, list):
            colors = palette
        else:
            colors = sns.color_palette(palette, len(unique_groups))
        for i, g in enumerate(unique_groups):
            gd = df[df[var_group] == g][var_dep].dropna()
            ax.hist(gd, bins='auto', alpha=0.6,
                    color=colors[i % len(colors)],
                    label=str(g), edgecolor='white')
        ax.legend(framealpha=0.8)
    else:
        color = palette[0] if isinstance(palette, list) else sns.color_palette(palette, 1)[0]
        ax.hist(df[var_dep].dropna(), bins='auto', alpha=0.7,
                color=color, edgecolor='white')
    ax.set_xlabel(var_dep)
    ax.set_ylabel('Frecuencia')
    ax.set_title(f'Distribucion de {var_dep}')


def _p_to_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'ns'


def _add_bracket(ax, result):
    """G4: Bracket de significancia entre 2 grupos."""
    p = result['p_value']
    stars = _p_to_stars(p)

    # Posicion vertical: encima del dato mas alto
    y_max = ax.get_ylim()[1]
    y_range = y_max - ax.get_ylim()[0]
    bar_y = y_max - y_range * 0.02
    tip_len = y_range * 0.02

    ax.plot([0, 0, 1, 1], [bar_y - tip_len, bar_y, bar_y, bar_y - tip_len],
            color='black', linewidth=1.2)
    ax.text(0.5, bar_y + tip_len * 0.5, f'{stars}\np = {p:.4f}',
            ha='center', va='bottom', fontsize=10)
    ax.set_ylim(top=bar_y + y_range * 0.12)


def _add_significance_text(ax, result):
    """Texto flotante con p-valor para >2 grupos o scatter."""
    p = result['p_value']
    stars = _p_to_stars(p)
    ax.text(0.95, 0.95, f'p = {p:.4f} ({stars})', transform=ax.transAxes,
            fontsize=10, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
