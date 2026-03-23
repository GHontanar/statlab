"""Generación de figuras estadísticas con Plotly."""

import numpy as np
from scipy import stats
import plotly.graph_objects as go

PALETTE = ['#3182ce', '#e53e3e', '#38a169', '#d69e2e', '#805ad5', '#dd6b20']

# Tamaños en píxeles por tipo de gráfico
_FIGSIZE = {
    'boxplot': (800, 600),
    'violin': (800, 600),
    'bar_error': (800, 600),
    'scatter': (700, 700),
    'paired': (600, 700),
    'histogram': (800, 550),
    'bland_altman': (800, 600),
    'roc': (700, 700),
    'kaplan_meier': (800, 600),
}


def _figsize_for(fig_type, n_groups=None):
    """Ajusta figsize según tipo y número de grupos."""
    w, h = _FIGSIZE.get(fig_type, (800, 600))
    if n_groups and n_groups > 4 and fig_type in ('boxplot', 'violin', 'bar_error'):
        w = max(w, n_groups * 180)
    return (w, h)


def _get_palette(options=None):
    """Retorna la paleta a usar según opciones del usuario."""
    if options and options.get('palette'):
        pal = options['palette']
        if isinstance(pal, str):
            return _named_palette(pal)
        return pal
    return PALETTE


def _named_palette(name):
    """Convierte nombre de paleta seaborn/matplotlib a lista de colores hex."""
    _palettes = {
        'Blues': ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7'],
        'Reds': ['#a50f15', '#de2d26', '#fb6a4a', '#fc9272', '#fcbba1', '#fee0d2'],
        'Greens': ['#006d2c', '#31a354', '#74c476', '#a1d99b', '#c7e9c0', '#e5f5e0'],
        'pastel': ['#a1c9f4', '#ffb482', '#8de5a1', '#ff9f9b', '#d0bbff', '#debb9b'],
        'Set2': ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f'],
        'Greys': ['#252525', '#636363', '#969696', '#bdbdbd', '#d9d9d9', '#f0f0f0'],
    }
    return _palettes.get(name, PALETTE)


def _p_to_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'ns'


def _base_layout(title='', xlabel='', ylabel='', width=800, height=600):
    """Layout base compartido por todas las figuras."""
    return go.Layout(
        title=dict(text=title, font=dict(size=16, color='#1a365d')),
        xaxis=dict(title=xlabel, showgrid=True, gridcolor='#e2e8f0'),
        yaxis=dict(title=ylabel, showgrid=True, gridcolor='#e2e8f0'),
        plot_bgcolor='white',
        width=width,
        height=height,
        margin=dict(l=60, r=30, t=60, b=60),
        font=dict(family='Arial, Helvetica, sans-serif', size=12),
    )


def generate_figure(fig_type, df, var_dep, var_group, groups=None, result=None, options=None):
    """Genera una figura Plotly según el tipo seleccionado."""
    palette = _get_palette(options)

    n_groups = len(groups) if groups else None
    width, height = _figsize_for(fig_type, n_groups)

    if groups:
        plot_df = df[df[var_group].isin(groups)].copy()
    else:
        plot_df = df

    try:
        if fig_type == 'boxplot':
            fig = _plot_boxplot(plot_df, var_dep, var_group, palette, width, height)
        elif fig_type == 'violin':
            fig = _plot_violin(plot_df, var_dep, var_group, palette, width, height)
        elif fig_type == 'bar_error':
            fig = _plot_bar_error(plot_df, var_dep, var_group, groups, palette, width, height)
        elif fig_type == 'scatter':
            fig = _plot_scatter(plot_df, var_dep, var_group, palette, width, height)
        elif fig_type == 'paired':
            fig = _plot_paired(plot_df, var_dep, var_group, groups, palette, width, height)
        elif fig_type == 'histogram':
            fig = _plot_histogram(plot_df, var_dep, var_group, groups, palette, width, height)
        elif fig_type == 'bland_altman':
            fig = _plot_bland_altman(plot_df, var_dep, var_group, result, palette, width, height)
        elif fig_type == 'roc':
            fig = _plot_roc(result, palette, width, height)
        elif fig_type == 'kaplan_meier':
            fig = _plot_kaplan_meier(result, palette, width, height)
        else:
            fig = go.Figure(layout=_base_layout('Tipo no soportado', width=width, height=height))

        # Aplicar personalizaciones del usuario
        if options:
            if options.get('title'):
                fig.update_layout(title_text=options['title'])
            if options.get('xlabel'):
                fig.update_layout(xaxis_title=options['xlabel'])
            if options.get('ylabel'):
                fig.update_layout(yaxis_title=options['ylabel'])

        # Significancia
        if result and isinstance(result.get('p_value'), (int, float)):
            if fig_type in ('boxplot', 'violin', 'bar_error') and n_groups == 2:
                _add_bracket(fig, result)
            elif fig_type not in ('roc', 'kaplan_meier', 'bland_altman'):
                _add_significance_text(fig, result)

    except Exception as e:
        fig = go.Figure(layout=_base_layout(width=width, height=height))
        fig.add_annotation(x=0.5, y=0.5, text=f'Error: {str(e)}',
                           showarrow=False, xref='paper', yref='paper', font=dict(size=14))

    return fig


def _plot_boxplot(df, var_dep, var_group, palette, width, height):
    unique_groups = sorted(df[var_group].dropna().unique())
    fig = go.Figure(layout=_base_layout(
        f'{var_dep} por {var_group}', var_group, var_dep, width, height))

    for i, g in enumerate(unique_groups):
        gdata = df[df[var_group] == g][var_dep].dropna()
        color = palette[i % len(palette)]
        fig.add_trace(go.Box(
            y=gdata, name=str(g), marker_color=color,
            boxmean=False, boxpoints=False,
            line=dict(width=1.5),
        ))
        # Puntos individuales
        fig.add_trace(go.Scatter(
            x=[str(g)] * len(gdata),
            y=gdata,
            mode='markers',
            marker=dict(color='#2d3748', size=5, opacity=0.6),
            showlegend=False,
        ))

    fig.update_layout(showlegend=False)
    return fig


def _plot_violin(df, var_dep, var_group, palette, width, height):
    unique_groups = sorted(df[var_group].dropna().unique())
    fig = go.Figure(layout=_base_layout(
        f'{var_dep} por {var_group}', var_group, var_dep, width, height))

    for i, g in enumerate(unique_groups):
        gdata = df[df[var_group] == g][var_dep].dropna()
        color = palette[i % len(palette)]
        fig.add_trace(go.Violin(
            y=gdata, name=str(g), fillcolor=color,
            line_color=color, opacity=0.7,
            box_visible=True, meanline_visible=True,
            points='all', pointpos=0, jitter=0.3,
            marker=dict(color='#2d3748', size=4, opacity=0.5),
        ))

    fig.update_layout(showlegend=False)
    return fig


def _plot_bar_error(df, var_dep, var_group, groups, palette, width, height):
    grouped = df.groupby(var_group)[var_dep]
    means = grouped.mean()
    sems = grouped.sem()
    order = groups if groups else sorted(means.index)

    fig = go.Figure(layout=_base_layout(
        f'{var_dep} por {var_group} (media ± SEM)', var_group, var_dep, width, height))

    colors = [palette[i % len(palette)] for i in range(len(order))]
    fig.add_trace(go.Bar(
        x=[str(g) for g in order],
        y=[means[g] for g in order],
        error_y=dict(type='data', array=[sems[g] for g in order], visible=True),
        marker_color=colors, marker_line=dict(color='black', width=0.5),
        opacity=0.85,
    ))

    # Puntos individuales
    for i, g in enumerate(order):
        vals = df[df[var_group] == g][var_dep].dropna()
        fig.add_trace(go.Scatter(
            x=[str(g)] * len(vals), y=vals,
            mode='markers',
            marker=dict(color='#2d3748', size=5, opacity=0.5),
            showlegend=False,
        ))

    fig.update_layout(showlegend=False)
    return fig


def _plot_scatter(df, var_dep, var_group, palette, width, height):
    color = palette[0]
    fig = go.Figure(layout=_base_layout(
        f'{var_dep} vs {var_group}', var_group, var_dep, width, height))

    fig.add_trace(go.Scatter(
        x=df[var_group], y=df[var_dep],
        mode='markers',
        marker=dict(color=color, size=8, opacity=0.65,
                    line=dict(color='#2d3748', width=0.5)),
        showlegend=False,
    ))

    clean = df[[var_dep, var_group]].dropna()
    if len(clean) > 2:
        z = np.polyfit(clean[var_group], clean[var_dep], 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(clean[var_group].min(), clean[var_group].max(), 100)
        fig.add_trace(go.Scatter(
            x=x_line, y=p_line(x_line),
            mode='lines', line=dict(color=palette[1], width=2, dash='dash'),
            showlegend=False,
        ))

        r, pval = stats.pearsonr(clean[var_group], clean[var_dep])
        fig.add_annotation(
            x=0.05, y=0.95, xref='paper', yref='paper',
            text=f'r = {r:.3f}, p = {pval:.4f}',
            showarrow=False, font=dict(size=11),
            bgcolor='wheat', opacity=0.8, bordercolor='#999', borderwidth=1,
        )

    return fig


def _plot_paired(df, var_dep, var_group, groups, palette, width, height):
    unique_groups = groups if groups else sorted(df[var_group].dropna().unique())[:2]
    fig = go.Figure(layout=_base_layout(
        f'{var_dep}: {unique_groups[0]} → {unique_groups[1]}' if len(unique_groups) >= 2 else var_dep,
        '', var_dep, width, height))

    if len(unique_groups) >= 2:
        g1_data = df[df[var_group] == unique_groups[0]][var_dep].dropna().values
        g2_data = df[df[var_group] == unique_groups[1]][var_dep].dropna().values
        min_n = min(len(g1_data), len(g2_data))
        c_up, c_down = palette[1], palette[0]

        for i in range(min_n):
            color = c_up if g2_data[i] > g1_data[i] else c_down
            fig.add_trace(go.Scatter(
                x=[str(unique_groups[0]), str(unique_groups[1])],
                y=[g1_data[i], g2_data[i]],
                mode='lines+markers',
                line=dict(color=color, width=1.2),
                marker=dict(size=7, color=color, opacity=0.55),
                showlegend=False,
            ))

        mean1 = np.mean(g1_data[:min_n])
        mean2 = np.mean(g2_data[:min_n])
        fig.add_trace(go.Scatter(
            x=[str(unique_groups[0]), str(unique_groups[1])],
            y=[mean1, mean2],
            mode='lines+markers', name='Media',
            line=dict(color='black', width=2.5),
            marker=dict(size=10, color='black', symbol='square'),
        ))

    return fig


def _plot_histogram(df, var_dep, var_group, groups, palette, width, height):
    fig = go.Figure(layout=_base_layout(
        f'Distribución de {var_dep}', var_dep, 'Frecuencia', width, height))

    if var_group and df[var_group].dtype == 'object':
        unique_groups = groups if groups else sorted(df[var_group].dropna().unique())
        for i, g in enumerate(unique_groups):
            gd = df[df[var_group] == g][var_dep].dropna()
            fig.add_trace(go.Histogram(
                x=gd, name=str(g),
                marker_color=palette[i % len(palette)],
                opacity=0.6,
            ))
        fig.update_layout(barmode='overlay')
    else:
        fig.add_trace(go.Histogram(
            x=df[var_dep].dropna(),
            marker_color=palette[0], opacity=0.7,
        ))

    return fig


def _plot_bland_altman(df, var_dep, var_group, result, palette, width, height):
    clean = df[[var_dep, var_group]].dropna()
    m1, m2 = clean[var_dep].values, clean[var_group].values
    mean_vals = (m1 + m2) / 2
    diff_vals = m1 - m2

    bias = result['bias'] if result and 'bias' in result else np.mean(diff_vals)
    sd = result['sd_diff'] if result and 'sd_diff' in result else np.std(diff_vals, ddof=1)

    fig = go.Figure(layout=_base_layout(
        'Bland-Altman: concordancia entre métodos',
        f'Media de {var_dep} y {var_group}',
        f'Diferencia ({var_dep} − {var_group})',
        width, height))

    fig.add_trace(go.Scatter(
        x=mean_vals, y=diff_vals, mode='markers',
        marker=dict(color=palette[0], size=8, opacity=0.65,
                    line=dict(color='#2d3748', width=0.5)),
        showlegend=False,
    ))

    x_range = [min(mean_vals), max(mean_vals)]
    # Sesgo
    fig.add_shape(type='line', x0=x_range[0], x1=x_range[1], y0=bias, y1=bias,
                  line=dict(color=palette[1], width=1.5))
    # Límites de acuerdo
    for y_val, label in [(bias + 1.96 * sd, '+1.96 DE'), (bias - 1.96 * sd, '-1.96 DE')]:
        fig.add_shape(type='line', x0=x_range[0], x1=x_range[1], y0=y_val, y1=y_val,
                      line=dict(color='#718096', width=1, dash='dash'))
    # Línea cero
    fig.add_shape(type='line', x0=x_range[0], x1=x_range[1], y0=0, y1=0,
                  line=dict(color='black', width=0.5, dash='dot'))

    # Anotaciones
    fig.add_annotation(x=x_range[1], y=bias, text=f'Sesgo: {bias:.3f}',
                       showarrow=False, xanchor='left', font=dict(size=9, color=palette[1]))
    fig.add_annotation(x=x_range[1], y=bias + 1.96 * sd,
                       text=f'+1.96 DE: {bias + 1.96 * sd:.3f}',
                       showarrow=False, xanchor='left', font=dict(size=9, color='#718096'))
    fig.add_annotation(x=x_range[1], y=bias - 1.96 * sd,
                       text=f'-1.96 DE: {bias - 1.96 * sd:.3f}',
                       showarrow=False, xanchor='left', font=dict(size=9, color='#718096'))

    return fig


def _plot_roc(result, palette, width, height):
    fig = go.Figure(layout=_base_layout(
        'Curva ROC', '1 − Especificidad (FPR)', 'Sensibilidad (TPR)', width, height))

    if not result or 'fpr' not in result:
        fig.add_annotation(x=0.5, y=0.5, text='Ejecuta el análisis ROC primero',
                           showarrow=False, xref='paper', yref='paper', font=dict(size=14))
        return fig

    fpr = result['fpr']
    tpr = result['tpr']
    auc = result.get('auc', 0)

    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines',
        line=dict(color=palette[0], width=2),
        name=f'AUC = {auc:.3f}',
    ))

    # Diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        line=dict(color='#a0aec0', width=1, dash='dash'),
        showlegend=False,
    ))

    # Punto de corte óptimo
    if result.get('sensitivity') is not None:
        sens = result['sensitivity']
        spec = result['specificity']
        fig.add_trace(go.Scatter(
            x=[1 - spec], y=[sens], mode='markers',
            marker=dict(color=palette[1], size=10),
            name=f'Corte = {result["best_threshold"]:.2f} '
                 f'(Sens={sens:.2f}, Esp={spec:.2f})',
        ))

    fig.update_layout(
        xaxis=dict(range=[-0.02, 1.02]),
        yaxis=dict(range=[-0.02, 1.02]),
        legend=dict(x=0.55, y=0.05),
    )
    return fig


def _plot_kaplan_meier(result, palette, width, height):
    fig = go.Figure(layout=_base_layout(
        'Kaplan-Meier', 'Tiempo', 'Probabilidad de supervivencia', width, height))

    if not result or 'curves' not in result:
        fig.add_annotation(x=0.5, y=0.5, text='Ejecuta el análisis Kaplan-Meier primero',
                           showarrow=False, xref='paper', yref='paper', font=dict(size=14))
        return fig

    curves = result['curves']
    for i, (label, data) in enumerate(curves.items()):
        color = palette[i % len(palette)]
        timeline = data['timeline']
        survival = data['survival']
        median = data.get('median')
        lbl = f'{label} (n={data["n"]})'
        if median is not None:
            lbl += f', med={median:.1f}'

        fig.add_trace(go.Scatter(
            x=timeline, y=survival,
            mode='lines', line=dict(color=color, width=2, shape='hv'),
            name=lbl,
        ))

    fig.update_layout(
        yaxis=dict(range=[-0.02, 1.05]),
        legend=dict(x=0.01, y=0.05, bgcolor='rgba(255,255,255,0.8)'),
    )

    if isinstance(result.get('p_value'), (int, float)):
        p = result['p_value']
        fig.add_annotation(
            x=0.95, y=0.95, xref='paper', yref='paper',
            text=f'Log-rank p = {p:.4f}',
            showarrow=False, font=dict(size=11),
            bgcolor='lightyellow', bordercolor='#999', borderwidth=1,
        )

    return fig


def _add_bracket(fig, result):
    """Bracket de significancia entre 2 grupos."""
    p = result['p_value']
    stars = _p_to_stars(p)

    # Calcular posición vertical
    y_range = fig.layout.yaxis.range
    if y_range:
        y_max = y_range[1]
    else:
        # Estimar del data
        all_y = []
        for trace in fig.data:
            if hasattr(trace, 'y') and trace.y is not None:
                all_y.extend([v for v in trace.y if v is not None and isinstance(v, (int, float))])
        y_max = max(all_y) if all_y else 1

    bar_y = y_max * 1.05
    tip = y_max * 0.02

    fig.add_shape(type='line', x0=0, x1=0, y0=bar_y - tip, y1=bar_y,
                  line=dict(color='black', width=1.2))
    fig.add_shape(type='line', x0=0, x1=1, y0=bar_y, y1=bar_y,
                  line=dict(color='black', width=1.2))
    fig.add_shape(type='line', x0=1, x1=1, y0=bar_y - tip, y1=bar_y,
                  line=dict(color='black', width=1.2))
    fig.add_annotation(x=0.5, y=bar_y + tip, text=f'{stars}<br>p = {p:.4f}',
                       showarrow=False, font=dict(size=11))
    fig.update_layout(yaxis=dict(range=[None, bar_y * 1.1]))


def _add_significance_text(fig, result):
    """Texto flotante con p-valor."""
    p = result['p_value']
    stars = _p_to_stars(p)
    fig.add_annotation(
        x=0.95, y=0.95, xref='paper', yref='paper',
        text=f'p = {p:.4f} ({stars})',
        showarrow=False, font=dict(size=11),
        bgcolor='lightyellow', bordercolor='#999', borderwidth=1,
    )
