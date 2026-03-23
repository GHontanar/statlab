"""Ejecución del análisis, métricas, interpretación y auto-figura."""

import pandas as pd
import streamlit as st

from charts.figures import generate_figure
from reports.text import format_result_text, generate_interpretation
from stats.tests import run_test
from ui.constants import AUTO_FIGURE_MAP, Q_DIFF_GROUPS


def render_results(config, df, fig_options):
    """Ejecuta el test y muestra resultados.

    Args:
        config: AnalysisConfig con los parámetros del análisis.
        df: DataFrame con los datos.
        fig_options: dict con opciones de personalización de figuras.
    """
    if st.button("Ejecutar análisis", type="primary", use_container_width=True):
        with st.spinner("Calculando..."):
            extra = config.extra or None
            result = run_test(config.selected_test_id, df, config.var_dep, config.var_group,
                              config.selected_groups, config.alpha, paired_id_col=config.paired_id_col,
                              extra=extra)
            st.session_state.results.append(result)

        if result.get('success'):
            _display_result(result, config, df, fig_options)
        else:
            st.error(f"Error en el análisis: {result.get('error', 'Desconocido')}")


def _display_result(result, config, df, fig_options):
    """Muestra los resultados de un análisis exitoso."""
    st.subheader("Resultados")

    if result.get('groups') and isinstance(result.get('n'), list):
        desc_data = {'Grupo': result['groups'], 'n': result['n']}
        if result.get('mean') and isinstance(result['mean'], list):
            desc_data['Media'] = [f"{m:.2f}" for m in result['mean']]
            desc_data['DE'] = [f"{s:.2f}" for s in result['std']]
        if result.get('median') and isinstance(result['median'], list):
            desc_data['Mediana'] = [f"{m:.2f}" for m in result['median']]
        st.dataframe(pd.DataFrame(desc_data).set_index('Grupo'),
                     use_container_width=True)

    rcols = st.columns(3)
    with rcols[0]:
        stat_val = result.get('statistic')
        if isinstance(stat_val, (int, float)):
            st.metric("Estadístico", f"{stat_val:.4f}")
        else:
            st.metric("Estadístico", "N/A")
    with rcols[1]:
        p = result.get('p_value')
        if isinstance(p, (int, float)):
            st.metric("p-valor", f"{p:.6f}")
        else:
            st.metric("p-valor", "N/A")
    with rcols[2]:
        sig = result.get('significant')
        if sig is True:
            st.markdown("### Significativo")
        elif sig is False:
            st.markdown("### No significativo")
        else:
            st.markdown("### —")

    if result.get('warning'):
        st.warning(result['warning'])
    if result.get('posthoc_error'):
        st.warning(result['posthoc_error'])

    if result.get('ci_lower') is not None and result.get('ci_upper') is not None:
        st.info(f"IC 95%: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")

    if result.get('cohens_d'):
        st.info(f"Tamaño del efecto: d de Cohen = {result['cohens_d']:.3f}")
    if result.get('eta_squared'):
        st.info(f"Tamaño del efecto: η² = {result['eta_squared']:.3f}")
    if result.get('r_squared') is not None:
        st.info(f"R² = {result['r_squared']:.3f}")
    if result.get('auc') is not None:
        st.info(f"AUC = {result['auc']:.3f}")
        if result.get('best_threshold') is not None:
            st.info(f"Corte óptimo: {result['best_threshold']:.3f} "
                    f"(Sens={result['sensitivity']:.3f}, "
                    f"Esp={result['specificity']:.3f})")
    if result.get('bias') is not None:
        st.info(f"Sesgo = {result['bias']:.3f}, "
                f"Límites de acuerdo: [{result['loa_lower']:.3f}, {result['loa_upper']:.3f}]")
    if result.get('curves'):
        with st.expander("Curvas de supervivencia", expanded=True):
            for label, data in result['curves'].items():
                med = data.get('median')
                med_str = f"{med:.1f}" if med is not None else "no alcanzada"
                st.markdown(f"**{label}**: n={data['n']}, mediana={med_str}")

    if result.get('odds_ratio') is not None:
        or_val = result['odds_ratio']
        st.info(f"Odds Ratio = {or_val:.3f} "
                f"(IC 95%: [{result['or_ci_lower']:.3f}, {result['or_ci_upper']:.3f}])")
        if result.get('pseudo_r2') is not None:
            st.info(f"Pseudo R² (McFadden) = {result['pseudo_r2']:.3f}")

    if result.get('icc') is not None:
        st.info(f"ICC = {result['icc']:.3f} ({result.get('quality', '')}), "
                f"{result.get('n_raters', 0)} evaluadores, "
                f"{result.get('n_subjects', 0)} sujetos")

    if result.get('power'):
        pw = result['power']
        power_pct = pw['power'] * 100
        if pw['power'] >= 0.8:
            st.success(f"Potencia estadística: {power_pct:.0f}% (adecuada). "
                       f"n por grupo para 80%: {pw.get('n_for_80', '—')}")
        elif pw['power'] >= 0.5:
            st.warning(f"Potencia estadística: {power_pct:.0f}% (baja). "
                       f"n por grupo para 80%: {pw.get('n_for_80', '—')}")
        else:
            st.error(f"Potencia estadística: {power_pct:.0f}% (insuficiente). "
                     f"n por grupo para 80%: {pw.get('n_for_80', '—')}")

    interpretation = generate_interpretation(result)
    if interpretation:
        st.markdown("**Para tu publicación:**")
        st.code(interpretation, language=None)

    _auto_fig_type = AUTO_FIGURE_MAP.get(config.analysis_type)
    if _auto_fig_type:
        _auto_groups = config.selected_groups if config.analysis_type == Q_DIFF_GROUPS else None
        auto_fig = generate_figure(
            _auto_fig_type, df, config.var_dep, config.var_group,
            _auto_groups, result, options=fig_options)
        st.plotly_chart(auto_fig, use_container_width=True)
        st.session_state.figures.append(auto_fig)

    with st.expander("Detalle completo"):
        st.text(format_result_text(result))

    if result.get('contingency_table'):
        with st.expander("Tabla de contingencia", expanded=True):
            ct_df = pd.DataFrame(result['contingency_table'])
            st.dataframe(ct_df, use_container_width=True)

    if result.get('posthoc'):
        with st.expander(f"Post-hoc: {result['posthoc_name']}", expanded=True):
            ph_df = pd.DataFrame(result['posthoc'])
            st.dataframe(ph_df.round(4), use_container_width=True)
