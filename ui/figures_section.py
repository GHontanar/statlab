"""Generación manual de figuras y descarga PNG/SVG."""

from io import BytesIO

import streamlit as st

from ui.constants import Q_DIFF_GROUPS, AVAILABLE_FIGURES, DEFAULT_FIGURES, AnalysisConfig
from charts.figures import generate_figure


def close_figures():
    """Limpia la lista de figuras almacenadas en session_state."""
    st.session_state['figures'] = []


def render_figures_section(config, df, fig_options):
    """Renderiza la sección de generación manual de figuras.

    Args:
        config: AnalysisConfig con el tipo de análisis y variables.
        df: DataFrame con los datos.
        fig_options: dict con opciones de personalización.
    """
    st.header("4. Figuras")

    available_figs = AVAILABLE_FIGURES.get(config.analysis_type, DEFAULT_FIGURES)

    selected_figs = st.multiselect("Selecciona las figuras a generar",
                                   list(available_figs.keys()))

    if selected_figs and st.button("Generar figuras", use_container_width=True):
        close_figures()
        last_result = st.session_state.results[-1] if st.session_state.results else None

        for fig_name in selected_figs:
            fig_type = available_figs[fig_name]
            fig = generate_figure(
                fig_type, df, config.var_dep, config.var_group,
                config.selected_groups if config.analysis_type == Q_DIFF_GROUPS else None,
                last_result,
                options=fig_options,
            )
            st.session_state.figures.append(fig)
            st.plotly_chart(fig, use_container_width=True)

            # Descarga en PNG y SVG
            dl_cols = st.columns(2)
            with dl_cols[0]:
                buf_png = fig.to_image(format='png', scale=2)
                st.download_button(
                    f"PNG 300dpi — {fig_name}",
                    data=buf_png,
                    file_name=f"statlab_{fig_type}.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with dl_cols[1]:
                buf_svg = fig.to_image(format='svg')
                st.download_button(
                    f"SVG — {fig_name}",
                    data=buf_svg,
                    file_name=f"statlab_{fig_type}.svg",
                    mime="image/svg+xml",
                    use_container_width=True,
                )
