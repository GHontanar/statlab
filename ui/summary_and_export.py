"""Tabla resumen, informe TXT/PDF e historial de análisis."""

import streamlit as st
import pandas as pd

from reports.text import format_result_text
from reports.pdf import generate_pdf_report
from ui.figures_section import close_figures


def render_summary_and_export():
    """Renderiza resumen, exportación e historial. Lee de st.session_state."""
    valid_results = [r for r in st.session_state.results if r.get('success')]

    if len(valid_results) > 1:
        st.header("5. Resumen de análisis")
        _summary_rows = []
        for r in valid_results:
            row = {
                'Test': r.get('test_name', 'N/A'),
                'Variable': r.get('var_dep', ''),
                'Grupo/Predictor': r.get('var_group', ''),
            }
            p = r.get('p_value')
            row['p-valor'] = f"{p:.4f}" if isinstance(p, (int, float)) else "—"
            sig = r.get('significant')
            row['Sig.'] = "Sí" if sig is True else "No" if sig is False else "—"

            if r.get('ci_lower') is not None:
                row['IC 95%'] = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
            else:
                row['IC 95%'] = "—"

            if r.get('cohens_d') is not None:
                row['Efecto'] = f"d={r['cohens_d']:.2f}"
            elif r.get('eta_squared') is not None:
                row['Efecto'] = f"η²={r['eta_squared']:.3f}"
            elif r.get('r_squared') is not None:
                row['Efecto'] = f"R²={r['r_squared']:.3f}"
            elif r.get('auc') is not None:
                row['Efecto'] = f"AUC={r['auc']:.3f}"
            else:
                row['Efecto'] = "—"

            _summary_rows.append(row)

        st.dataframe(pd.DataFrame(_summary_rows), use_container_width=True, hide_index=True)

    st.header("6. Descargar informe" if len(valid_results) > 1 else "5. Descargar informe")

    if valid_results:
        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            full_text = "STATLAB — INFORME DE ANÁLISIS ESTADÍSTICO\n"
            full_text += f"{'='*60}\n\n"
            for r in valid_results:
                full_text += format_result_text(r) + "\n"

            st.download_button(
                "Informe TXT",
                data=full_text,
                file_name="statlab_informe.txt",
                mime="text/plain",
                use_container_width=True
            )

        with col_dl2:
            if st.button("Generar informe PDF", use_container_width=True):
                progress = st.progress(0, text="Preparando informe...")
                progress.progress(10, text="Generando resultados...")
                pdf_buf = generate_pdf_report(valid_results, st.session_state.figures)
                progress.progress(90, text="Finalizando...")
                st.download_button(
                    "Descargar PDF",
                    data=pdf_buf,
                    file_name="statlab_informe.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                progress.progress(100, text="Listo")
    else:
        st.info("Ejecuta al menos un análisis para poder generar el informe.")

    if st.session_state.results:
        with st.expander(f"Historial de análisis ({len(valid_results)} realizados)"):
            for i, r in enumerate(valid_results):
                p_val = r.get('p_value')
                if isinstance(p_val, float):
                    st.markdown(f"**{i+1}. {r.get('test_name', 'N/A')}** — "
                                f"{r['var_dep']} × {r['var_group']} — "
                                f"p = {p_val:.4f}")
                else:
                    st.markdown(f"**{i+1}.** Error")
            if st.button("Limpiar historial", type="secondary"):
                close_figures()
                st.session_state.results = []
                st.session_state.figures = []
                st.rerun()
