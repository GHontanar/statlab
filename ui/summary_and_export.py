"""Tabla resumen, informe TXT/PDF e historial de análisis."""

from io import BytesIO

import streamlit as st
import pandas as pd

from reports.text import format_result_text
from reports.pdf import generate_pdf_report
from ui.figures_section import close_figures


def _build_summary_df(valid_results):
    """Construye el DataFrame resumen a partir de resultados válidos."""
    rows = []
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

        rows.append(row)
    return pd.DataFrame(rows)


def _summary_to_excel(summary_df):
    """Convierte el DataFrame resumen a bytes XLSX."""
    buf = BytesIO()
    summary_df.to_excel(buf, index=False, sheet_name="Resumen", engine="openpyxl")
    buf.seek(0)
    return buf.getvalue()


def render_summary_and_export():
    """Renderiza resumen, exportación e historial. Lee de st.session_state."""
    valid_results = [r for r in st.session_state.results if r.get('success')]

    if len(valid_results) > 1:
        st.header("5. Resumen de análisis")
        summary_df = _build_summary_df(valid_results)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar resumen Excel",
            data=_summary_to_excel(summary_df),
            file_name="statlab_resumen.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

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
                cols = st.columns([10, 1])
                p_val = r.get('p_value')
                with cols[0]:
                    if isinstance(p_val, float):
                        st.markdown(f"**{i+1}. {r.get('test_name', 'N/A')}** — "
                                    f"{r['var_dep']} × {r['var_group']} — "
                                    f"p = {p_val:.4f}")
                    else:
                        st.markdown(f"**{i+1}.** Error")
                with cols[1]:
                    if st.button("✕", key=f"del_{i}", help="Eliminar este resultado"):
                        _remove_result(i, valid_results)
                        st.rerun()
            if st.button("Limpiar historial", type="secondary"):
                close_figures()
                st.session_state.results = []
                st.session_state.figures = []
                st.rerun()


def _remove_result(valid_index, valid_results):
    """Elimina un resultado y su figura asociada del historial."""
    target = valid_results[valid_index]
    real_index = st.session_state.results.index(target)
    st.session_state.results.pop(real_index)
    if real_index < len(st.session_state.figures):
        fig = st.session_state.figures.pop(real_index)
        if hasattr(fig, 'close'):
            import matplotlib.pyplot as plt
            plt.close(fig)
