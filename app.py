"""
StatLab - Análisis Estadístico y Generación de Figuras
"""

import warnings

import streamlit as st

warnings.filterwarnings('ignore', category=FutureWarning)

from ui.analysis_config import render_analysis_config
from ui.data_upload import render_data_preview, render_data_upload
from ui.figures_section import render_figures_section
from ui.landing import render_landing
from ui.results_display import render_results
from ui.sample_size import render_sample_size
from ui.sidebar import render_sidebar
from ui.summary_and_export import render_summary_and_export
from ui.variable_definition import render_variable_definition

# --- Configuración general ---------------------------------------------------
st.set_page_config(
    page_title="StatLab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .sig-yes { color: #e53e3e; font-weight: bold; }
    .sig-no { color: #38a169; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- Estado de la sesión ------------------------------------------------------
if 'df' not in st.session_state:
    st.session_state.df = None
if 'var_types' not in st.session_state:
    st.session_state.var_types = {}
if 'results' not in st.session_state:
    st.session_state.results = []
if 'figures' not in st.session_state:
    st.session_state.figures = []

# === NAVEGACIÓN ===============================================================

MODE_ANALYSIS = "Análisis estadístico"
MODE_SAMPLE_SIZE = "Calculadora de tamaño muestral"

st.title("📊 StatLab")

mode = st.radio(
    "Módulo",
    [MODE_ANALYSIS, MODE_SAMPLE_SIZE],
    horizontal=True,
    label_visibility="collapsed",
)

# === ORQUESTACIÓN =============================================================

if mode == MODE_ANALYSIS:
    st.caption("Análisis estadístico y generación de figuras")
    fig_options = render_sidebar()

    render_data_upload()
    df = st.session_state.df

    if df is not None:
        render_data_preview(df)
        var_types = render_variable_definition(df)
        config = render_analysis_config(df, var_types)
        render_results(config, df, fig_options)
        render_figures_section(config, df, fig_options)
        render_summary_and_export()
    else:
        render_landing()

elif mode == MODE_SAMPLE_SIZE:
    st.caption("Calcula el n necesario antes de recolectar datos")
    render_sample_size()
