"""Carga de datos CSV/Excel, preview y resumen descriptivo."""

import os

import pandas as pd
import streamlit as st

MAX_FILE_SIZE_MB = 100

# Raíz del proyecto (carpeta padre de ui/). Anclar aquí las rutas de los datasets
# de ejemplo evita que dependan del directorio de ejecución (cwd), que cambia según
# cómo se lance la app (local, Dev Container, Streamlit Cloud).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def validate_file_size(uploaded_file):
    """Valida que el archivo no exceda el límite de tamaño.

    Returns:
        True si el archivo es válido, False si excede el límite.
    """
    uploaded_file.seek(0, 2)
    size_bytes = uploaded_file.tell()
    uploaded_file.seek(0)
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        st.error(f"El archivo pesa {size_mb:.1f} MB y excede el límite de {MAX_FILE_SIZE_MB} MB.")
        return False
    return True


def render_data_upload():
    """Renderiza la sección de carga de datos.

    Escribe el DataFrame en st.session_state.df.
    """
    st.header("1. Datos")

    upload_col, sample_col = st.columns([3, 1])
    with upload_col:
        uploaded_file = st.file_uploader("Sube tu archivo", type=['csv', 'xlsx', 'xls'],
                                         help="Formatos aceptados: CSV, XLSX, XLS")
    with sample_col:
        st.markdown("<br>", unsafe_allow_html=True)
        # Etiqueta descriptiva -> archivo. Dos datasets de demostración:
        #  - wide (una fila por paciente): cubre la mayoría de tests.
        #  - long (medidas repetidas): cubre pareados e ICC, que la app pide apilados.
        _sample_labels = {
            "Ensayo clínico (grupos, correlación, ROC, supervivencia…)": "datos_demo.csv",
            "Concordancia entre observadores (pareados, ICC)": "datos_demo_observadores.csv",
        }
        _sample_files = {label: os.path.join(_PROJECT_ROOT, fname)
                         for label, fname in _sample_labels.items()
                         if os.path.exists(os.path.join(_PROJECT_ROOT, fname))}
        if _sample_files:
            _chosen = st.selectbox("Datos de ejemplo", [""] + list(_sample_files.keys()),
                                   label_visibility="collapsed")
            if _chosen and st.button("Cargar ejemplo", use_container_width=True):
                _path = _sample_files[_chosen]
                st.session_state.df = pd.read_csv(_path)
                st.success(f"Cargado: {_path} — "
                           f"{st.session_state.df.shape[0]} filas × {st.session_state.df.shape[1]} columnas")

    if uploaded_file:
        if not validate_file_size(uploaded_file):
            return
        try:
            if uploaded_file.name.endswith('.csv'):
                sample = uploaded_file.read(4096).decode('utf-8', errors='ignore')
                uploaded_file.seek(0)
                sep = ',' if sample.count(',') > sample.count(';') else ';'
                st.session_state.df = pd.read_csv(uploaded_file, sep=sep)
            else:
                st.session_state.df = pd.read_excel(uploaded_file)
            st.success(f"Cargado: {uploaded_file.name} — "
                       f"{st.session_state.df.shape[0]} filas × {st.session_state.df.shape[1]} columnas")
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")


def render_data_preview(df):
    """Muestra preview y resumen descriptivo del DataFrame."""
    with st.expander("Vista previa de los datos", expanded=True):
        st.dataframe(df.head(20), use_container_width=True)

    with st.expander("Resumen descriptivo"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Filas", df.shape[0])
        col2.metric("Columnas", df.shape[1])
        col3.metric("Valores faltantes", int(df.isna().sum().sum()))
        st.dataframe(df.describe(include='all').round(3), use_container_width=True)
