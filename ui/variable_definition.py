"""Inferencia y corrección de tipos de variable."""

import streamlit as st

from utils.constants import TIPO_CATEGORICA, TIPO_NUMERICA
from utils.data import infer_variable_type, is_ambiguous_numeric


def render_variable_definition(df):
    """Renderiza la sección de definición de variables.

    Args:
        df: DataFrame con los datos cargados.

    Returns:
        dict: Mapeo {nombre_columna: tipo} donde tipo es 'Numérica' o 'Categórica'.
    """
    st.header("2. Definición de variables")

    cols = df.columns.tolist()
    var_types = {}

    # infer_variable_type es la unica fuente de verdad; is_ambiguous_numeric
    # decide cuales conviene que el usuario revise (numericas de baja cardinalidad).
    _clear_vars = []
    _ambiguous_vars = []
    for col_name in cols:
        series = df[col_name]
        inferred = infer_variable_type(series)
        if is_ambiguous_numeric(series):
            _ambiguous_vars.append((col_name, inferred))
        else:
            _clear_vars.append((col_name, inferred))

    if _clear_vars:
        _num = [c for c, t in _clear_vars if t == TIPO_NUMERICA]
        _cat = [c for c, t in _clear_vars if t == TIPO_CATEGORICA]
        _summary = []
        if _num:
            _summary.append(f"**Numéricas**: {', '.join(_num)}")
        if _cat:
            _summary.append(f"**Categóricas**: {', '.join(_cat)}")
        st.success("Detectadas automáticamente: " + " | ".join(_summary))

    for col_name, vtype in _clear_vars:
        var_types[col_name] = vtype

    if _ambiguous_vars:
        st.info("Revisa estas variables (la inferencia puede no ser correcta):")
        amb_grid = st.columns(min(3, len(_ambiguous_vars)))
        for i, (col_name, inferred) in enumerate(_ambiguous_vars):
            with amb_grid[i % min(3, len(_ambiguous_vars))]:
                var_types[col_name] = st.selectbox(
                    f"`{col_name}`",
                    [TIPO_NUMERICA, TIPO_CATEGORICA],
                    index=0 if inferred == TIPO_NUMERICA else 1,
                    key=f"vtype_{col_name}"
                )

    with st.expander("Corregir tipos de variable"):
        corr_grid = st.columns(3)
        for i, col_name in enumerate(cols):
            current = var_types.get(col_name, TIPO_NUMERICA)
            with corr_grid[i % 3]:
                var_types[col_name] = st.selectbox(
                    f"`{col_name}`",
                    [TIPO_NUMERICA, TIPO_CATEGORICA],
                    index=0 if current == TIPO_NUMERICA else 1,
                    key=f"vtype_corr_{col_name}"
                )

    st.session_state.var_types = var_types
    return var_types
