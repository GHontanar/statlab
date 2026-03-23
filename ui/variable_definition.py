"""Inferencia y corrección de tipos de variable."""

import streamlit as st

from utils.data import infer_variable_type


def render_variable_definition(df):
    """Renderiza la sección de definición de variables.

    Args:
        df: DataFrame con los datos cargados.

    Returns:
        dict: Mapeo {nombre_columna: tipo} donde tipo es 'Continua' o 'Categórica'.
    """
    st.header("2. Definición de variables")

    cols = df.columns.tolist()
    var_types = {}

    _clear_vars = []
    _ambiguous_vars = []
    for col_name in cols:
        dtype = df[col_name].dtype
        if dtype in ('object', 'category', 'bool'):
            _clear_vars.append((col_name, 'Categórica'))
        elif dtype in ('float64', 'float32'):
            _clear_vars.append((col_name, 'Continua'))
        else:
            _ambiguous_vars.append((col_name, infer_variable_type(df[col_name])))

    if _clear_vars:
        _cont = [c for c, t in _clear_vars if t == 'Continua']
        _cat = [c for c, t in _clear_vars if t == 'Categórica']
        _summary = []
        if _cont:
            _summary.append(f"**Continuas**: {', '.join(_cont)}")
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
                    ['Continua', 'Categórica'],
                    index=0 if inferred == 'Continua' else 1,
                    key=f"vtype_{col_name}"
                )

    with st.expander("Corregir tipos de variable"):
        corr_grid = st.columns(3)
        for i, col_name in enumerate(cols):
            current = var_types.get(col_name, 'Continua')
            with corr_grid[i % 3]:
                var_types[col_name] = st.selectbox(
                    f"`{col_name}`",
                    ['Continua', 'Categórica'],
                    index=0 if current == 'Continua' else 1,
                    key=f"vtype_corr_{col_name}"
                )

    st.session_state.var_types = var_types
    return var_types
