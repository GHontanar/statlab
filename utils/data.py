"""Utilidades para carga, inferencia y validacion de datos."""

import pandas as pd

from utils.constants import (
    TIPO_CATEGORICA,
    TIPO_NUMERICA,
    UMBRAL_CARDINALIDAD_CATEGORICA,
)


def _is_low_cardinality_integer(series):
    """True si la serie numerica toma pocos valores unicos y todos son enteros.

    El umbral de cardinalidad solo tiene sentido para valores entero-equivalentes
    (codigos 1/2/3, 0.0/1.0, enteros-con-NaN que pandas lee como float). Un float
    con parte decimal es una medicion continua y NO debe caer por cardinalidad,
    por pequena que sea la muestra. Asume que la serie ya es de dtype numerico.
    """
    clean = series.dropna()
    if len(clean) == 0:
        return True  # todo-NaN: sin senal, se trata como categorica
    if not (clean % 1 == 0).all():
        return False  # tiene decimales -> continua
    return series.nunique() <= UMBRAL_CARDINALIDAD_CATEGORICA


def infer_variable_type(series):
    """Infiere si una variable es numerica o categorica.

    Unica fuente de verdad de la inferencia de tipos. Reglas (en orden):
      1. object / category / bool -> categorica.
      2. datetime -> categorica (nunca debe ofrecerse como numerica para un
         t-test; validate_continuous la rechazaria de todos modos).
      3. numerica (cualquier dtype: int8/16/32/64, uint*, Int64 nullable,
         float32/64, y enteros con NaN que pandas lee como float): entero de
         baja cardinalidad -> categorica; si no -> numerica. Los floats con
         decimales son siempre numericos (ver _is_low_cardinality_integer).
      4. cualquier otro dtype -> categorica (no es analizable como numerica).
    """
    if series.dtype == 'object' or series.dtype.name == 'category':
        return TIPO_CATEGORICA
    if series.dtype == 'bool':
        return TIPO_CATEGORICA
    if pd.api.types.is_datetime64_any_dtype(series):
        return TIPO_CATEGORICA
    if pd.api.types.is_numeric_dtype(series):
        if _is_low_cardinality_integer(series):
            return TIPO_CATEGORICA
        return TIPO_NUMERICA
    return TIPO_CATEGORICA


def is_ambiguous_numeric(series):
    """True si la clasificacion numerica/categorica de la serie es dudosa.

    Es la zona donde infer_variable_type "adivina" categorica por cardinalidad:
    una serie numerica entera con pocos valores unicos (codigos 1/2/3, o un
    float 0.0/1.0). Conviene que el usuario la revise. Reutiliza el mismo umbral
    que infer_variable_type para no duplicar la heuristica. Bool se excluye
    (pandas lo considera numerico, pero es claramente categorico).
    """
    if series.dtype == 'bool':
        return False
    if not pd.api.types.is_numeric_dtype(series):
        return False
    return _is_low_cardinality_integer(series)


def validate_continuous(df, col_name):
    """Verifica que una columna marcada como continua sea realmente numerica.
    Retorna (ok, mensaje_error)."""
    if col_name not in df.columns:
        return False, f"La columna '{col_name}' no existe en los datos."
    if not pd.api.types.is_numeric_dtype(df[col_name]):
        return False, (f"La columna '{col_name}' no es numérica. "
                       f"Tipo detectado: {df[col_name].dtype}. "
                       f"Cámbiala a '{TIPO_CATEGORICA}' o revisa los datos.")
    n_valid = df[col_name].notna().sum()
    if n_valid == 0:
        return False, f"La columna '{col_name}' no tiene valores validos (todos NaN)."
    return True, ""


def validate_group_sizes(df, var_dep, var_group, groups, min_n=2):
    """Valida que cada grupo tenga suficientes observaciones.
    Retorna (ok, mensaje_error, detalle_por_grupo)."""
    issues = []
    group_counts = {}
    for g in groups:
        n = df[df[var_group] == g][var_dep].dropna().shape[0]
        group_counts[g] = n
        if n < min_n:
            issues.append(f"Grupo '{g}': n={n} (minimo requerido: {min_n})")

    if issues:
        msg = "Grupos con datos insuficientes:\n" + "\n".join(issues)
        return False, msg, group_counts
    return True, "", group_counts
