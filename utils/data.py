"""Utilidades para carga, inferencia y validacion de datos."""

import pandas as pd


def infer_variable_type(series):
    """Infiere si una variable es categorica o continua."""
    if series.dtype == 'object' or series.dtype.name == 'category':
        return 'Categorica'
    if series.nunique() <= 10 and series.dtype in ['int64', 'int32']:
        return 'Categorica'
    return 'Continua'


def validate_continuous(df, col_name):
    """Verifica que una columna marcada como continua sea realmente numerica.
    Retorna (ok, mensaje_error)."""
    if col_name not in df.columns:
        return False, f"La columna '{col_name}' no existe en los datos."
    if not pd.api.types.is_numeric_dtype(df[col_name]):
        return False, (f"La columna '{col_name}' no es numerica. "
                       f"Tipo detectado: {df[col_name].dtype}. "
                       f"Cambiala a 'Categorica' o revisa los datos.")
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
