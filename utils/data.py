"""Utilidades para carga, inferencia y validacion de datos."""


def infer_variable_type(series):
    """Infiere si una variable es categorica o continua."""
    if series.dtype == 'object' or series.dtype.name == 'category':
        return 'Categorica'
    if series.nunique() <= 10 and series.dtype in ['int64', 'int32']:
        return 'Categorica'
    return 'Continua'
