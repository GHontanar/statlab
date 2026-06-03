"""Tests para utils/data.py"""

import numpy as np
import pandas as pd

from utils.constants import TIPO_CATEGORICA, TIPO_NUMERICA
from utils.data import (
    infer_variable_type,
    is_ambiguous_numeric,
    validate_continuous,
    validate_group_sizes,
)


class TestInferVariableType:
    """Tests para infer_variable_type."""

    def test_string_column_is_categorical(self):
        s = pd.Series(['A', 'B', 'C', 'A', 'B'])
        assert infer_variable_type(s) == TIPO_CATEGORICA

    def test_category_dtype_is_categorical(self):
        s = pd.Series(['X', 'Y', 'Z']).astype('category')
        assert infer_variable_type(s) == TIPO_CATEGORICA

    def test_float_column_is_numeric(self):
        s = pd.Series([1.5, 2.3, 4.7, 8.1, 3.2])
        assert infer_variable_type(s) == TIPO_NUMERICA

    def test_int_few_unique_is_categorical(self):
        s = pd.Series([1, 2, 3, 1, 2, 3, 1, 2])
        assert infer_variable_type(s) == TIPO_CATEGORICA

    def test_int_many_unique_is_numeric(self):
        s = pd.Series(range(50))
        assert infer_variable_type(s) == TIPO_NUMERICA

    def test_int_boundary_10_unique_is_categorical(self):
        s = pd.Series(list(range(10)) * 3)
        assert infer_variable_type(s) == TIPO_CATEGORICA

    def test_int_11_unique_is_numeric(self):
        s = pd.Series(list(range(11)) * 2)
        assert infer_variable_type(s) == TIPO_NUMERICA

    def test_bool_column_treated_as_categorical(self):
        """B5: bool dtype se detecta como categorica."""
        s = pd.Series([True, False, True, False])
        assert infer_variable_type(s) == TIPO_CATEGORICA

    # --- Bug A: dtypes enteros no estandar ---

    def test_nullable_int_few_unique_is_categorical(self):
        """Bug A: Int64 nullable de baja cardinalidad -> categorica."""
        s = pd.Series([1, 2, 3, None, 1, 2], dtype='Int64')
        assert infer_variable_type(s) == TIPO_CATEGORICA

    def test_nullable_int_many_unique_is_numeric(self):
        s = pd.Series(range(50), dtype='Int64')
        assert infer_variable_type(s) == TIPO_NUMERICA

    def test_int8_few_unique_is_categorical(self):
        """Bug A: int8 de baja cardinalidad -> categorica."""
        s = pd.Series([1, 2, 3, 1, 2], dtype='int8')
        assert infer_variable_type(s) == TIPO_CATEGORICA

    def test_uint8_few_unique_is_categorical(self):
        s = pd.Series([0, 1, 2, 0, 1], dtype='uint8')
        assert infer_variable_type(s) == TIPO_CATEGORICA

    # --- Bug A + B: enteros con NaN leidos como float ---

    def test_int_with_nan_as_float_low_cardinality_is_categorical(self):
        """Bug A+B: enteros con NaN que pandas lee como float64."""
        s = pd.Series([1.0, 2.0, np.nan, 1.0, 2.0, 3.0])
        assert infer_variable_type(s) == TIPO_CATEGORICA

    # --- Bug B: floats categoricos ---

    def test_float_binary_codes_is_categorical(self):
        """Bug B: codigos categoricos en float (0.0/1.0) -> categorica."""
        s = pd.Series([0.0, 1.0, 0.0, 1.0, 1.0])
        assert infer_variable_type(s) == TIPO_CATEGORICA

    def test_float_high_cardinality_is_numeric(self):
        """Regresion: un float continuo de verdad sigue siendo numerico."""
        s = pd.Series([float(x) / 7 for x in range(50)])
        assert infer_variable_type(s) == TIPO_NUMERICA

    # --- datetime y edge cases ---

    def test_datetime_is_categorical(self):
        """Decision documentada: datetime -> categorica (no analizable como numerica)."""
        s = pd.Series(pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']))
        assert infer_variable_type(s) == TIPO_CATEGORICA

    def test_all_nan_numeric_is_categorical(self):
        """Serie numerica todo-NaN (nunique()==0) cae en <= umbral -> categorica."""
        s = pd.Series([np.nan, np.nan, np.nan])
        assert infer_variable_type(s) == TIPO_CATEGORICA


class TestIsAmbiguousNumeric:
    """Tests para is_ambiguous_numeric (zona de revision)."""

    def test_float_high_cardinality_not_ambiguous(self):
        s = pd.Series([float(x) / 7 for x in range(50)])
        assert is_ambiguous_numeric(s) is False

    def test_int_low_cardinality_ambiguous(self):
        s = pd.Series([1, 2, 3, 1, 2])
        assert is_ambiguous_numeric(s) is True

    def test_float_binary_codes_ambiguous(self):
        s = pd.Series([0.0, 1.0, 0.0, 1.0])
        assert is_ambiguous_numeric(s) is True

    def test_string_not_ambiguous(self):
        s = pd.Series(['a', 'b', 'c'])
        assert is_ambiguous_numeric(s) is False

    def test_category_not_ambiguous(self):
        s = pd.Series(['x', 'y']).astype('category')
        assert is_ambiguous_numeric(s) is False

    def test_bool_not_ambiguous(self):
        s = pd.Series([True, False, True])
        assert is_ambiguous_numeric(s) is False


class TestValidateContinuous:
    def test_numeric_column_ok(self):
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0]})
        ok, err = validate_continuous(df, 'x')
        assert ok
        assert err == ""

    def test_string_column_fails(self):
        df = pd.DataFrame({'x': ['a', 'b', 'c']})
        ok, err = validate_continuous(df, 'x')
        assert not ok
        assert 'no es numérica' in err

    def test_missing_column_fails(self):
        df = pd.DataFrame({'x': [1, 2, 3]})
        ok, err = validate_continuous(df, 'inexistente')
        assert not ok
        assert 'no existe' in err

    def test_all_nan_fails(self):
        df = pd.DataFrame({'x': [np.nan, np.nan, np.nan]})
        ok, err = validate_continuous(df, 'x')
        assert not ok
        assert 'NaN' in err

    def test_int_column_ok(self):
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        ok, err = validate_continuous(df, 'x')
        assert ok

    def test_partial_nan_ok(self):
        df = pd.DataFrame({'x': [1.0, np.nan, 3.0]})
        ok, err = validate_continuous(df, 'x')
        assert ok


class TestValidateGroupSizes:
    def test_sufficient_groups(self):
        df = pd.DataFrame({
            'valor': [1, 2, 3, 4, 5, 6],
            'grupo': ['A', 'A', 'A', 'B', 'B', 'B'],
        })
        ok, err, counts = validate_group_sizes(df, 'valor', 'grupo', ['A', 'B'])
        assert ok
        assert counts == {'A': 3, 'B': 3}

    def test_group_too_small(self):
        df = pd.DataFrame({
            'valor': [1, 2, 3, 4],
            'grupo': ['A', 'A', 'A', 'B'],
        })
        ok, err, counts = validate_group_sizes(df, 'valor', 'grupo', ['A', 'B'], min_n=2)
        assert not ok
        assert "n=1" in err

    def test_empty_group(self):
        df = pd.DataFrame({
            'valor': [1, 2, 3, np.nan],
            'grupo': ['A', 'A', 'A', 'B'],
        })
        ok, err, counts = validate_group_sizes(df, 'valor', 'grupo', ['A', 'B'])
        assert not ok
        assert counts['B'] == 0

    def test_custom_min_n(self):
        df = pd.DataFrame({
            'valor': [1, 2, 3, 4, 5, 6],
            'grupo': ['A', 'A', 'A', 'B', 'B', 'B'],
        })
        ok, err, counts = validate_group_sizes(df, 'valor', 'grupo', ['A', 'B'], min_n=5)
        assert not ok
        assert "minimo requerido: 5" in err


# --- B5: Bool inference ---

class TestBoolInference:
    def test_bool_series_is_categorical(self):
        s = pd.Series([True, False, True, False])
        assert infer_variable_type(s) == TIPO_CATEGORICA

    def test_bool_dtype_is_categorical(self):
        s = pd.Series([True, False, True], dtype='bool')
        assert infer_variable_type(s) == TIPO_CATEGORICA
