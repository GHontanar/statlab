"""Tests para utils/data.py"""

import pandas as pd
import numpy as np
import pytest
from utils.data import infer_variable_type, validate_continuous, validate_group_sizes


class TestInferVariableType:
    """Tests para infer_variable_type."""

    def test_string_column_is_categorical(self):
        s = pd.Series(['A', 'B', 'C', 'A', 'B'])
        assert infer_variable_type(s) == 'Categorica'

    def test_category_dtype_is_categorical(self):
        s = pd.Series(['X', 'Y', 'Z']).astype('category')
        assert infer_variable_type(s) == 'Categorica'

    def test_float_column_is_continuous(self):
        s = pd.Series([1.5, 2.3, 4.7, 8.1, 3.2])
        assert infer_variable_type(s) == 'Continua'

    def test_int_few_unique_is_categorical(self):
        s = pd.Series([1, 2, 3, 1, 2, 3, 1, 2])
        assert infer_variable_type(s) == 'Categorica'

    def test_int_many_unique_is_continuous(self):
        s = pd.Series(range(50))
        assert infer_variable_type(s) == 'Continua'

    def test_int_boundary_10_unique_is_categorical(self):
        s = pd.Series(list(range(10)) * 3)
        assert infer_variable_type(s) == 'Categorica'

    def test_int_11_unique_is_continuous(self):
        s = pd.Series(list(range(11)) * 2)
        assert infer_variable_type(s) == 'Continua'

    def test_bool_column_treated_as_categorical(self):
        """B5: bool dtype se detecta como Categorica."""
        s = pd.Series([True, False, True, False])
        assert infer_variable_type(s) == 'Categorica'


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
        assert 'no es numerica' in err

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
        assert infer_variable_type(s) == 'Categorica'

    def test_bool_dtype_is_categorical(self):
        s = pd.Series([True, False, True], dtype='bool')
        assert infer_variable_type(s) == 'Categorica'
