"""Tests para utils/data.py"""

import pandas as pd
import numpy as np
import pytest
from utils.data import infer_variable_type


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

    def test_bool_column_treated_as_continuous(self):
        """bool dtype no es object ni int, por lo que se infiere como Continua.
        Esto es un caso edge conocido — el usuario puede corregirlo en la UI."""
        s = pd.Series([True, False, True, False])
        assert infer_variable_type(s) == 'Continua'
