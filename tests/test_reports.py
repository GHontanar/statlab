"""Tests para reports/text.py"""

import pytest
from reports.text import format_result_text


@pytest.fixture
def ttest_result():
    return {
        'test_name': 'T-test de Welch',
        'var_dep': 'Hemoglobina',
        'var_group': 'Grupo',
        'alpha': 0.05,
        'groups': ['Treatment', 'Control'],
        'n': [15, 15],
        'mean': [10.74, 8.89],
        'std': [0.98, 0.53],
        'median': [10.6, 8.8],
        'statistic': 6.12,
        'p_value': 0.00001,
        'significant': True,
        'cohens_d': 2.35,
        'success': True,
    }


@pytest.fixture
def anova_result():
    return {
        'test_name': 'ANOVA one-way',
        'var_dep': 'Score',
        'var_group': 'Grupo',
        'alpha': 0.05,
        'groups': ['A', 'B', 'C'],
        'n': [20, 20, 20],
        'mean': [10.0, 12.0, 15.0],
        'std': [1.0, 1.5, 1.2],
        'statistic': 45.3,
        'p_value': 0.0001,
        'significant': True,
        'eta_squared': 0.62,
        'success': True,
    }


@pytest.fixture
def regression_result():
    return {
        'test_name': 'Regresion lineal',
        'var_dep': 'y',
        'var_group': 'x',
        'alpha': 0.05,
        'statistic': 0.95,
        'r_squared': 0.9025,
        'p_value': 0.00001,
        'significant': True,
        'slope': 2.1,
        'intercept': 3.5,
        'std_error': 0.12,
        'n': 30,
        'success': True,
    }


@pytest.fixture
def nonsig_result():
    return {
        'test_name': 'Mann-Whitney U',
        'var_dep': 'Valor',
        'var_group': 'Grupo',
        'alpha': 0.05,
        'groups': ['A', 'B'],
        'n': [10, 10],
        'mean': [5.0, 5.1],
        'std': [1.0, 1.1],
        'median': [5.0, 5.0],
        'statistic': 48.0,
        'p_value': 0.85,
        'significant': False,
        'success': True,
    }


class TestFormatResultText:
    def test_contains_test_name(self, ttest_result):
        text = format_result_text(ttest_result)
        assert 'T-test de Welch' in text

    def test_contains_variables(self, ttest_result):
        text = format_result_text(ttest_result)
        assert 'Hemoglobina' in text
        assert 'Grupo' in text

    def test_contains_p_value(self, ttest_result):
        text = format_result_text(ttest_result)
        assert 'p-valor' in text
        assert '***' in text

    def test_contains_groups(self, ttest_result):
        text = format_result_text(ttest_result)
        assert 'Treatment' in text
        assert 'Control' in text

    def test_cohens_d_shown(self, ttest_result):
        text = format_result_text(ttest_result)
        assert 'd de Cohen' in text
        assert 'grande' in text

    def test_eta_squared_shown(self, anova_result):
        text = format_result_text(anova_result)
        assert 'eta2' in text
        assert 'grande' in text

    def test_r_squared_shown(self, regression_result):
        text = format_result_text(regression_result)
        assert 'R2' in text

    def test_slope_shown(self, regression_result):
        text = format_result_text(regression_result)
        assert 'Pendiente' in text
        assert 'Intercepto' in text

    def test_conclusion_significant(self, ttest_result):
        text = format_result_text(ttest_result)
        assert 'Se rechaza H0' in text

    def test_conclusion_not_significant(self, nonsig_result):
        text = format_result_text(nonsig_result)
        assert 'No se rechaza H0' in text

    def test_ns_for_high_p(self, nonsig_result):
        text = format_result_text(nonsig_result)
        assert 'ns' in text

    def test_n_scalar_shown(self, regression_result):
        text = format_result_text(regression_result)
        assert 'N: 30' in text

    def test_effect_size_small(self):
        r = {'statistic': 1.0, 'p_value': 0.3, 'significant': False,
             'cohens_d': 0.2, 'test_name': 'test', 'alpha': 0.05}
        text = format_result_text(r)
        assert 'pequeno' in text

    def test_effect_size_medium(self):
        r = {'statistic': 2.0, 'p_value': 0.04, 'significant': True,
             'cohens_d': 0.6, 'test_name': 'test', 'alpha': 0.05}
        text = format_result_text(r)
        assert 'mediano' in text

    def test_significance_double_star(self):
        r = {'statistic': 3.0, 'p_value': 0.005, 'significant': True,
             'test_name': 'test', 'alpha': 0.05}
        text = format_result_text(r)
        assert '** (p < 0.01)' in text

    def test_significance_single_star(self):
        r = {'statistic': 2.0, 'p_value': 0.03, 'significant': True,
             'test_name': 'test', 'alpha': 0.05}
        text = format_result_text(r)
        assert '* (p < 0.05)' in text

    def test_minimal_result_no_crash(self):
        """Resultado minimo no debe crashear."""
        r = {'test': 'unknown'}
        text = format_result_text(r)
        assert 'N/A' in text
