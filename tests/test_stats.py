"""Tests para stats/tests.py"""

import pandas as pd
import numpy as np
import pytest
from stats.tests import check_normality, suggest_test, run_test


# --- Fixtures ----------------------------------------------------------------

@pytest.fixture
def sample_df():
    """DataFrame con 2 grupos y diferencia clara."""
    np.random.seed(42)
    n = 20
    return pd.DataFrame({
        'valor': np.concatenate([
            np.random.normal(10, 1, n),
            np.random.normal(15, 1, n),
        ]),
        'grupo': ['A'] * n + ['B'] * n,
    })


@pytest.fixture
def three_group_df():
    """DataFrame con 3 grupos."""
    np.random.seed(42)
    n = 20
    return pd.DataFrame({
        'valor': np.concatenate([
            np.random.normal(10, 1, n),
            np.random.normal(12, 1, n),
            np.random.normal(15, 1, n),
        ]),
        'grupo': ['A'] * n + ['B'] * n + ['C'] * n,
    })


@pytest.fixture
def correlation_df():
    """DataFrame con correlacion lineal fuerte."""
    np.random.seed(42)
    x = np.linspace(0, 10, 30)
    y = 2 * x + 3 + np.random.normal(0, 1, 30)
    return pd.DataFrame({'x': x, 'y': y})


@pytest.fixture
def categorical_df():
    """DataFrame para chi-cuadrado / Fisher."""
    return pd.DataFrame({
        'tratamiento': ['A'] * 30 + ['B'] * 30,
        'resultado': (['Mejora'] * 20 + ['No'] * 10 +
                       ['Mejora'] * 8 + ['No'] * 22),
    })


# --- check_normality --------------------------------------------------------

class TestCheckNormality:
    def test_normal_data_high_p(self):
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 50))
        stat, p = check_normality(data)
        assert p > 0.05

    def test_skewed_data_low_p(self):
        np.random.seed(42)
        data = pd.Series(np.random.exponential(1, 100))
        stat, p = check_normality(data)
        assert p < 0.05

    def test_too_few_samples_returns_none(self):
        data = pd.Series([1.0, 2.0])
        stat, p = check_normality(data)
        assert stat is None and p is None

    def test_too_many_samples_returns_none(self):
        data = pd.Series(np.random.normal(0, 1, 5001))
        stat, p = check_normality(data)
        assert stat is None and p is None

    def test_with_nans_cleaned(self):
        data = pd.Series([1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10])
        stat, p = check_normality(data)
        assert stat is not None

    def test_all_nan_returns_none(self):
        data = pd.Series([np.nan, np.nan, np.nan])
        stat, p = check_normality(data)
        assert stat is None


# --- suggest_test ------------------------------------------------------------

class TestSuggestTest:
    def test_2_groups_normal_independent(self):
        s = suggest_test('Continua', 'Categorica', 2, paired=False, normal=True)
        ids = [tid for _, tid in s]
        assert 't_independent' in ids
        assert 't_welch' in ids

    def test_2_groups_normal_paired(self):
        s = suggest_test('Continua', 'Categorica', 2, paired=True, normal=True)
        assert s[0][1] == 't_paired'

    def test_2_groups_nonnormal_independent(self):
        s = suggest_test('Continua', 'Categorica', 2, paired=False, normal=False)
        assert s[0][1] == 'mann_whitney'

    def test_2_groups_nonnormal_paired(self):
        s = suggest_test('Continua', 'Categorica', 2, paired=True, normal=False)
        assert s[0][1] == 'wilcoxon'

    def test_3_groups_normal(self):
        s = suggest_test('Continua', 'Categorica', 3, paired=False, normal=True)
        assert s[0][1] == 'anova'

    def test_3_groups_nonnormal(self):
        s = suggest_test('Continua', 'Categorica', 3, paired=False, normal=False)
        assert s[0][1] == 'kruskal'

    def test_3_groups_paired_normal(self):
        s = suggest_test('Continua', 'Categorica', 3, paired=True, normal=True)
        assert s[0][1] == 'rm_anova'

    def test_3_groups_paired_nonnormal(self):
        s = suggest_test('Continua', 'Categorica', 3, paired=True, normal=False)
        assert s[0][1] == 'friedman'

    def test_two_categorical(self):
        s = suggest_test('Categorica', 'Categorica', 2)
        ids = [tid for _, tid in s]
        assert 'chi2' in ids
        assert 'fisher' in ids

    def test_two_continuous_normal(self):
        s = suggest_test('Continua', 'Continua', 1, normal=True)
        ids = [tid for _, tid in s]
        assert 'pearson' in ids
        assert 'linear_reg' in ids

    def test_two_continuous_nonnormal(self):
        s = suggest_test('Continua', 'Continua', 1, normal=False)
        assert s[0][1] == 'spearman'

    def test_no_match_returns_empty(self):
        s = suggest_test('Categorica', 'Continua', 2)
        assert s == []


# --- run_test: camino feliz --------------------------------------------------

class TestRunTestHappyPath:
    def test_t_independent(self, sample_df):
        r = run_test('t_independent', sample_df, 'valor', 'grupo', ['A', 'B'])
        assert r['success']
        assert r['significant']
        assert r['p_value'] < 0.05
        assert 'cohens_d' in r
        assert len(r['groups']) == 2
        assert r['n'] == [20, 20]

    def test_t_welch(self, sample_df):
        r = run_test('t_welch', sample_df, 'valor', 'grupo', ['A', 'B'])
        assert r['success']
        assert r['test_name'] == "T-test de Welch"

    def test_mann_whitney(self, sample_df):
        r = run_test('mann_whitney', sample_df, 'valor', 'grupo', ['A', 'B'])
        assert r['success']
        assert r['test_name'] == "Mann-Whitney U"

    def test_t_paired(self, sample_df):
        r = run_test('t_paired', sample_df, 'valor', 'grupo', ['A', 'B'])
        assert r['success']
        assert r['test_name'] == "T-test pareado"

    def test_wilcoxon(self, sample_df):
        r = run_test('wilcoxon', sample_df, 'valor', 'grupo', ['A', 'B'])
        assert r['success']

    def test_anova(self, three_group_df):
        r = run_test('anova', three_group_df, 'valor', 'grupo',
                     ['A', 'B', 'C'])
        assert r['success']
        assert r['significant']
        assert 'eta_squared' in r
        assert len(r['groups']) == 3

    def test_anova_posthoc(self, three_group_df):
        r = run_test('anova', three_group_df, 'valor', 'grupo',
                     ['A', 'B', 'C'])
        assert 'posthoc' in r or 'posthoc_error' in r

    def test_kruskal(self, three_group_df):
        r = run_test('kruskal', three_group_df, 'valor', 'grupo',
                     ['A', 'B', 'C'])
        assert r['success']
        assert r['test_name'] == "Kruskal-Wallis"

    def test_pearson(self, correlation_df):
        r = run_test('pearson', correlation_df, 'y', 'x')
        assert r['success']
        assert r['statistic'] > 0.9
        assert 'r_squared' in r

    def test_spearman(self, correlation_df):
        r = run_test('spearman', correlation_df, 'y', 'x')
        assert r['success']

    def test_linear_reg(self, correlation_df):
        r = run_test('linear_reg', correlation_df, 'y', 'x')
        assert r['success']
        assert 'slope' in r
        assert 'intercept' in r
        assert abs(r['slope'] - 2.0) < 0.5  # slope ~2

    def test_chi2(self, categorical_df):
        r = run_test('chi2', categorical_df, 'resultado', 'tratamiento')
        assert r['success']
        assert r['significant']
        assert 'dof' in r

    def test_fisher(self, categorical_df):
        r = run_test('fisher', categorical_df, 'resultado', 'tratamiento')
        assert r['success']
        assert r['test_name'] == "Test exacto de Fisher"


# --- run_test: edge cases y errores -----------------------------------------

class TestRunTestEdgeCases:
    def test_auto_group_detection(self, sample_df):
        """Sin pasar groups, debe detectar automaticamente."""
        r = run_test('t_welch', sample_df, 'valor', 'grupo')
        assert r['success']
        assert len(r['groups']) == 2

    def test_paired_unequal_n_warns(self):
        """Grupos con distinto n deben generar warning."""
        df = pd.DataFrame({
            'valor': [1, 2, 3, 4, 5, 10, 20, 30],
            'grupo': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B'],
        })
        r = run_test('t_paired', df, 'valor', 'grupo', ['A', 'B'])
        assert r['success']
        assert 'warning' in r

    def test_invalid_column_fails_gracefully(self, sample_df):
        r = run_test('t_welch', sample_df, 'columna_inexistente', 'grupo',
                     ['A', 'B'])
        assert not r['success']
        assert 'error' in r

    def test_cohens_d_is_positive(self, sample_df):
        r = run_test('t_independent', sample_df, 'valor', 'grupo', ['A', 'B'])
        assert r['cohens_d'] > 0

    def test_cohens_d_large_effect(self, sample_df):
        """Diferencia de 5 con std ~1 => d grande."""
        r = run_test('t_independent', sample_df, 'valor', 'grupo', ['A', 'B'])
        assert r['cohens_d'] > 0.8

    def test_eta_squared_range(self, three_group_df):
        r = run_test('anova', three_group_df, 'valor', 'grupo',
                     ['A', 'B', 'C'])
        assert 0 < r['eta_squared'] < 1

    def test_no_significant_result(self):
        """Mismos datos => no significativo."""
        np.random.seed(42)
        vals = np.random.normal(10, 1, 40)
        df = pd.DataFrame({
            'valor': vals,
            'grupo': ['A'] * 20 + ['B'] * 20,
        })
        r = run_test('t_welch', df, 'valor', 'grupo', ['A', 'B'])
        assert r['success']
        assert not r['significant']

    def test_custom_alpha(self, sample_df):
        r = run_test('t_welch', sample_df, 'valor', 'grupo', ['A', 'B'],
                     alpha=0.001)
        assert r['alpha'] == 0.001

    def test_result_has_metadata(self, sample_df):
        r = run_test('t_welch', sample_df, 'valor', 'grupo', ['A', 'B'])
        assert r['test'] == 't_welch'
        assert r['var_dep'] == 'valor'
        assert r['var_group'] == 'grupo'

    def test_paired_with_id_column(self):
        """Emparejar por columna ID en vez de por posicion."""
        df = pd.DataFrame({
            'id': ['s1', 's2', 's3', 's1', 's2', 's3'],
            'valor': [10, 20, 30, 15, 25, 35],
            'grupo': ['Pre', 'Pre', 'Pre', 'Post', 'Post', 'Post'],
        })
        r = run_test('t_paired', df, 'valor', 'grupo', ['Pre', 'Post'],
                     paired_id_col='id')
        assert r['success']
        assert 'warning' not in r  # 3 vs 3, all matched

    def test_paired_with_id_partial_match(self):
        """ID parcial genera warning con conteo correcto."""
        df = pd.DataFrame({
            'id': ['s1', 's2', 's3', 's1', 's2'],
            'valor': [10, 20, 30, 15, 25],
            'grupo': ['Pre', 'Pre', 'Pre', 'Post', 'Post'],
        })
        r = run_test('t_paired', df, 'valor', 'grupo', ['Pre', 'Post'],
                     paired_id_col='id')
        assert r['success']
        assert 'warning' in r
        assert '2 pares' in r['warning']
