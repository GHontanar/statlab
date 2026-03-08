"""Tests para stats/tests.py"""

import pandas as pd
import numpy as np
import pytest
from stats.tests import check_normality, check_homogeneity, suggest_test, run_test


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

    def test_anova_posthoc_default(self, three_group_df):
        r = run_test('anova', three_group_df, 'valor', 'grupo',
                     ['A', 'B', 'C'])
        assert 'posthoc' in r or 'posthoc_error' in r
        if 'posthoc' in r:
            assert r['posthoc_name'] == 'Tukey HSD'

    def test_anova_posthoc_scheffe(self, three_group_df):
        r = run_test('anova', three_group_df, 'valor', 'grupo',
                     ['A', 'B', 'C'], extra={'posthoc_method': 'scheffe'})
        assert r['success']
        if 'posthoc' in r:
            assert r['posthoc_name'] == 'Scheffe'

    def test_anova_posthoc_bonferroni(self, three_group_df):
        r = run_test('anova', three_group_df, 'valor', 'grupo',
                     ['A', 'B', 'C'], extra={'posthoc_method': 'bonferroni_t'})
        assert r['success']
        if 'posthoc' in r:
            assert r['posthoc_name'] == 'Bonferroni (t-test)'

    def test_anova_posthoc_holm(self, three_group_df):
        r = run_test('anova', three_group_df, 'valor', 'grupo',
                     ['A', 'B', 'C'], extra={'posthoc_method': 'holm_t'})
        assert r['success']
        if 'posthoc' in r:
            assert r['posthoc_name'] == 'Holm (t-test)'

    def test_kruskal(self, three_group_df):
        r = run_test('kruskal', three_group_df, 'valor', 'grupo',
                     ['A', 'B', 'C'])
        assert r['success']
        assert r['test_name'] == "Kruskal-Wallis"

    def test_kruskal_posthoc_dunn_holm(self, three_group_df):
        r = run_test('kruskal', three_group_df, 'valor', 'grupo',
                     ['A', 'B', 'C'], extra={'posthoc_method': 'dunn_holm'})
        assert r['success']
        if 'posthoc' in r:
            assert r['posthoc_name'] == 'Dunn (Holm)'

    def test_kruskal_posthoc_dunn_bh(self, three_group_df):
        r = run_test('kruskal', three_group_df, 'valor', 'grupo',
                     ['A', 'B', 'C'], extra={'posthoc_method': 'dunn_bh'})
        assert r['success']
        if 'posthoc' in r:
            assert r['posthoc_name'] == 'Dunn (Benjamini-Hochberg)'

    def test_kruskal_posthoc_conover(self, three_group_df):
        r = run_test('kruskal', three_group_df, 'valor', 'grupo',
                     ['A', 'B', 'C'], extra={'posthoc_method': 'conover_bonferroni'})
        assert r['success']
        if 'posthoc' in r:
            assert r['posthoc_name'] == 'Conover (Bonferroni)'

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


# --- Bland-Altman -----------------------------------------------------------

@pytest.fixture
def bland_altman_df():
    """DataFrame con dos metodos de medicion correlacionados."""
    np.random.seed(42)
    n = 30
    method1 = np.random.normal(100, 15, n)
    method2 = method1 + np.random.normal(2, 5, n)  # sesgo de ~2
    return pd.DataFrame({'metodo1': method1, 'metodo2': method2})


class TestBlandAltman:
    def test_basic_result(self, bland_altman_df):
        r = run_test('bland_altman', bland_altman_df, 'metodo1', 'metodo2')
        assert r['success']
        assert r['test_name'] == 'Bland-Altman'
        assert 'bias' in r
        assert 'sd_diff' in r
        assert 'loa_upper' in r
        assert 'loa_lower' in r

    def test_bias_approximately_correct(self, bland_altman_df):
        r = run_test('bland_altman', bland_altman_df, 'metodo1', 'metodo2')
        # Sesgo esperado ~-2 (method1 - method2, method2 = method1 + ~2)
        assert abs(r['bias'] - (-2)) < 3

    def test_limits_of_agreement_symmetric(self, bland_altman_df):
        r = run_test('bland_altman', bland_altman_df, 'metodo1', 'metodo2')
        # LOA deben ser simetricas respecto al sesgo
        upper_dist = r['loa_upper'] - r['bias']
        lower_dist = r['bias'] - r['loa_lower']
        assert abs(upper_dist - lower_dist) < 0.001

    def test_has_p_value(self, bland_altman_df):
        r = run_test('bland_altman', bland_altman_df, 'metodo1', 'metodo2')
        assert 'p_value' in r
        assert isinstance(r['p_value'], float)

    def test_n_correct(self, bland_altman_df):
        r = run_test('bland_altman', bland_altman_df, 'metodo1', 'metodo2')
        assert r['n'] == 30

    def test_means_and_diffs_lists(self, bland_altman_df):
        r = run_test('bland_altman', bland_altman_df, 'metodo1', 'metodo2')
        assert len(r['means']) == 30
        assert len(r['diffs']) == 30


# --- ROC -------------------------------------------------------------------

@pytest.fixture
def roc_df():
    """DataFrame con predictor continuo y desenlace binario."""
    np.random.seed(42)
    n = 60
    # Buen predictor: enfermos tienen valores mas altos
    score = np.concatenate([
        np.random.normal(5, 1, n // 2),   # sanos
        np.random.normal(8, 1, n // 2),    # enfermos
    ])
    outcome = ['Sano'] * (n // 2) + ['Enfermo'] * (n // 2)
    return pd.DataFrame({'score': score, 'desenlace': outcome})


class TestROC:
    def test_basic_result(self, roc_df):
        r = run_test('roc', roc_df, 'desenlace', 'score',
                     extra={'positive_label': 'Enfermo'})
        assert r['success']
        assert r['test_name'] == 'Curva ROC'
        assert 'auc' in r
        assert 'fpr' in r
        assert 'tpr' in r

    def test_auc_high_for_good_predictor(self, roc_df):
        r = run_test('roc', roc_df, 'desenlace', 'score',
                     extra={'positive_label': 'Enfermo'})
        assert r['auc'] > 0.85

    def test_auc_range(self, roc_df):
        r = run_test('roc', roc_df, 'desenlace', 'score',
                     extra={'positive_label': 'Enfermo'})
        assert 0 <= r['auc'] <= 1

    def test_optimal_cutoff_exists(self, roc_df):
        r = run_test('roc', roc_df, 'desenlace', 'score',
                     extra={'positive_label': 'Enfermo'})
        assert 'best_threshold' in r
        assert 'sensitivity' in r
        assert 'specificity' in r

    def test_sensitivity_specificity_range(self, roc_df):
        r = run_test('roc', roc_df, 'desenlace', 'score',
                     extra={'positive_label': 'Enfermo'})
        assert 0 <= r['sensitivity'] <= 1
        assert 0 <= r['specificity'] <= 1

    def test_positive_label_custom(self, roc_df):
        r = run_test('roc', roc_df, 'desenlace', 'score',
                     extra={'positive_label': 'Enfermo'})
        assert r['success']
        assert r['positive_label'] == 'Enfermo'

    def test_default_positive_label(self, roc_df):
        """Sin positive_label usa el segundo valor ordenado."""
        r = run_test('roc', roc_df, 'desenlace', 'score')
        assert r['success']
        assert r['positive_label'] == 'Sano'  # sorted: ['Enfermo', 'Sano']

    def test_non_binary_outcome_fails(self):
        df = pd.DataFrame({
            'score': [1, 2, 3, 4, 5, 6],
            'desenlace': ['A', 'B', 'C', 'A', 'B', 'C'],
        })
        r = run_test('roc', df, 'desenlace', 'score')
        assert not r['success']
        assert '2 categorias' in r['error']

    def test_p_value_is_none(self, roc_df):
        r = run_test('roc', roc_df, 'desenlace', 'score',
                     extra={'positive_label': 'Enfermo'})
        assert r['p_value'] is None
        assert r['significant'] is None


# --- Kaplan-Meier -----------------------------------------------------------

@pytest.fixture
def survival_df():
    """DataFrame con datos de supervivencia."""
    np.random.seed(42)
    n = 40
    return pd.DataFrame({
        'tiempo': np.concatenate([
            np.random.exponential(20, n // 2),
            np.random.exponential(10, n // 2),
        ]),
        'evento': np.random.binomial(1, 0.7, n),
        'grupo': ['A'] * (n // 2) + ['B'] * (n // 2),
    })


class TestKaplanMeier:
    def test_basic_no_groups(self, survival_df):
        r = run_test('kaplan_meier', survival_df, 'tiempo', 'evento')
        assert r['success']
        assert r['test_name'] == 'Kaplan-Meier'
        assert 'Global' in r['curves']
        assert r['p_value'] is None

    def test_curve_has_timeline_and_survival(self, survival_df):
        r = run_test('kaplan_meier', survival_df, 'tiempo', 'evento')
        curve = r['curves']['Global']
        assert len(curve['timeline']) > 0
        assert len(curve['survival']) > 0
        assert curve['n'] == 40

    def test_survival_starts_at_one(self, survival_df):
        r = run_test('kaplan_meier', survival_df, 'tiempo', 'evento')
        curve = r['curves']['Global']
        assert curve['survival'][0] == 1.0 or curve['survival'][0] > 0.9

    def test_with_groups_logrank(self, survival_df):
        r = run_test('kaplan_meier', survival_df, 'tiempo', 'evento',
                     extra={'group_col': 'grupo', 'groups': ['A', 'B']})
        assert r['success']
        assert 'A' in r['curves']
        assert 'B' in r['curves']
        assert isinstance(r['p_value'], float)
        assert isinstance(r['statistic'], float)

    def test_n_per_group(self, survival_df):
        r = run_test('kaplan_meier', survival_df, 'tiempo', 'evento',
                     extra={'group_col': 'grupo', 'groups': ['A', 'B']})
        assert r['curves']['A']['n'] == 20
        assert r['curves']['B']['n'] == 20

    def test_invalid_event_variable(self):
        df = pd.DataFrame({
            'tiempo': [1, 2, 3, 4, 5],
            'evento': [0, 1, 2, 3, 4],
        })
        r = run_test('kaplan_meier', df, 'tiempo', 'evento')
        assert not r['success']
        assert 'binaria' in r['error']


# --- S1: Levene (homogeneidad de varianzas) ---------------------------------

class TestCheckHomogeneity:
    def test_equal_variances(self):
        np.random.seed(42)
        g1 = pd.Series(np.random.normal(10, 1, 30))
        g2 = pd.Series(np.random.normal(12, 1, 30))
        stat, p = check_homogeneity([g1, g2])
        assert stat is not None
        assert p > 0.05  # varianzas iguales

    def test_unequal_variances(self):
        np.random.seed(42)
        g1 = pd.Series(np.random.normal(10, 1, 30))
        g2 = pd.Series(np.random.normal(12, 5, 30))
        stat, p = check_homogeneity([g1, g2])
        assert stat is not None
        assert p < 0.05  # varianzas distintas

    def test_too_few_groups(self):
        g1 = pd.Series([1, 2, 3])
        stat, p = check_homogeneity([g1])
        assert stat is None
        assert p is None

    def test_too_few_samples(self):
        g1 = pd.Series([1.0])
        g2 = pd.Series([2.0, 3.0])
        stat, p = check_homogeneity([g1, g2])
        assert stat is None
        assert p is None

    def test_three_groups(self):
        np.random.seed(42)
        g1 = pd.Series(np.random.normal(10, 1, 20))
        g2 = pd.Series(np.random.normal(12, 1, 20))
        g3 = pd.Series(np.random.normal(14, 1, 20))
        stat, p = check_homogeneity([g1, g2, g3])
        assert stat is not None
        assert p is not None


class TestSuggestTestEqualVar:
    def test_normal_equal_var_recommends_ttest(self):
        suggestions = suggest_test('Continua', 'Categorica', 2, paired=False,
                                   normal=True, equal_var=True)
        assert suggestions[0][1] == 't_independent'

    def test_normal_unequal_var_recommends_welch(self):
        suggestions = suggest_test('Continua', 'Categorica', 2, paired=False,
                                   normal=True, equal_var=False)
        assert suggestions[0][1] == 't_welch'

    def test_paired_ignores_equal_var(self):
        suggestions = suggest_test('Continua', 'Categorica', 2, paired=True,
                                   normal=True, equal_var=False)
        assert suggestions[0][1] == 't_paired'


# --- S2: Intervalos de confianza --------------------------------------------

class TestConfidenceIntervals:
    def test_two_groups_has_ci(self, sample_df):
        r = run_test('t_independent', sample_df, 'valor', 'grupo', ['A', 'B'])
        assert 'ci_lower' in r
        assert 'ci_upper' in r
        assert r['ci_lower'] < r['ci_upper']
        assert 'mean_diff' in r

    def test_welch_has_ci(self, sample_df):
        r = run_test('t_welch', sample_df, 'valor', 'grupo', ['A', 'B'])
        assert 'ci_lower' in r
        assert 'ci_upper' in r

    def test_ci_contains_mean_diff(self, sample_df):
        r = run_test('t_independent', sample_df, 'valor', 'grupo', ['A', 'B'])
        assert r['ci_lower'] <= r['mean_diff'] <= r['ci_upper']

    def test_correlation_has_ci(self):
        np.random.seed(42)
        df = pd.DataFrame({
            'x': np.random.normal(0, 1, 50),
            'y': np.random.normal(0, 1, 50),
        })
        r = run_test('pearson', df, 'x', 'y')
        assert 'ci_lower' in r
        assert 'ci_upper' in r
        assert -1 <= r['ci_lower'] <= 1
        assert -1 <= r['ci_upper'] <= 1

    def test_regression_has_ci(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 30)
        df = pd.DataFrame({'y': 2 * x + np.random.normal(0, 0.5, 30), 'x': x})
        r = run_test('linear_reg', df, 'y', 'x')
        assert 'ci_lower' in r
        assert 'ci_upper' in r
        # CI should contain the slope
        assert r['ci_lower'] <= r['slope'] <= r['ci_upper']

    def test_bland_altman_has_ci(self):
        np.random.seed(42)
        m1 = np.random.normal(100, 10, 30)
        df = pd.DataFrame({'m1': m1, 'm2': m1 + np.random.normal(0, 2, 30)})
        r = run_test('bland_altman', df, 'm1', 'm2')
        assert 'ci_lower' in r
        assert 'ci_upper' in r
        # CI should contain the bias
        assert r['ci_lower'] <= r['bias'] <= r['ci_upper']
