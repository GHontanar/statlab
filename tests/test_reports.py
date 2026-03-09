"""Tests para reports/text.py y reports/pdf.py"""

import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reports.text import format_result_text, generate_interpretation
from reports.pdf import generate_pdf_report


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

    def test_bland_altman_format(self):
        r = {'test_name': 'Bland-Altman', 'alpha': 0.05,
             'bias': 1.5, 'sd_diff': 3.0, 'loa_upper': 7.38,
             'loa_lower': -4.38, 'statistic': 2.1, 'p_value': 0.04,
             'significant': True, 'n': 30}
        text = format_result_text(r)
        assert 'Sesgo' in text
        assert 'Limite superior' in text
        assert 'Limite inferior' in text

    def test_roc_format(self):
        r = {'test_name': 'Curva ROC', 'alpha': 0.05,
             'auc': 0.87, 'best_threshold': 5.5,
             'sensitivity': 0.85, 'specificity': 0.82,
             'statistic': 0.87, 'p_value': None, 'significant': None, 'n': 60}
        text = format_result_text(r)
        assert 'AUC' in text
        assert 'Corte optimo' in text
        assert 'Sensibilidad' in text

    def test_kaplan_meier_format(self):
        r = {'test_name': 'Kaplan-Meier', 'alpha': 0.05,
             'curves': {
                 'A': {'n': 20, 'median': 15.0, 'timeline': [], 'survival': []},
                 'B': {'n': 20, 'median': 8.0, 'timeline': [], 'survival': []},
             },
             'logrank_name': 'Log-rank: A vs B',
             'statistic': 4.5, 'p_value': 0.03, 'significant': True, 'n': 40}
        text = format_result_text(r)
        assert 'CURVAS DE SUPERVIVENCIA' in text
        assert 'Log-rank' in text

    def test_kaplan_meier_no_median(self):
        r = {'test_name': 'Kaplan-Meier', 'alpha': 0.05,
             'curves': {
                 'Global': {'n': 30, 'median': None, 'timeline': [], 'survival': []},
             },
             'statistic': None, 'p_value': None, 'significant': None, 'n': 30}
        text = format_result_text(r)
        assert 'no alcanzada' in text


# --- generate_interpretation -------------------------------------------------

class TestGenerateInterpretation:
    def test_ttest_significant(self):
        r = {'test': 't_welch', 'test_name': 'T-test de Welch',
             'var_dep': 'Hemoglobina', 'var_group': 'Grupo',
             'groups': ['Tratamiento', 'Control'],
             'n': [20, 20], 'mean': [13.6, 11.2], 'std': [1.2, 0.8],
             'statistic': 6.12, 'p_value': 0.00001, 'significant': True,
             'cohens_d': 2.35}
        txt = generate_interpretation(r)
        assert 'diferencia estadisticamente significativa' in txt
        assert 'Hemoglobina' in txt
        assert 'Tratamiento' in txt
        assert 'p < 0.001' in txt
        assert 'efecto grande' in txt

    def test_ttest_not_significant(self):
        r = {'test': 't_welch', 'test_name': 'T-test de Welch',
             'var_dep': 'Peso', 'var_group': 'Grupo',
             'groups': ['A', 'B'], 'n': [15, 15],
             'mean': [70.1, 70.5], 'std': [5.0, 4.8],
             'statistic': 0.23, 'p_value': 0.82, 'significant': False,
             'cohens_d': 0.08}
        txt = generate_interpretation(r)
        assert 'No se encontraron diferencias' in txt
        assert 'p = 0.820' in txt

    def test_anova_with_posthoc(self):
        r = {'test': 'anova', 'test_name': 'ANOVA one-way',
             'var_dep': 'Score', 'var_group': 'Grupo',
             'groups': ['A', 'B', 'C'],
             'statistic': 15.3, 'p_value': 0.0001, 'significant': True,
             'eta_squared': 0.35,
             'posthoc': {'A': {'A': 1.0, 'B': 0.001, 'C': 0.0001},
                         'B': {'A': 0.001, 'B': 1.0, 'C': 0.04},
                         'C': {'A': 0.0001, 'B': 0.04, 'C': 1.0}},
             'posthoc_name': 'Tukey HSD'}
        txt = generate_interpretation(r)
        assert 'diferencias estadisticamente significativas' in txt
        assert 'eta2' in txt
        assert 'post-hoc' in txt
        assert 'A vs B' in txt

    def test_pearson_significant(self):
        r = {'test': 'pearson', 'test_name': 'Correlacion de Pearson',
             'var_dep': 'Colesterol', 'var_group': 'IMC',
             'statistic': 0.72, 'p_value': 0.0001,
             'significant': True, 'r_squared': 0.5184, 'n': 50}
        txt = generate_interpretation(r)
        assert 'correlacion positiva fuerte' in txt
        assert 'IMC' in txt
        assert '51.8%' in txt

    def test_spearman_negative(self):
        r = {'test': 'spearman', 'test_name': 'Correlacion de Spearman',
             'var_dep': 'Y', 'var_group': 'X',
             'statistic': -0.45, 'p_value': 0.03,
             'significant': True, 'r_squared': 0.2025, 'n': 30}
        txt = generate_interpretation(r)
        assert 'negativa' in txt
        assert 'moderada' in txt

    def test_regression(self):
        r = {'test': 'linear_reg', 'test_name': 'Regresion lineal',
             'var_dep': 'Peso', 'var_group': 'Edad',
             'slope': 0.5, 'intercept': 50.0, 'std_error': 0.1,
             'statistic': 0.8, 'r_squared': 0.64,
             'p_value': 0.001, 'significant': True, 'n': 40}
        txt = generate_interpretation(r)
        assert 'predice significativamente' in txt
        assert 'pendiente' in txt
        assert 'unidad de incremento' in txt

    def test_chi2(self):
        r = {'test': 'chi2', 'test_name': 'Chi-cuadrado',
             'var_dep': 'Desenlace', 'var_group': 'Tratamiento',
             'statistic': 8.5, 'p_value': 0.004, 'significant': True}
        txt = generate_interpretation(r)
        assert 'asociacion estadisticamente significativa' in txt

    def test_bland_altman(self):
        r = {'test': 'bland_altman', 'test_name': 'Bland-Altman',
             'var_dep': 'Metodo1', 'var_group': 'Metodo2',
             'bias': 1.5, 'sd_diff': 3.0, 'loa_lower': -4.38,
             'loa_upper': 7.38, 'statistic': 2.1,
             'p_value': 0.04, 'significant': True, 'n': 30}
        txt = generate_interpretation(r)
        assert 'sesgo medio' in txt
        assert 'limites de acuerdo' in txt
        assert 'significativo' in txt

    def test_roc(self):
        r = {'test': 'roc', 'test_name': 'Curva ROC',
             'var_dep': 'Enfermedad', 'var_group': 'Biomarcador',
             'auc': 0.87, 'best_threshold': 5.5,
             'sensitivity': 0.85, 'specificity': 0.82,
             'positive_label': 'Si', 'n': 60,
             'statistic': 0.87, 'p_value': None, 'significant': None}
        txt = generate_interpretation(r)
        assert 'buena' in txt
        assert 'AUC = 0.870' in txt
        assert 'corte optimo' in txt.lower()

    def test_kaplan_meier_with_groups(self):
        r = {'test': 'kaplan_meier', 'test_name': 'Kaplan-Meier',
             'var_dep': 'Tiempo', 'var_group': 'Evento',
             'curves': {
                 'A': {'n': 20, 'median': 15.0},
                 'B': {'n': 20, 'median': 8.0},
             },
             'statistic': 5.2, 'p_value': 0.02, 'significant': True, 'n': 40}
        txt = generate_interpretation(r)
        assert 'supervivencia' in txt
        assert 'log-rank' in txt
        assert 'significativas' in txt

    def test_kaplan_meier_no_groups(self):
        r = {'test': 'kaplan_meier', 'test_name': 'Kaplan-Meier',
             'var_dep': 'Tiempo', 'var_group': 'Evento',
             'curves': {'Global': {'n': 40, 'median': 12.0}},
             'statistic': None, 'p_value': None, 'significant': None, 'n': 40}
        txt = generate_interpretation(r)
        assert 'supervivencia' in txt
        assert 'Global' in txt

    def test_logistic_regression(self):
        r = {'test': 'logistic', 'test_name': 'Regresion logistica',
             'var_dep': 'Diabetes', 'var_group': 'IMC',
             'odds_ratio': 1.15, 'or_ci_lower': 1.05, 'or_ci_upper': 1.27,
             'positive_label': 'Si',
             'p_value': 0.003, 'significant': True,
             'statistic': 0.14, 'n': 100, 'pseudo_r2': 0.12}
        txt = generate_interpretation(r)
        assert 'predictor significativo' in txt
        assert 'OR = 1.15' in txt
        assert 'riesgo aumenta' in txt

    def test_logistic_not_significant(self):
        r = {'test': 'logistic', 'test_name': 'Regresion logistica',
             'var_dep': 'Diabetes', 'var_group': 'Altura',
             'odds_ratio': 1.01, 'or_ci_lower': 0.95, 'or_ci_upper': 1.08,
             'positive_label': 'Si',
             'p_value': 0.65, 'significant': False,
             'statistic': 0.01, 'n': 100}
        txt = generate_interpretation(r)
        assert 'no fue un predictor significativo' in txt

    def test_icc_interpretation(self):
        r = {'test': 'icc', 'test_name': 'ICC',
             'var_dep': 'Score', 'var_group': 'Evaluador',
             'icc': 0.85, 'quality': 'buena',
             'ci_lower': 0.70, 'ci_upper': 0.93,
             'n_subjects': 20, 'n_raters': 2,
             'statistic': 0.85, 'p_value': None, 'significant': None}
        txt = generate_interpretation(r)
        assert 'fiabilidad buena' in txt
        assert 'ICC' in txt
        assert '0.850' in txt

    def test_unknown_test_returns_empty(self):
        r = {'test': 'unknown_test'}
        txt = generate_interpretation(r)
        assert txt == ""


# --- generate_pdf_report ----------------------------------------------------

class TestGeneratePdfReport:
    def test_generates_pdf_bytes(self):
        results = [{
            'test': 't_welch', 'test_name': 'T-test de Welch',
            'var_dep': 'Peso', 'var_group': 'Grupo',
            'groups': ['A', 'B'], 'n': [10, 10],
            'mean': [70.0, 75.0], 'std': [5.0, 4.0],
            'median': [70.0, 75.0],
            'statistic': 2.5, 'p_value': 0.02, 'significant': True,
            'cohens_d': 1.1, 'alpha': 0.05, 'success': True,
        }]
        buf = generate_pdf_report(results, [])
        data = buf.read()
        assert data[:5] == b'%PDF-'
        assert len(data) > 500

    def test_with_figures(self):
        results = [{
            'test': 'pearson', 'test_name': 'Pearson',
            'var_dep': 'Y', 'var_group': 'X',
            'statistic': 0.9, 'p_value': 0.001, 'significant': True,
            'r_squared': 0.81, 'n': 30, 'alpha': 0.05, 'success': True,
        }]
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        buf = generate_pdf_report(results, [fig])
        data = buf.read()
        assert data[:5] == b'%PDF-'
        assert len(data) > 1000
        plt.close(fig)

    def test_empty_results(self):
        buf = generate_pdf_report([], [])
        data = buf.read()
        assert data[:5] == b'%PDF-'

    def test_failed_results_skipped(self):
        results = [
            {'test': 'fail', 'success': False, 'error': 'bad'},
            {'test': 't_welch', 'test_name': 'Welch',
             'var_dep': 'X', 'var_group': 'G',
             'groups': ['A', 'B'], 'n': [5, 5],
             'mean': [1.0, 2.0], 'std': [0.5, 0.5],
             'statistic': 2.0, 'p_value': 0.05, 'significant': False,
             'alpha': 0.05, 'success': True},
        ]
        buf = generate_pdf_report(results, [])
        data = buf.read()
        assert data[:5] == b'%PDF-'

    def test_multiple_result_types(self):
        results = [
            {'test': 'anova', 'test_name': 'ANOVA',
             'var_dep': 'Score', 'var_group': 'Grupo',
             'groups': ['A', 'B', 'C'], 'n': [10, 10, 10],
             'mean': [5.0, 7.0, 9.0], 'std': [1.0, 1.0, 1.0],
             'statistic': 20.0, 'p_value': 0.0001, 'significant': True,
             'eta_squared': 0.6, 'alpha': 0.05, 'success': True},
            {'test': 'roc', 'test_name': 'Curva ROC',
             'var_dep': 'Enf', 'var_group': 'Bio',
             'auc': 0.85, 'best_threshold': 3.0,
             'sensitivity': 0.8, 'specificity': 0.85,
             'positive_label': 'Si', 'n': 50,
             'statistic': 0.85, 'p_value': None, 'significant': None,
             'alpha': 0.05, 'success': True},
        ]
        buf = generate_pdf_report(results, [])
        data = buf.read()
        assert len(data) > 1000
