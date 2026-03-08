"""Tests para charts/figures.py"""

import pandas as pd
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from charts.figures import generate_figure, _p_to_stars, _figsize_for


@pytest.fixture
def group_df():
    np.random.seed(42)
    n = 15
    return pd.DataFrame({
        'valor': np.concatenate([
            np.random.normal(10, 1, n),
            np.random.normal(14, 1, n),
            np.random.normal(12, 1, n),
        ]),
        'grupo': ['A'] * n + ['B'] * n + ['C'] * n,
    })


@pytest.fixture
def scatter_df():
    np.random.seed(42)
    x = np.linspace(0, 10, 30)
    y = 2 * x + np.random.normal(0, 1, 30)
    return pd.DataFrame({'x': x, 'y': y})


@pytest.fixture
def result_significant():
    return {'p_value': 0.003, 'significant': True}


@pytest.fixture
def result_not_significant():
    return {'p_value': 0.42, 'significant': False}


# --- generate_figure: todos los tipos generan sin error ----------------------

class TestGenerateFigure:
    @pytest.mark.parametrize("fig_type", [
        'boxplot', 'violin', 'bar_error', 'paired', 'histogram'
    ])
    def test_group_figure_types(self, group_df, result_significant, fig_type):
        fig = generate_figure(fig_type, group_df, 'valor', 'grupo',
                              ['A', 'B'], result_significant)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_scatter(self, scatter_df, result_significant):
        fig = generate_figure('scatter', scatter_df, 'y', 'x',
                              result=result_significant)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_histogram_no_groups(self, scatter_df):
        fig = generate_figure('histogram', scatter_df, 'y', 'x')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_result_no_error(self, group_df):
        fig = generate_figure('boxplot', group_df, 'valor', 'grupo',
                              ['A', 'B'], result=None)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_invalid_data_shows_error_text(self):
        """Columna inexistente no crashea, muestra error en figura."""
        df = pd.DataFrame({'a': [1, 2], 'b': ['X', 'Y']})
        fig = generate_figure('boxplot', df, 'inexistente', 'b', ['X', 'Y'])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# --- DPI y tamano ------------------------------------------------------------

class TestFigureQuality:
    def test_dpi_is_150(self, group_df):
        fig = generate_figure('boxplot', group_df, 'valor', 'grupo', ['A', 'B'])
        assert fig.dpi == 150
        plt.close(fig)

    def test_scatter_is_square(self, scatter_df):
        fig = generate_figure('scatter', scatter_df, 'y', 'x')
        w, h = fig.get_size_inches()
        assert w == h
        plt.close(fig)

    def test_many_groups_widens_figure(self, group_df):
        w, _ = _figsize_for('boxplot', n_groups=6)
        assert w > 8


# --- Bracket de significancia ------------------------------------------------

class TestSignificance:
    def test_bracket_for_2_groups(self, group_df, result_significant):
        """2 grupos con p significativo debe tener bracket (mas de 2 artistas)."""
        fig = generate_figure('boxplot', group_df, 'valor', 'grupo',
                              ['A', 'B'], result_significant)
        ax = fig.axes[0]
        # El bracket agrega lineas y textos al axes
        assert len(ax.texts) >= 1
        plt.close(fig)

    def test_text_for_3_groups(self, group_df, result_significant):
        """3 grupos debe usar texto flotante, no bracket."""
        fig = generate_figure('boxplot', group_df, 'valor', 'grupo',
                              ['A', 'B', 'C'], result_significant)
        ax = fig.axes[0]
        text_content = [t.get_text() for t in ax.texts]
        assert any('p =' in t for t in text_content)
        plt.close(fig)


# --- Helpers -----------------------------------------------------------------

class TestPToStars:
    def test_triple_star(self):
        assert _p_to_stars(0.0001) == '***'

    def test_double_star(self):
        assert _p_to_stars(0.005) == '**'

    def test_single_star(self):
        assert _p_to_stars(0.03) == '*'

    def test_ns(self):
        assert _p_to_stars(0.1) == 'ns'

    def test_boundary_001(self):
        assert _p_to_stars(0.001) == '**'  # 0.001 is not < 0.001

    def test_boundary_01(self):
        assert _p_to_stars(0.01) == '*'  # 0.01 is not < 0.01

    def test_boundary_05(self):
        assert _p_to_stars(0.05) == 'ns'  # 0.05 is not < 0.05


# --- Nuevos tipos de figura -------------------------------------------------

class TestBlandAltmanFigure:
    def test_generates_without_error(self):
        np.random.seed(42)
        m1 = np.random.normal(100, 10, 20)
        m2 = m1 + np.random.normal(1, 3, 20)
        df = pd.DataFrame({'m1': m1, 'm2': m2})
        result = {'bias': 1.0, 'sd_diff': 3.0, 'loa_upper': 6.88,
                  'loa_lower': -4.88, 'p_value': 0.1}
        fig = generate_figure('bland_altman', df, 'm1', 'm2', result=result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_without_result(self):
        df = pd.DataFrame({'m1': [1, 2, 3], 'm2': [1.1, 2.1, 3.1]})
        fig = generate_figure('bland_altman', df, 'm1', 'm2', result=None)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestROCFigure:
    def test_generates_with_result(self):
        result = {
            'fpr': [0.0, 0.1, 0.5, 1.0],
            'tpr': [0.0, 0.8, 0.95, 1.0],
            'auc': 0.9,
            'best_threshold': 5.0,
            'sensitivity': 0.85,
            'specificity': 0.90,
            'p_value': None,
        }
        df = pd.DataFrame({'a': [1], 'b': [1]})
        fig = generate_figure('roc', df, 'a', 'b', result=result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_without_result_no_crash(self):
        df = pd.DataFrame({'a': [1], 'b': [1]})
        fig = generate_figure('roc', df, 'a', 'b', result=None)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestKaplanMeierFigure:
    def test_generates_with_curves(self):
        result = {
            'curves': {
                'A': {'timeline': [0, 5, 10, 20], 'survival': [1.0, 0.9, 0.7, 0.5],
                      'median': 15.0, 'n': 20},
                'B': {'timeline': [0, 3, 8, 15], 'survival': [1.0, 0.8, 0.5, 0.3],
                      'median': 8.0, 'n': 20},
            },
            'p_value': 0.03,
            'statistic': 4.5,
        }
        df = pd.DataFrame({'a': [1], 'b': [1]})
        fig = generate_figure('kaplan_meier', df, 'a', 'b', result=result)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        # Should show log-rank p value
        text_content = [t.get_text() for t in ax.texts]
        assert any('p =' in t for t in text_content)
        plt.close(fig)

    def test_without_result_no_crash(self):
        df = pd.DataFrame({'a': [1], 'b': [1]})
        fig = generate_figure('kaplan_meier', df, 'a', 'b', result=None)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_group_no_pvalue(self):
        result = {
            'curves': {
                'Global': {'timeline': [0, 5, 10], 'survival': [1.0, 0.8, 0.6],
                           'median': 12.0, 'n': 30},
            },
            'p_value': None,
        }
        df = pd.DataFrame({'a': [1], 'b': [1]})
        fig = generate_figure('kaplan_meier', df, 'a', 'b', result=result)
        ax = fig.axes[0]
        text_content = [t.get_text() for t in ax.texts]
        assert not any('p =' in t for t in text_content)
        plt.close(fig)
