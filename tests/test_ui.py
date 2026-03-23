"""Tests para funcionalidades UI: validación de archivo, historial, exportación Excel."""

from io import BytesIO
from unittest.mock import MagicMock

import pandas as pd

from ui.data_upload import MAX_FILE_SIZE_MB, validate_file_size
from ui.summary_and_export import _build_summary_df, _remove_result, _summary_to_excel


class TestValidateFileSize:
    """Tests para U6: validación de tamaño de archivo."""

    def _make_file(self, size_bytes):
        buf = BytesIO(b'\0' * size_bytes)
        return buf

    def test_small_file_passes(self):
        f = self._make_file(1024)
        assert validate_file_size(f) is True

    def test_exact_limit_passes(self):
        f = self._make_file(MAX_FILE_SIZE_MB * 1024 * 1024)
        assert validate_file_size(f) is True

    def test_over_limit_fails(self, monkeypatch):
        monkeypatch.setattr("streamlit.error", MagicMock())
        f = self._make_file(MAX_FILE_SIZE_MB * 1024 * 1024 + 1)
        assert validate_file_size(f) is False

    def test_file_position_reset_after_check(self):
        f = self._make_file(512)
        validate_file_size(f)
        assert f.tell() == 0


class TestBuildSummaryDf:
    """Tests para la tabla resumen (base de U8)."""

    def _result(self, **overrides):
        base = {
            'success': True,
            'test_name': 'T-test',
            'var_dep': 'peso',
            'var_group': 'grupo',
            'p_value': 0.0312,
            'significant': True,
            'ci_lower': -2.5,
            'ci_upper': -0.1,
            'cohens_d': 0.75,
        }
        base.update(overrides)
        return base

    def test_single_result(self):
        df = _build_summary_df([self._result()])
        assert len(df) == 1
        assert df.iloc[0]['Test'] == 'T-test'
        assert df.iloc[0]['Sig.'] == 'Sí'

    def test_no_ci(self):
        df = _build_summary_df([self._result(ci_lower=None, ci_upper=None)])
        assert df.iloc[0]['IC 95%'] == '—'

    def test_eta_squared_effect(self):
        df = _build_summary_df([self._result(cohens_d=None, eta_squared=0.14)])
        assert 'η²' in df.iloc[0]['Efecto']

    def test_r_squared_effect(self):
        df = _build_summary_df([self._result(cohens_d=None, r_squared=0.65)])
        assert 'R²' in df.iloc[0]['Efecto']

    def test_auc_effect(self):
        df = _build_summary_df([self._result(cohens_d=None, auc=0.85)])
        assert 'AUC' in df.iloc[0]['Efecto']

    def test_no_effect(self):
        df = _build_summary_df([self._result(cohens_d=None)])
        assert df.iloc[0]['Efecto'] == '—'

    def test_not_significant(self):
        df = _build_summary_df([self._result(significant=False)])
        assert df.iloc[0]['Sig.'] == 'No'

    def test_no_p_value(self):
        df = _build_summary_df([self._result(p_value=None)])
        assert df.iloc[0]['p-valor'] == '—'

    def test_multiple_results(self):
        df = _build_summary_df([self._result(), self._result(test_name='ANOVA')])
        assert len(df) == 2


class TestSummaryToExcel:
    """Tests para U8: exportar tabla resumen a Excel."""

    def test_returns_bytes(self):
        df = pd.DataFrame({'A': [1], 'B': [2]})
        data = _summary_to_excel(df)
        assert isinstance(data, bytes)

    def test_valid_xlsx(self):
        df = pd.DataFrame({'Test': ['T-test'], 'p-valor': ['0.031']})
        data = _summary_to_excel(df)
        result = pd.read_excel(BytesIO(data), engine='openpyxl')
        assert list(result.columns) == ['Test', 'p-valor']
        assert result.iloc[0]['Test'] == 'T-test'

    def test_empty_df(self):
        df = pd.DataFrame()
        data = _summary_to_excel(df)
        assert isinstance(data, bytes)


class TestRemoveResult:
    """Tests para U7: eliminar resultados individuales."""

    def _setup_state(self, monkeypatch):
        state = {
            'results': [
                {'success': True, 'test_name': 'A'},
                {'success': False, 'test_name': 'B'},
                {'success': True, 'test_name': 'C'},
            ],
            'figures': ['fig_A', 'fig_B', 'fig_C'],
        }
        mock_session = MagicMock()
        mock_session.results = state['results']
        mock_session.figures = state['figures']
        monkeypatch.setattr("ui.summary_and_export.st.session_state", mock_session)
        return mock_session

    def test_remove_first_valid(self, monkeypatch):
        session = self._setup_state(monkeypatch)
        valid = [r for r in session.results if r.get('success')]
        _remove_result(0, valid)
        assert len(session.results) == 2
        assert session.results[0]['test_name'] == 'B'

    def test_remove_second_valid(self, monkeypatch):
        session = self._setup_state(monkeypatch)
        valid = [r for r in session.results if r.get('success')]
        _remove_result(1, valid)
        assert len(session.results) == 2
        assert session.results[0]['test_name'] == 'A'

    def test_figure_removed(self, monkeypatch):
        session = self._setup_state(monkeypatch)
        valid = [r for r in session.results if r.get('success')]
        _remove_result(0, valid)
        assert len(session.figures) == 2
