"""UI de la calculadora de tamaño muestral."""

import pandas as pd
import streamlit as st

from stats.sample_size import (
    sample_size_anova,
    sample_size_correlation,
    sample_size_proportions,
    sample_size_survival,
    sample_size_ttest_ind,
    sample_size_ttest_paired,
    sensitivity_table,
)

_STUDY_TYPES = [
    "Comparar 2 grupos independientes",
    "Muestras pareadas (antes/después)",
    "Comparar 3+ grupos (ANOVA)",
    "Correlación",
    "Comparar dos proporciones",
    "Supervivencia (log-rank)",
]

_COHEN_D_PRESETS = {"Pequeño (0.2)": 0.2, "Mediano (0.5)": 0.5, "Grande (0.8)": 0.8, "Personalizado": None}
_COHEN_F_PRESETS = {"Pequeño (0.1)": 0.1, "Mediano (0.25)": 0.25, "Grande (0.4)": 0.4, "Personalizado": None}
_R_PRESETS = {"Pequeño (0.1)": 0.1, "Mediano (0.3)": 0.3, "Grande (0.5)": 0.5, "Personalizado": None}


def render_sample_size():
    """Renderiza la calculadora de tamaño muestral."""
    st.header("Calculadora de tamaño muestral")
    st.caption("Calcula cuántos sujetos necesitas antes de recolectar datos")

    study_type = st.radio("¿Qué tipo de estudio planeas?", _STUDY_TYPES, horizontal=True)

    _dispatchers = {
        _STUDY_TYPES[0]: _config_ttest_ind,
        _STUDY_TYPES[1]: _config_ttest_paired,
        _STUDY_TYPES[2]: _config_anova,
        _STUDY_TYPES[3]: _config_correlation,
        _STUDY_TYPES[4]: _config_proportions,
        _STUDY_TYPES[5]: _config_survival,
    }

    _dispatchers[study_type]()


def _show_result(result, sensitivity_rows=None):
    """Muestra el resultado del cálculo."""
    if not result.get('success'):
        st.error(result.get('error', 'Error en el cálculo'))
        return

    st.subheader("Resultado")

    cols = st.columns(3)
    with cols[0]:
        st.metric("n por grupo", result['n_per_group'])
    with cols[1]:
        st.metric("n total", result['n_total'])
    with cols[2]:
        st.metric("Test", result['test'])

    if result.get('n_events'):
        st.info(f"Eventos necesarios: {result['n_events']}")

    if sensitivity_rows:
        st.subheader("Tabla de sensibilidad")
        st.caption("n por grupo para distintos tamaños del efecto y potencias")

        _col_map = {'effect_size': 'Tamaño efecto'}
        for key in sensitivity_rows[0]:
            if key.startswith('power_'):
                pct = key.split('_')[1]
                _col_map[key] = f"Potencia {pct}%"

        df = pd.DataFrame(sensitivity_rows).rename(columns=_col_map)
        st.dataframe(df, use_container_width=True, hide_index=True)


def _effect_size_selector(presets, label="Tamaño del efecto esperado"):
    """Selector de tamaño del efecto con presets y opción personalizada."""
    col1, col2 = st.columns([2, 1])
    with col1:
        preset_name = st.selectbox(label, list(presets.keys()))
    preset_value = presets[preset_name]

    if preset_value is None:
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            preset_value = st.number_input("Valor", min_value=0.01, max_value=5.0, value=0.5, step=0.05)

    return preset_value


def _common_params():
    """Parámetros comunes: alpha y potencia."""
    col1, col2 = st.columns(2)
    with col1:
        alpha = st.slider("Nivel de significancia (alpha)", 0.001, 0.10, 0.05, 0.005)
    with col2:
        power = st.slider("Potencia deseada", 0.50, 0.99, 0.80, 0.05)
    return alpha, power


# --- Configuradores por tipo de estudio ---


def _config_ttest_ind():
    st.info("Compara la media de una variable continua entre dos grupos independientes.")

    effect_size = _effect_size_selector(_COHEN_D_PRESETS, "Cohen's d esperado")
    alpha, power = _common_params()
    alternative = st.selectbox("Tipo de hipótesis", ["two-sided", "larger", "smaller"],
                               format_func=lambda x: {"two-sided": "Bilateral", "larger": "Unilateral (mayor)", "smaller": "Unilateral (menor)"}[x])

    if st.button("Calcular", type="primary", use_container_width=True):
        result = sample_size_ttest_ind(effect_size, alpha, power, alternative)
        rows = sensitivity_table(sample_size_ttest_ind, [0.2, 0.3, 0.5, 0.8, 1.0], alpha=alpha, alternative=alternative)
        _show_result(result, rows)


def _config_ttest_paired():
    st.info("Mediciones repetidas del mismo sujeto (antes/después, dos condiciones).")

    effect_size = _effect_size_selector(_COHEN_D_PRESETS, "Cohen's d esperado para las diferencias")
    alpha, power = _common_params()

    if st.button("Calcular", type="primary", use_container_width=True):
        result = sample_size_ttest_paired(effect_size, alpha, power)
        rows = sensitivity_table(sample_size_ttest_paired, [0.2, 0.3, 0.5, 0.8, 1.0], alpha=alpha)
        _show_result(result, rows)


def _config_anova():
    st.info("Compara la media de una variable continua entre 3 o más grupos.")

    effect_size = _effect_size_selector(_COHEN_F_PRESETS, "Cohen's f esperado")
    k_groups = st.number_input("Número de grupos", min_value=2, max_value=20, value=3)
    alpha, power = _common_params()

    if st.button("Calcular", type="primary", use_container_width=True):
        result = sample_size_anova(effect_size, k_groups, alpha, power)
        rows = sensitivity_table(sample_size_anova, [0.1, 0.15, 0.25, 0.4, 0.5], k_groups=k_groups, alpha=alpha)
        _show_result(result, rows)


def _config_correlation():
    st.info("Detectar una correlación significativa entre dos variables continuas.")

    r = _effect_size_selector(_R_PRESETS, "Correlación esperada (r)")
    alpha, power = _common_params()

    if st.button("Calcular", type="primary", use_container_width=True):
        result = sample_size_correlation(r, alpha, power)
        rows = sensitivity_table(sample_size_correlation, [0.1, 0.2, 0.3, 0.4, 0.5], alpha=alpha)
        _show_result(result, rows)


def _config_proportions():
    st.info("Compara dos proporciones (ej. tasa de respuesta grupo A vs grupo B).")

    col1, col2 = st.columns(2)
    with col1:
        p1 = st.number_input("Proporción grupo 1", min_value=0.01, max_value=0.99, value=0.30, step=0.05)
    with col2:
        p2 = st.number_input("Proporción grupo 2", min_value=0.01, max_value=0.99, value=0.50, step=0.05)

    alpha, power = _common_params()

    if st.button("Calcular", type="primary", use_container_width=True):
        result = sample_size_proportions(p1, p2, alpha, power)
        _p2_range = [p1 + d for d in [0.05, 0.10, 0.15, 0.20, 0.25] if 0 < p1 + d < 1]
        if _p2_range:
            rows = []
            for p2_val in _p2_range:
                row = {'effect_size': f"{p1:.0%} vs {p2_val:.0%}"}
                for pw in (0.7, 0.8, 0.9):
                    r = sample_size_proportions(p1, p2_val, alpha, pw)
                    row[f"power_{int(pw*100)}"] = r.get('n_per_group', '—') if r.get('success') else '—'
                rows.append(row)
            _show_result(result, rows)
        else:
            _show_result(result)


def _config_survival():
    st.info("Número de eventos necesarios para detectar diferencia en supervivencia (Schoenfeld).")

    col1, col2 = st.columns(2)
    with col1:
        hr = st.number_input("Hazard ratio esperado", min_value=0.05, max_value=10.0, value=0.60, step=0.05,
                             help="HR < 1 = tratamiento protector, HR > 1 = tratamiento peor")
    with col2:
        ratio = st.number_input("Ratio de asignación (n2/n1)", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                                help="1.0 = grupos iguales, 2.0 = doble en grupo 2")

    alpha, power = _common_params()

    if st.button("Calcular", type="primary", use_container_width=True):
        result = sample_size_survival(hr, alpha, power, ratio)
        rows = sensitivity_table(sample_size_survival, [0.4, 0.5, 0.6, 0.7, 0.8], alpha=alpha, ratio=ratio)
        _show_result(result, rows)
