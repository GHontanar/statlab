"""Configuración UI por tipo de análisis y dispatcher."""

import pandas as pd
import streamlit as st

from stats.tests import POSTHOC_METHODS, check_homogeneity, check_normality, suggest_test
from ui.constants import (
    ANALYSIS_TYPES,
    Q_AGREEMENT,
    Q_ASSOCIATION,
    Q_CORRELATION,
    Q_DIFF_GROUPS,
    Q_PREDICTION,
    Q_RELIABILITY,
    Q_RISK,
    Q_SURVIVAL,
    AnalysisConfig,
)
from utils.data import validate_continuous, validate_group_sizes


def render_analysis_config(df, var_types):
    """Renderiza la configuración del análisis y retorna AnalysisConfig.

    Args:
        df: DataFrame con los datos.
        var_types: dict {columna: 'Continua'|'Categórica'}.

    Returns:
        AnalysisConfig con todos los parámetros necesarios para ejecutar el test.
    """
    st.header("3. Análisis estadístico")

    continuous_vars = [c for c, t in var_types.items() if t == 'Continua']
    categorical_vars = [c for c, t in var_types.items() if t == 'Categórica']
    cols = df.columns.tolist()

    analysis_type = st.radio(
        "¿Qué quieres averiguar?",
        ANALYSIS_TYPES,
        horizontal=True,
        help="Elige la pregunta que mejor describe tu objetivo de investigación."
    )

    _dispatchers = {
        Q_DIFF_GROUPS: _config_diff_groups,
        Q_CORRELATION: _config_correlation,
        Q_ASSOCIATION: _config_association,
        Q_AGREEMENT: _config_agreement,
        Q_PREDICTION: _config_prediction,
        Q_SURVIVAL: _config_survival,
        Q_RISK: _config_risk,
        Q_RELIABILITY: _config_reliability,
    }

    config = _dispatchers[analysis_type](df, cols, continuous_vars, categorical_vars)
    config.analysis_type = analysis_type

    config.alpha = st.slider("Nivel de significancia (alpha)", 0.001, 0.10, 0.05, 0.005,
                             help="Usualmente 0.05. Valores menores (0.01) son más exigentes. Solo cambia esto si sabes por qué.")

    return config


# --- Configuradores por tipo de análisis ---


def _config_diff_groups(df, cols, continuous_vars, categorical_vars):
    col_left, col_right = st.columns(2)

    with col_left:
        if not continuous_vars:
            st.warning("No hay variables continuas definidas")
            st.stop()
        var_dep = st.selectbox("¿Qué mediste?", continuous_vars,
                               help="La variable numérica que quieres comparar: peso, hemoglobina, score...")
    with col_right:
        if not categorical_vars:
            st.warning("No hay variables categóricas definidas")
            st.stop()
        var_group = st.selectbox("¿Cómo se dividen los sujetos?", categorical_vars,
                                 help="La variable que define los grupos: tratamiento, sexo, diagnóstico...")

    ok, err = validate_continuous(df, var_dep)
    if not ok:
        st.error(err)
        st.stop()

    all_groups = sorted(df[var_group].dropna().unique())

    _lower_map = {}
    for g in all_groups:
        key = str(g).strip().lower()
        _lower_map.setdefault(key, []).append(str(g))
    _case_dupes = {k: v for k, v in _lower_map.items() if len(v) > 1}
    if _case_dupes:
        _dupe_list = ", ".join(f"{' / '.join(v)}" for v in _case_dupes.values())
        st.warning(f"Posibles duplicados por mayúsculas/minúsculas: {_dupe_list}. "
                   "Revisa los datos o unifica antes de analizar.")

    selected_groups = st.multiselect(
        f"Grupos a comparar (de `{var_group}`)",
        options=all_groups,
        default=all_groups[:min(4, len(all_groups))]
    )

    if len(selected_groups) < 2:
        st.warning("Selecciona al menos 2 grupos para comparar.")
        st.stop()

    ok, err, counts = validate_group_sizes(df, var_dep, var_group, selected_groups)
    if not ok:
        st.error(err)
        st.stop()

    n_groups = len(selected_groups)

    _min_count = min(counts.values())
    _max_count = max(counts.values())
    if _min_count < 10:
        st.warning(f"Muestra pequeña (n = {_min_count} en algún grupo). "
                   "Los resultados pueden no ser fiables. "
                   "Considera usar un test no paramétrico.")
    if _max_count > 0 and _min_count > 0 and _max_count / _min_count > 3:
        st.warning(f"Grupos muy desbalanceados (n = {_min_count} vs {_max_count}). "
                   "Los resultados pueden verse afectados por el desbalance.")

    st.subheader("Verificación de supuestos")

    all_normal = True
    _norm_results = {}
    for g in selected_groups:
        gdata = df[df[var_group] == g][var_dep].dropna()
        stat, p = check_normality(gdata)
        is_normal = p > 0.05 if p is not None else None
        if is_normal is False:
            all_normal = False
        _norm_results[g] = (stat, p, is_normal, len(gdata))

    _group_data = [df[df[var_group] == g][var_dep].dropna() for g in selected_groups]
    levene_stat, levene_p = check_homogeneity(_group_data)
    equal_var = levene_p > 0.05 if levene_p is not None else True

    _assumption_rows = []
    for g in selected_groups:
        stat, p, is_normal, n = _norm_results[g]
        if p is not None:
            _assumption_rows.append({
                'Grupo': g, 'n': n,
                'Shapiro-Wilk p': f"{p:.4f}",
                'Normal': "Sí" if is_normal else "No",
            })
        else:
            reason = "n < 3" if n < 3 else "n > 5000"
            _assumption_rows.append({
                'Grupo': g, 'n': n,
                'Shapiro-Wilk p': f"N/A ({reason})",
                'Normal': "—",
            })

    if levene_p is not None:
        _levene_summary = f"Levene p = {levene_p:.4f} → varianzas {'iguales' if equal_var else 'distintas'}"
    else:
        _levene_summary = "Levene: N/A"

    st.dataframe(pd.DataFrame(_assumption_rows).set_index('Grupo'),
                 use_container_width=True)
    if levene_p is not None:
        if equal_var:
            st.success(f"**Homogeneidad de varianzas**: {_levene_summary}")
        else:
            st.warning(f"**Varianzas no homogéneas**: {_levene_summary}. Se recomienda Welch o test no paramétrico.")

    if _min_count < 20 and all_normal:
        st.info("Con muestras pequeñas (n < 20), el test de normalidad tiene poco poder. "
                "Si tienes dudas, un test no paramétrico es más seguro.")

    paired = st.checkbox("¿Mediciones repetidas del mismo sujeto?", value=False,
                         help="Marca esto si mediste lo mismo antes y después, o el mismo paciente con dos métodos.")

    paired_id_col = None
    if paired and n_groups == 2:
        id_candidates = [c for c in cols if c != var_dep and c != var_group]
        if id_candidates:
            paired_id_col = st.selectbox(
                "Columna ID del sujeto (para emparejar)",
                ["(orden por posición)"] + id_candidates
            )
            if paired_id_col == "(orden por posición)":
                paired_id_col = None

    suggestions = suggest_test('Continua', 'Categorica', n_groups, paired, all_normal, equal_var)

    selected_test_id = None
    _posthoc_method = None

    if suggestions:
        recommended_name, recommended_id = suggestions[0]

        if all_normal:
            _reason = "Tus datos siguen una distribución normal (Shapiro-Wilk p > 0.05)"
            if paired:
                _reason += " y son mediciones repetidas del mismo sujeto"
            elif n_groups == 2:
                if equal_var:
                    _reason += ", varianzas homogéneas (Levene p > 0.05)"
                else:
                    _reason += ", varianzas NO homogéneas (Levene p < 0.05) → se recomienda Welch"
            else:
                _reason += f" y tienes {n_groups} grupos independientes"
            _reason += " → test paramétrico."
        else:
            _reason = "Tus datos NO siguen una distribución normal (Shapiro-Wilk p < 0.05)"
            _reason += " → test no paramétrico (no asume normalidad)."

        st.success(f"**Recomendado: {recommended_name}**  \n{_reason}")

        if n_groups == 2:
            all_tests = {
                "T-test independiente": "t_independent",
                "T-test Welch": "t_welch",
                "T-test pareado": "t_paired",
                "Mann-Whitney U": "mann_whitney",
                "Wilcoxon signed-rank": "wilcoxon",
            }
        else:
            all_tests = {
                "ANOVA one-way": "anova",
                "Kruskal-Wallis": "kruskal",
            }

        selected_test_id = recommended_id
        with st.expander("Cambiar test (avanzado)"):
            selected_test_name = st.selectbox(
                "Elige otro test",
                options=list(all_tests.keys()),
                index=list(all_tests.keys()).index(recommended_name) if recommended_name in all_tests else 0
            )
            selected_test_id = all_tests[selected_test_name]

        if n_groups > 2:
            _ph_key = 'anova' if selected_test_id == 'anova' else 'kruskal'
            _ph_options = POSTHOC_METHODS[_ph_key]
            _ph_default = 'tukey' if _ph_key == 'anova' else 'dunn_bonferroni'
            _ph_labels = {v[0]: k for k, v in _ph_options.items()}
            with st.expander("Método post-hoc (comparaciones múltiples)"):
                for method_id, (name, desc) in _ph_options.items():
                    st.markdown(f"- **{name}**: {desc}")
                _ph_choice = st.selectbox(
                    "Método post-hoc",
                    options=[v[0] for v in _ph_options.values()],
                    index=list(_ph_options.keys()).index(_ph_default),
                    help="Se aplica solo si el test principal es significativo y hay >2 grupos."
                )
                _posthoc_method = _ph_labels[_ph_choice]
    else:
        st.error("No se encontró un test adecuado para esta configuración.")
        st.stop()

    extra = {}
    if n_groups > 2:
        extra['posthoc_method'] = _posthoc_method

    return AnalysisConfig(
        analysis_type="",
        var_dep=var_dep,
        var_group=var_group,
        selected_test_id=selected_test_id,
        selected_groups=selected_groups,
        paired_id_col=paired_id_col,
        extra=extra,
    )


def _config_correlation(df, cols, continuous_vars, categorical_vars):
    col_left, col_right = st.columns(2)

    with col_left:
        if len(continuous_vars) < 2:
            st.warning("Necesitas al menos 2 variables continuas")
            st.stop()
        var_dep = st.selectbox("Variable a explicar", continuous_vars,
                               help="La variable que quieres predecir o cuya variación quieres entender.")
    with col_right:
        remaining = [c for c in continuous_vars if c != var_dep]
        var_group = st.selectbox("Variable explicativa", remaining,
                                 help="La variable que crees que puede influir o estar relacionada.")

    for _v in [var_dep, var_group]:
        ok, err = validate_continuous(df, _v)
        if not ok:
            st.error(err)
            st.stop()

    all_tests = {
        "Correlación de Pearson": "pearson",
        "Correlación de Spearman": "spearman",
        "Regresión lineal": "linear_reg",
    }
    selected_test_name = st.selectbox("Elige el test", list(all_tests.keys()))
    selected_test_id = all_tests[selected_test_name]

    return AnalysisConfig(
        analysis_type="",
        var_dep=var_dep,
        var_group=var_group,
        selected_test_id=selected_test_id,
    )


def _config_association(df, cols, continuous_vars, categorical_vars):
    col_left, col_right = st.columns(2)

    with col_left:
        if len(categorical_vars) < 2:
            st.warning("Necesitas al menos 2 variables categóricas")
            st.stop()
        var_dep = st.selectbox("Variable 1", categorical_vars)
    with col_right:
        remaining = [c for c in categorical_vars if c != var_dep]
        var_group = st.selectbox("Variable 2", remaining)

    all_tests = {
        "Chi-cuadrado": "chi2",
        "Test exacto de Fisher": "fisher",
    }
    selected_test_name = st.selectbox("Elige el test", list(all_tests.keys()))
    selected_test_id = all_tests[selected_test_name]

    return AnalysisConfig(
        analysis_type="",
        var_dep=var_dep,
        var_group=var_group,
        selected_test_id=selected_test_id,
    )


def _config_agreement(df, cols, continuous_vars, categorical_vars):
    st.info("Compara concordancia entre dos métodos de medición (variables continuas).")
    col_left, col_right = st.columns(2)

    with col_left:
        if len(continuous_vars) < 2:
            st.warning("Necesitas al menos 2 variables continuas")
            st.stop()
        var_dep = st.selectbox("Método 1", continuous_vars)
    with col_right:
        remaining = [c for c in continuous_vars if c != var_dep]
        var_group = st.selectbox("Método 2", remaining)

    for _v in [var_dep, var_group]:
        ok, err = validate_continuous(df, _v)
        if not ok:
            st.error(err)
            st.stop()

    return AnalysisConfig(
        analysis_type="",
        var_dep=var_dep,
        var_group=var_group,
        selected_test_id='bland_altman',
    )


def _config_prediction(df, cols, continuous_vars, categorical_vars):
    st.info("Evalúa capacidad discriminativa de un predictor continuo para un desenlace binario.")
    col_left, col_right = st.columns(2)

    with col_left:
        if not categorical_vars:
            st.warning("Necesitas una variable categórica binaria (desenlace)")
            st.stop()
        var_dep = st.selectbox("Variable de desenlace (binaria)", categorical_vars)
    with col_right:
        if not continuous_vars:
            st.warning("Necesitas una variable continua (predictor)")
            st.stop()
        var_group = st.selectbox("Variable predictora (continua)", continuous_vars)

    n_cats = df[var_dep].dropna().nunique()
    if n_cats != 2:
        st.error(f"La variable '{var_dep}' debe tener exactamente 2 categorías, tiene {n_cats}.")
        st.stop()

    ok, err = validate_continuous(df, var_group)
    if not ok:
        st.error(err)
        st.stop()

    cat_labels = sorted(df[var_dep].dropna().unique())
    positive_label = st.selectbox("Etiqueta positiva", cat_labels,
                                  index=len(cat_labels) - 1)

    return AnalysisConfig(
        analysis_type="",
        var_dep=var_dep,
        var_group=var_group,
        selected_test_id='roc',
        extra={'positive_label': positive_label},
    )


def _config_survival(df, cols, continuous_vars, categorical_vars):
    st.info("Análisis de supervivencia. Necesitas: tiempo hasta evento y variable de evento (0/1).")
    col_left, col_right = st.columns(2)

    with col_left:
        if not continuous_vars:
            st.warning("Necesitas una variable continua (tiempo)")
            st.stop()
        var_dep = st.selectbox("Tiempo hasta evento", continuous_vars)
    with col_right:
        _event_candidates = []
        for c in continuous_vars + categorical_vars:
            _unique = df[c].dropna().unique()
            if set(_unique).issubset({0, 1, 0.0, 1.0, True, False}):
                _event_candidates.append(c)
        if not _event_candidates:
            _event_candidates = continuous_vars + categorical_vars
            st.warning("No se detectaron variables binarias (0/1). Revisa tus datos.")
        var_group = st.selectbox("Variable de evento (0=censurado, 1=evento)",
                                 _event_candidates)

    ok, err = validate_continuous(df, var_dep)
    if not ok:
        st.error(err)
        st.stop()

    km_group_col = None
    km_groups = None
    if categorical_vars:
        _km_group_options = ["(sin comparación)"] + categorical_vars
        km_group_col = st.selectbox("Variable de grupo (opcional, para comparar curvas)",
                                    _km_group_options)
        if km_group_col == "(sin comparación)":
            km_group_col = None
        else:
            all_km_groups = sorted(df[km_group_col].dropna().unique())
            km_groups = st.multiselect("Grupos a comparar", all_km_groups,
                                       default=all_km_groups[:min(4, len(all_km_groups))])
            if len(km_groups) < 1:
                st.warning("Selecciona al menos 1 grupo.")
                st.stop()

    return AnalysisConfig(
        analysis_type="",
        var_dep=var_dep,
        var_group=var_group,
        selected_test_id='kaplan_meier',
        extra={'group_col': km_group_col, 'groups': km_groups},
    )


def _config_risk(df, cols, continuous_vars, categorical_vars):
    st.info("Regresión logística: evalúa si un predictor continuo aumenta el riesgo de un desenlace binario.")
    col_left, col_right = st.columns(2)

    with col_left:
        if not categorical_vars:
            st.warning("Necesitas una variable categórica binaria (desenlace)")
            st.stop()
        var_dep = st.selectbox("Variable de desenlace (binaria)", categorical_vars,
                               help="La variable que quieres predecir: enfermedad sí/no, muerte sí/no...")
    with col_right:
        if not continuous_vars:
            st.warning("Necesitas una variable continua (predictor)")
            st.stop()
        var_group = st.selectbox("Variable predictora (factor de riesgo)", continuous_vars,
                                 help="El factor que crees que puede aumentar el riesgo: edad, IMC, dosis...")

    n_cats = df[var_dep].dropna().nunique()
    if n_cats != 2:
        st.error(f"La variable '{var_dep}' debe tener exactamente 2 categorías, tiene {n_cats}.")
        st.stop()

    ok, err = validate_continuous(df, var_group)
    if not ok:
        st.error(err)
        st.stop()

    return AnalysisConfig(
        analysis_type="",
        var_dep=var_dep,
        var_group=var_group,
        selected_test_id='logistic',
    )


def _config_reliability(df, cols, continuous_vars, categorical_vars):
    st.info("ICC: evalúa reproducibilidad entre evaluadores o métodos. "
            "Necesitas una variable de medición y una variable que identifique al evaluador/método.")
    col_left, col_right = st.columns(2)

    with col_left:
        if not continuous_vars:
            st.warning("Necesitas una variable continua (medición)")
            st.stop()
        var_dep = st.selectbox("Variable de medición", continuous_vars,
                               help="Lo que se midió: score, longitud, concentración...")
    with col_right:
        if not categorical_vars:
            st.warning("Necesitas una variable categórica (evaluador/método)")
            st.stop()
        var_group = st.selectbox("Evaluador / método", categorical_vars,
                                 help="Quién o qué hizo la medición: observador A/B, método 1/2...")

    ok, err = validate_continuous(df, var_dep)
    if not ok:
        st.error(err)
        st.stop()

    n_raters = df[var_group].dropna().nunique()
    if n_raters < 2:
        st.error(f"Se necesitan al menos 2 evaluadores/métodos. '{var_group}' tiene {n_raters}.")
        st.stop()

    return AnalysisConfig(
        analysis_type="",
        var_dep=var_dep,
        var_group=var_group,
        selected_test_id='icc',
    )
