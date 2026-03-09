"""
StatLab - Analisis Estadistico y Generacion de Figuras
"""

import os
import streamlit as st
import pandas as pd
from io import BytesIO
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

from utils.data import infer_variable_type, validate_continuous, validate_group_sizes
from stats.tests import check_normality, check_homogeneity, suggest_test, run_test, POSTHOC_METHODS
from charts.figures import generate_figure
from reports.text import format_result_text, generate_interpretation
import matplotlib.pyplot as plt
from reports.pdf import generate_pdf_report

# Constantes para tipos de analisis
Q_DIFF_GROUPS = "Hay diferencia entre grupos?"
Q_CORRELATION = "Estan relacionadas dos mediciones?"
Q_ASSOCIATION = "Se asocian dos categorias?"
Q_AGREEMENT = "Dos metodos miden lo mismo?"
Q_PREDICTION = "Un valor predice un desenlace?"
Q_SURVIVAL = "Cuanto tiempo hasta un evento?"
Q_RISK = "Un factor aumenta el riesgo?"
Q_RELIABILITY = "Las mediciones son reproducibles?"


def _close_figures():
    """Cierra las figuras matplotlib almacenadas en session_state."""
    for fig in st.session_state.get('figures', []):
        plt.close(fig)

# --- Configuracion general ---------------------------------------------------
st.set_page_config(
    page_title="StatLab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .stMetric { background: #f8f9fa; padding: 1rem; border-radius: 8px; }
    h1 { color: #1a365d; }
    h2, h3 { color: #2c5282; }
    .result-box {
        background: #f0f7ff;
        border-left: 4px solid #3182ce;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .sig-yes { color: #e53e3e; font-weight: bold; }
    .sig-no { color: #38a169; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- Estado de la sesion ------------------------------------------------------
if 'df' not in st.session_state:
    st.session_state.df = None
if 'var_types' not in st.session_state:
    st.session_state.var_types = {}
if 'results' not in st.session_state:
    st.session_state.results = []
if 'figures' not in st.session_state:
    st.session_state.figures = []

# === GUIA DE TESTS (sidebar) ==================================================

with st.sidebar:
    st.header("Guia de tests")
    st.caption("Referencia rapida para elegir el test adecuado")

    with st.expander("Comparar 2 grupos"):
        st.markdown("""
**Datos independientes** (sujetos distintos en cada grupo)

| Condicion | Test |
|-----------|------|
| Datos normales, varianzas iguales | T-test independiente |
| Datos normales, varianzas distintas | T-test de Welch |
| Datos no normales u ordinales | Mann-Whitney U |

*Ejemplo: Comparar hemoglobina entre grupo tratamiento y control.*

**Datos pareados** (mismo sujeto medido 2 veces)

| Condicion | Test |
|-----------|------|
| Datos normales | T-test pareado |
| Datos no normales | Wilcoxon signed-rank |

*Ejemplo: Presion arterial antes y despues de un farmaco.*
""")

    with st.expander("Comparar 3+ grupos"):
        st.markdown("""
| Condicion | Test | Post-hoc |
|-----------|------|----------|
| Datos normales | ANOVA one-way | Tukey HSD |
| Datos no normales | Kruskal-Wallis | Dunn (Bonferroni) |

*Ejemplo: Comparar eficacia de 3 farmacos distintos.*

El post-hoc solo se ejecuta si el test principal es significativo
(p < alpha). Indica **que pares** de grupos difieren entre si.
""")

    with st.expander("Correlacion y regresion"):
        st.markdown("""
| Condicion | Test |
|-----------|------|
| Relacion lineal, datos normales | Pearson |
| Relacion monotona, datos no normales | Spearman |
| Predecir Y a partir de X | Regresion lineal |

- **r** cercano a +1/-1 = correlacion fuerte
- **R2** = proporcion de varianza explicada
- **p-valor** = probabilidad de obtener esa r por azar

*Ejemplo: Correlacion entre IMC y colesterol.*
""")

    with st.expander("Variables categoricas"):
        st.markdown("""
| Condicion | Test |
|-----------|------|
| Tabla >2x2, o n grande | Chi-cuadrado |
| Tabla 2x2 con n pequeno (<20) | Fisher exacto |

Comparan si la distribucion de una variable categorica
es independiente de otra.

*Ejemplo: Asociacion entre tratamiento (A/B) y desenlace
(mejora/no mejora).*
""")

    with st.expander("Normalidad"):
        st.markdown("""
StatLab usa el **test de Shapiro-Wilk** automaticamente.

- **p > 0.05**: No se rechaza normalidad -> test parametrico
- **p < 0.05**: Se rechaza normalidad -> test no parametrico

Funciona bien con n entre 3 y 5000.
Con n muy grande, casi todo resulta "no normal" por exceso de poder.
""")

    with st.expander("Bland-Altman"):
        st.markdown("""
Evalua **concordancia** entre dos metodos de medicion.

- **Sesgo (bias)**: diferencia media entre metodos
- **Limites de acuerdo**: sesgo +/- 1.96 DE
- Si el sesgo no es significativo y los limites son clinicamente aceptables,
  los metodos son intercambiables.

*Ejemplo: Comparar medicion de glucosa con dos dispositivos distintos.*
""")

    with st.expander("Curva ROC"):
        st.markdown("""
Evalua la **capacidad discriminativa** de un predictor continuo
para un desenlace binario.

| Metrica | Descripcion |
|---------|------------|
| AUC | Area bajo la curva (0.5=azar, 1.0=perfecto) |
| Sensibilidad | Verdaderos positivos correctamente identificados |
| Especificidad | Verdaderos negativos correctamente identificados |
| Corte optimo | Threshold que maximiza Youden's J |

*Ejemplo: Evaluar si un biomarcador predice enfermedad (si/no).*
""")

    with st.expander("Kaplan-Meier"):
        st.markdown("""
**Analisis de supervivencia**: estima la probabilidad de que un
evento no haya ocurrido en funcion del tiempo.

- Necesitas: **tiempo** hasta evento y **estado** del evento (0=censurado, 1=evento)
- Opcional: variable de grupo para comparar curvas
- **Log-rank test**: compara curvas entre 2 grupos

*Ejemplo: Tiempo hasta recaida en pacientes con dos tratamientos.*
""")

    with st.expander("Regresion logistica"):
        st.markdown("""
Evalua si un **factor continuo** aumenta o disminuye el riesgo
de un **desenlace binario**.

| Metrica | Descripcion |
|---------|------------|
| OR (Odds Ratio) | >1 = mas riesgo, <1 = menos riesgo, =1 sin efecto |
| IC 95% del OR | Si incluye 1, el efecto no es significativo |
| Pseudo R2 | Proporcion de varianza explicada (McFadden) |

*Ejemplo: ¿La edad aumenta el riesgo de diabetes (si/no)?*
""")

    with st.expander("ICC (fiabilidad)"):
        st.markdown("""
El **coeficiente de correlacion intraclase** mide la concordancia
entre evaluadores o metodos de medicion repetidos.

| ICC | Interpretacion |
|-----|---------------|
| < 0.50 | Pobre |
| 0.50 - 0.75 | Moderada |
| 0.75 - 0.90 | Buena |
| > 0.90 | Excelente |

Datos deben estar **balanceados**: mismo n de sujetos por evaluador.

*Ejemplo: Dos radiologos miden el tamano de un tumor en 20 pacientes.*
""")

    with st.expander("Tamano del efecto"):
        st.markdown("""
El p-valor indica si hay diferencia, pero no su magnitud.

| Metrica | Pequeno | Mediano | Grande |
|---------|---------|---------|--------|
| d de Cohen | <0.5 | 0.5-0.8 | >0.8 |
| eta2 (ANOVA) | <0.06 | 0.06-0.14 | >0.14 |
| R2 | <0.09 | 0.09-0.25 | >0.25 |

*Un p-valor muy bajo con efecto pequeno puede no ser
clinicamente relevante.*
""")

    st.markdown("---")
    st.header("Personalizar figuras")

    custom_title = st.text_input("Titulo", value="", placeholder="Auto")
    custom_xlabel = st.text_input("Eje X", value="", placeholder="Auto")
    custom_ylabel = st.text_input("Eje Y", value="", placeholder="Auto")

    _palettes = {
        "StatLab (defecto)": None,
        "Azules": "Blues",
        "Rojos": "Reds",
        "Verdes": "Greens",
        "Pastel": "pastel",
        "Set2": "Set2",
        "Escala de grises": "Greys",
    }
    _pal_choice = st.selectbox("Paleta de colores", list(_palettes.keys()))
    custom_palette = _palettes[_pal_choice]

    fig_options = {
        'title': custom_title or None,
        'xlabel': custom_xlabel or None,
        'ylabel': custom_ylabel or None,
        'palette': custom_palette,
    }

# === INTERFAZ PRINCIPAL =======================================================

st.title("📊 StatLab")
st.caption("Analisis estadistico y generacion de figuras")

# --- PASO 1: Carga de datos --------------------------------------------------
st.header("1. Datos")

upload_col, sample_col = st.columns([3, 1])
with upload_col:
    uploaded_file = st.file_uploader("Sube tu archivo", type=['csv', 'xlsx', 'xls'],
                                     help="Formatos aceptados: CSV, XLSX, XLS")
with sample_col:
    st.markdown("<br>", unsafe_allow_html=True)
    _sample_files = {f: f for f in ['sample_data.csv', 'test_completo.csv']
                     if os.path.exists(f)}
    if _sample_files:
        _chosen = st.selectbox("Datos de ejemplo", [""] + list(_sample_files.keys()),
                               label_visibility="collapsed")
        if _chosen and st.button("Cargar ejemplo", use_container_width=True):
            st.session_state.df = pd.read_csv(_chosen)
            st.success(f"Cargado: {_chosen} — "
                       f"{st.session_state.df.shape[0]} filas x {st.session_state.df.shape[1]} columnas")

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            sample = uploaded_file.read(4096).decode('utf-8', errors='ignore')
            uploaded_file.seek(0)
            sep = ',' if sample.count(',') > sample.count(';') else ';'
            st.session_state.df = pd.read_csv(uploaded_file, sep=sep)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)
        st.success(f"Cargado: {uploaded_file.name} — "
                   f"{st.session_state.df.shape[0]} filas x {st.session_state.df.shape[1]} columnas")
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")

df = st.session_state.df

if df is not None:
    with st.expander("Vista previa de los datos", expanded=True):
        st.dataframe(df.head(20), use_container_width=True)

    with st.expander("Resumen descriptivo"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Filas", df.shape[0])
        col2.metric("Columnas", df.shape[1])
        col3.metric("Valores faltantes", int(df.isna().sum().sum()))
        st.dataframe(df.describe(include='all').round(3), use_container_width=True)

    # --- PASO 2: Definicion de variables --------------------------------------
    st.header("2. Definicion de variables")

    cols = df.columns.tolist()
    var_types = {}
    # P7: Separar variables claras de ambiguas
    _clear_vars = []
    _ambiguous_vars = []
    for col_name in cols:
        dtype = df[col_name].dtype
        # Claramente categorica: string/object/category/bool
        if dtype in ('object', 'category', 'bool'):
            _clear_vars.append((col_name, 'Categorica'))
        # Claramente continua: float
        elif dtype in ('float64', 'float32'):
            _clear_vars.append((col_name, 'Continua'))
        # Ambiguo: integers, bools, etc.
        else:
            _ambiguous_vars.append((col_name, infer_variable_type(df[col_name])))

    # Mostrar variables claras como resumen
    if _clear_vars:
        _cont = [c for c, t in _clear_vars if t == 'Continua']
        _cat = [c for c, t in _clear_vars if t == 'Categorica']
        _summary = []
        if _cont:
            _summary.append(f"**Continuas**: {', '.join(_cont)}")
        if _cat:
            _summary.append(f"**Categoricas**: {', '.join(_cat)}")
        st.success("Detectadas automaticamente: " + " | ".join(_summary))

    # Asignar las claras
    for col_name, vtype in _clear_vars:
        var_types[col_name] = vtype

    # Solo mostrar selectbox para las ambiguas + expander para corregir las claras
    if _ambiguous_vars:
        st.info("Revisa estas variables (la inferencia puede no ser correcta):")
        amb_grid = st.columns(min(3, len(_ambiguous_vars)))
        for i, (col_name, inferred) in enumerate(_ambiguous_vars):
            with amb_grid[i % min(3, len(_ambiguous_vars))]:
                var_types[col_name] = st.selectbox(
                    f"`{col_name}`",
                    ['Continua', 'Categorica'],
                    index=0 if inferred == 'Continua' else 1,
                    key=f"vtype_{col_name}"
                )

    with st.expander("Corregir tipos de variable"):
        corr_grid = st.columns(3)
        for i, col_name in enumerate(cols):
            current = var_types.get(col_name, 'Continua')
            with corr_grid[i % 3]:
                var_types[col_name] = st.selectbox(
                    f"`{col_name}`",
                    ['Continua', 'Categorica'],
                    index=0 if current == 'Continua' else 1,
                    key=f"vtype_corr_{col_name}"
                )

    st.session_state.var_types = var_types

    # --- PASO 3: Configuracion del analisis -----------------------------------
    st.header("3. Analisis estadistico")

    continuous_vars = [c for c, t in var_types.items() if t == 'Continua']
    categorical_vars = [c for c, t in var_types.items() if t == 'Categorica']

    analysis_type = st.radio(
        "Que quieres averiguar?",
        [Q_DIFF_GROUPS, Q_CORRELATION, Q_ASSOCIATION,
         Q_AGREEMENT, Q_PREDICTION, Q_SURVIVAL,
         Q_RISK, Q_RELIABILITY],
        horizontal=True,
        help="Elige la pregunta que mejor describe tu objetivo de investigacion."
    )

    # Inicializar variables compartidas
    paired_id_col = None
    _posthoc_method = None
    n_groups = None
    selected_groups = None

    col_left, col_right = st.columns(2)

    if analysis_type == Q_DIFF_GROUPS:
        with col_left:
            if not continuous_vars:
                st.warning("No hay variables continuas definidas")
                st.stop()
            var_dep = st.selectbox("Que mediste?", continuous_vars,
                                   help="La variable numerica que quieres comparar: peso, hemoglobina, score...")
        with col_right:
            if not categorical_vars:
                st.warning("No hay variables categoricas definidas")
                st.stop()
            var_group = st.selectbox("Como se dividen los sujetos?", categorical_vars,
                                     help="La variable que define los grupos: tratamiento, sexo, diagnostico...")

        # A3: Validar que la variable continua sea numerica
        ok, err = validate_continuous(df, var_dep)
        if not ok:
            st.error(err)
            st.stop()

        all_groups = sorted(df[var_group].dropna().unique())

        # B8: Detectar categorias que solo difieren en mayusculas/minusculas
        _lower_map = {}
        for g in all_groups:
            key = str(g).strip().lower()
            _lower_map.setdefault(key, []).append(str(g))
        _case_dupes = {k: v for k, v in _lower_map.items() if len(v) > 1}
        if _case_dupes:
            _dupe_list = ", ".join(f"{' / '.join(v)}" for v in _case_dupes.values())
            st.warning(f"Posibles duplicados por mayusculas/minusculas: {_dupe_list}. "
                       "Revisa los datos o unifica antes de analizar.")

        selected_groups = st.multiselect(
            f"Grupos a comparar (de `{var_group}`)",
            options=all_groups,
            default=all_groups[:min(4, len(all_groups))]
        )

        if len(selected_groups) < 2:
            st.warning("Selecciona al menos 2 grupos para comparar.")
            st.stop()

        # A3: Validar n minimo por grupo
        ok, err, counts = validate_group_sizes(df, var_dep, var_group, selected_groups)
        if not ok:
            st.error(err)
            st.stop()

        n_groups = len(selected_groups)

        # P8: Advertencias contextuales
        _min_count = min(counts.values())
        _max_count = max(counts.values())
        if _min_count < 10:
            st.warning(f"Muestra pequena (n = {_min_count} en algun grupo). "
                       "Los resultados pueden no ser fiables. "
                       "Considera usar un test no parametrico.")
        if _max_count > 0 and _min_count > 0 and _max_count / _min_count > 3:
            st.warning(f"Grupos muy desbalanceados (n = {_min_count} vs {_max_count}). "
                       "Los resultados pueden verse afectados por el desbalance.")

        # S3: Seccion de supuestos
        st.subheader("Verificacion de supuestos")

        # Normalidad (Shapiro-Wilk por grupo)
        all_normal = True
        _norm_results = {}
        for g in selected_groups:
            gdata = df[df[var_group] == g][var_dep].dropna()
            stat, p = check_normality(gdata)
            is_normal = p > 0.05 if p is not None else None
            if is_normal is False:
                all_normal = False
            _norm_results[g] = (stat, p, is_normal, len(gdata))

        # S1: Homogeneidad de varianzas (Levene)
        _group_data = [df[df[var_group] == g][var_dep].dropna() for g in selected_groups]
        levene_stat, levene_p = check_homogeneity(_group_data)
        equal_var = levene_p > 0.05 if levene_p is not None else True

        # Tabla resumen de supuestos
        _assumption_rows = []
        for g in selected_groups:
            stat, p, is_normal, n = _norm_results[g]
            if p is not None:
                _assumption_rows.append({
                    'Grupo': g, 'n': n,
                    'Shapiro-Wilk p': f"{p:.4f}",
                    'Normal': "Si" if is_normal else "No",
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
                st.warning(f"**Varianzas no homogeneas**: {_levene_summary}. Se recomienda Welch o test no parametrico.")

        # P8: Advertencia normalidad con n pequeno
        if _min_count < 20 and all_normal:
            st.info("Con muestras pequenas (n < 20), el test de normalidad tiene poco poder. "
                    "Si tienes dudas, un test no parametrico es mas seguro.")

        paired = st.checkbox("Mediciones repetidas del mismo sujeto?", value=False,
                             help="Marca esto si mediste lo mismo antes y despues, o el mismo paciente con dos metodos.")

        # U3: Selector de columna ID para emparejar sujetos
        paired_id_col = None
        if paired and n_groups == 2:
            id_candidates = [c for c in cols if c != var_dep and c != var_group]
            if id_candidates:
                paired_id_col = st.selectbox(
                    "Columna ID del sujeto (para emparejar)",
                    ["(orden por posicion)"] + id_candidates
                )
                if paired_id_col == "(orden por posicion)":
                    paired_id_col = None

        suggestions = suggest_test('Continua', 'Categorica', n_groups, paired, all_normal, equal_var)

        if suggestions:
            recommended_name, recommended_id = suggestions[0]

            # P4: Explicar por que se recomienda este test
            if all_normal:
                _reason = "Tus datos siguen una distribucion normal (Shapiro-Wilk p > 0.05)"
                if paired:
                    _reason += " y son mediciones repetidas del mismo sujeto"
                elif n_groups == 2:
                    if equal_var:
                        _reason += ", varianzas homogeneas (Levene p > 0.05)"
                    else:
                        _reason += ", varianzas NO homogeneas (Levene p < 0.05) → se recomienda Welch"
                else:
                    _reason += f" y tienes {n_groups} grupos independientes"
                _reason += " → test parametrico."
            else:
                _reason = "Tus datos NO siguen una distribucion normal (Shapiro-Wilk p < 0.05)"
                _reason += " → test no parametrico (no asume normalidad)."

            st.success(f"**Recomendado: {recommended_name}**  \n{_reason}")

            # P4: Alternativas en expander
            all_tests = {}
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

            # F6: Selector de metodo post-hoc para >2 grupos
            _posthoc_method = None
            if n_groups > 2:
                _ph_key = 'anova' if selected_test_id == 'anova' else 'kruskal'
                _ph_options = POSTHOC_METHODS[_ph_key]
                _ph_default = 'tukey' if _ph_key == 'anova' else 'dunn_bonferroni'
                _ph_labels = {v[0]: k for k, v in _ph_options.items()}
                with st.expander("Metodo post-hoc (comparaciones multiples)"):
                    for method_id, (name, desc) in _ph_options.items():
                        st.markdown(f"- **{name}**: {desc}")
                    _ph_choice = st.selectbox(
                        "Metodo post-hoc",
                        options=[v[0] for v in _ph_options.values()],
                        index=list(_ph_options.keys()).index(_ph_default),
                        help="Se aplica solo si el test principal es significativo y hay >2 grupos."
                    )
                    _posthoc_method = _ph_labels[_ph_choice]
        else:
            st.error("No se encontro un test adecuado para esta configuracion.")
            st.stop()

    elif analysis_type == Q_CORRELATION:
        with col_left:
            if len(continuous_vars) < 2:
                st.warning("Necesitas al menos 2 variables continuas")
                st.stop()
            var_dep = st.selectbox("Variable a explicar", continuous_vars,
                                   help="La variable que quieres predecir o cuya variacion quieres entender.")
        with col_right:
            remaining = [c for c in continuous_vars if c != var_dep]
            var_group = st.selectbox("Variable explicativa", remaining,
                                     help="La variable que crees que puede influir o estar relacionada.")

        # A3: Validar que ambas variables sean numericas
        for _v in [var_dep, var_group]:
            ok, err = validate_continuous(df, _v)
            if not ok:
                st.error(err)
                st.stop()

        selected_groups = None
        all_tests = {
            "Correlacion de Pearson": "pearson",
            "Correlacion de Spearman": "spearman",
            "Regresion lineal": "linear_reg",
        }
        selected_test_name = st.selectbox("Elige el test", list(all_tests.keys()))
        selected_test_id = all_tests[selected_test_name]

    elif analysis_type == Q_ASSOCIATION:
        with col_left:
            if len(categorical_vars) < 2:
                st.warning("Necesitas al menos 2 variables categoricas")
                st.stop()
            var_dep = st.selectbox("Variable 1", categorical_vars)
        with col_right:
            remaining = [c for c in categorical_vars if c != var_dep]
            var_group = st.selectbox("Variable 2", remaining)

        selected_groups = None
        all_tests = {
            "Chi-cuadrado": "chi2",
            "Test exacto de Fisher": "fisher",
        }
        selected_test_name = st.selectbox("Elige el test", list(all_tests.keys()))
        selected_test_id = all_tests[selected_test_name]

    elif analysis_type == Q_AGREEMENT:
        st.info("Compara concordancia entre dos metodos de medicion (variables continuas).")
        with col_left:
            if len(continuous_vars) < 2:
                st.warning("Necesitas al menos 2 variables continuas")
                st.stop()
            var_dep = st.selectbox("Metodo 1", continuous_vars)
        with col_right:
            remaining = [c for c in continuous_vars if c != var_dep]
            var_group = st.selectbox("Metodo 2", remaining)

        for _v in [var_dep, var_group]:
            ok, err = validate_continuous(df, _v)
            if not ok:
                st.error(err)
                st.stop()

        selected_groups = None
        selected_test_id = 'bland_altman'

    elif analysis_type == Q_PREDICTION:
        st.info("Evalua capacidad discriminativa de un predictor continuo para un desenlace binario.")
        with col_left:
            if not categorical_vars:
                st.warning("Necesitas una variable categorica binaria (desenlace)")
                st.stop()
            var_dep = st.selectbox("Variable de desenlace (binaria)", categorical_vars)
        with col_right:
            if not continuous_vars:
                st.warning("Necesitas una variable continua (predictor)")
                st.stop()
            var_group = st.selectbox("Variable predictora (continua)", continuous_vars)

        # Validar que el desenlace sea binario
        n_cats = df[var_dep].dropna().nunique()
        if n_cats != 2:
            st.error(f"La variable '{var_dep}' debe tener exactamente 2 categorias, tiene {n_cats}.")
            st.stop()

        ok, err = validate_continuous(df, var_group)
        if not ok:
            st.error(err)
            st.stop()

        cat_labels = sorted(df[var_dep].dropna().unique())
        positive_label = st.selectbox("Etiqueta positiva", cat_labels,
                                      index=len(cat_labels) - 1)

        selected_groups = None
        selected_test_id = 'roc'

    elif analysis_type == Q_SURVIVAL:
        st.info("Analisis de supervivencia. Necesitas: tiempo hasta evento y variable de evento (0/1).")
        with col_left:
            if not continuous_vars:
                st.warning("Necesitas una variable continua (tiempo)")
                st.stop()
            var_dep = st.selectbox("Tiempo hasta evento", continuous_vars)
        with col_right:
            # B7: Filtrar a variables binarias (0/1)
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
            _km_group_options = ["(sin comparacion)"] + categorical_vars
            km_group_col = st.selectbox("Variable de grupo (opcional, para comparar curvas)",
                                        _km_group_options)
            if km_group_col == "(sin comparacion)":
                km_group_col = None
            else:
                all_km_groups = sorted(df[km_group_col].dropna().unique())
                km_groups = st.multiselect("Grupos a comparar", all_km_groups,
                                           default=all_km_groups[:min(4, len(all_km_groups))])
                if len(km_groups) < 1:
                    st.warning("Selecciona al menos 1 grupo.")
                    st.stop()

        selected_groups = None
        selected_test_id = 'kaplan_meier'

    elif analysis_type == Q_RISK:
        st.info("Regresion logistica: evalua si un predictor continuo aumenta el riesgo de un desenlace binario.")
        with col_left:
            if not categorical_vars:
                st.warning("Necesitas una variable categorica binaria (desenlace)")
                st.stop()
            var_dep = st.selectbox("Variable de desenlace (binaria)", categorical_vars,
                                   help="La variable que quieres predecir: enfermedad si/no, muerte si/no...")
        with col_right:
            if not continuous_vars:
                st.warning("Necesitas una variable continua (predictor)")
                st.stop()
            var_group = st.selectbox("Variable predictora (factor de riesgo)", continuous_vars,
                                     help="El factor que crees que puede aumentar el riesgo: edad, IMC, dosis...")

        n_cats = df[var_dep].dropna().nunique()
        if n_cats != 2:
            st.error(f"La variable '{var_dep}' debe tener exactamente 2 categorias, tiene {n_cats}.")
            st.stop()

        ok, err = validate_continuous(df, var_group)
        if not ok:
            st.error(err)
            st.stop()

        selected_groups = None
        selected_test_id = 'logistic'

    elif analysis_type == Q_RELIABILITY:
        st.info("ICC: evalua reproducibilidad entre evaluadores o metodos. "
                "Necesitas una variable de medicion y una variable que identifique al evaluador/metodo.")
        with col_left:
            if not continuous_vars:
                st.warning("Necesitas una variable continua (medicion)")
                st.stop()
            var_dep = st.selectbox("Variable de medicion", continuous_vars,
                                   help="Lo que se midio: score, longitud, concentracion...")
        with col_right:
            if not categorical_vars:
                st.warning("Necesitas una variable categorica (evaluador/metodo)")
                st.stop()
            var_group = st.selectbox("Evaluador / metodo", categorical_vars,
                                     help="Quien o que hizo la medicion: observador A/B, metodo 1/2...")

        ok, err = validate_continuous(df, var_dep)
        if not ok:
            st.error(err)
            st.stop()

        n_raters = df[var_group].dropna().nunique()
        if n_raters < 2:
            st.error(f"Se necesitan al menos 2 evaluadores/metodos. '{var_group}' tiene {n_raters}.")
            st.stop()

        selected_groups = None
        selected_test_id = 'icc'

    # Alpha
    alpha = st.slider("Nivel de significancia (alpha)", 0.001, 0.10, 0.05, 0.005,
                      help="Usualmente 0.05. Valores menores (0.01) son mas exigentes. Solo cambia esto si sabes por que.")

    # --- Ejecutar test --------------------------------------------------------
    if st.button("Ejecutar analisis", type="primary", use_container_width=True):
        with st.spinner("Calculando..."):
            _extra = {}
            if analysis_type == Q_DIFF_GROUPS and n_groups and n_groups > 2:
                _extra['posthoc_method'] = _posthoc_method
            if analysis_type == Q_PREDICTION:
                _extra['positive_label'] = positive_label
            elif analysis_type == Q_SURVIVAL:
                _extra['group_col'] = km_group_col
                _extra['groups'] = km_groups
            _extra = _extra or None
            result = run_test(selected_test_id, df, var_dep, var_group,
                              selected_groups, alpha, paired_id_col=paired_id_col,
                              extra=_extra)
            st.session_state.results.append(result)

        if result.get('success'):
            st.subheader("Resultados")

            # P6: Tabla descriptiva visible
            if result.get('groups') and isinstance(result.get('n'), list):
                desc_data = {'Grupo': result['groups'], 'n': result['n']}
                if result.get('mean') and isinstance(result['mean'], list):
                    desc_data['Media'] = [f"{m:.2f}" for m in result['mean']]
                    desc_data['DE'] = [f"{s:.2f}" for s in result['std']]
                if result.get('median') and isinstance(result['median'], list):
                    desc_data['Mediana'] = [f"{m:.2f}" for m in result['median']]
                st.dataframe(pd.DataFrame(desc_data).set_index('Grupo'),
                             use_container_width=True)

            rcols = st.columns(3)
            with rcols[0]:
                stat_val = result.get('statistic')
                if isinstance(stat_val, (int, float)):
                    st.metric("Estadistico", f"{stat_val:.4f}")
                else:
                    st.metric("Estadistico", "N/A")
            with rcols[1]:
                p = result.get('p_value')
                if isinstance(p, (int, float)):
                    st.metric("p-valor", f"{p:.6f}")
                else:
                    st.metric("p-valor", "N/A")
            with rcols[2]:
                sig = result.get('significant')
                if sig is True:
                    st.markdown("### Significativo")
                elif sig is False:
                    st.markdown("### No significativo")
                else:
                    st.markdown("### —")

            if result.get('warning'):
                st.warning(result['warning'])
            if result.get('posthoc_error'):
                st.warning(result['posthoc_error'])

            # S2: Intervalo de confianza
            if result.get('ci_lower') is not None and result.get('ci_upper') is not None:
                st.info(f"IC 95%: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")

            if result.get('cohens_d'):
                st.info(f"Tamano del efecto: d de Cohen = {result['cohens_d']:.3f}")
            if result.get('eta_squared'):
                st.info(f"Tamano del efecto: eta2 = {result['eta_squared']:.3f}")
            if result.get('r_squared') is not None:
                st.info(f"R2 = {result['r_squared']:.3f}")
            if result.get('auc') is not None:
                st.info(f"AUC = {result['auc']:.3f}")
                if result.get('best_threshold') is not None:
                    st.info(f"Corte optimo: {result['best_threshold']:.3f} "
                            f"(Sens={result['sensitivity']:.3f}, "
                            f"Esp={result['specificity']:.3f})")
            if result.get('bias') is not None:
                st.info(f"Sesgo = {result['bias']:.3f}, "
                        f"Limites de acuerdo: [{result['loa_lower']:.3f}, {result['loa_upper']:.3f}]")
            if result.get('curves'):
                with st.expander("Curvas de supervivencia", expanded=True):
                    for label, data in result['curves'].items():
                        med = data.get('median')
                        med_str = f"{med:.1f}" if med is not None else "no alcanzada"
                        st.markdown(f"**{label}**: n={data['n']}, mediana={med_str}")

            # F7: Regresion logistica
            if result.get('odds_ratio') is not None:
                or_val = result['odds_ratio']
                st.info(f"Odds Ratio = {or_val:.3f} "
                        f"(IC 95%: [{result['or_ci_lower']:.3f}, {result['or_ci_upper']:.3f}])")
                if result.get('pseudo_r2') is not None:
                    st.info(f"Pseudo R2 (McFadden) = {result['pseudo_r2']:.3f}")

            # F9: ICC
            if result.get('icc') is not None:
                st.info(f"ICC = {result['icc']:.3f} ({result.get('quality', '')}), "
                        f"{result.get('n_raters', 0)} evaluadores, "
                        f"{result.get('n_subjects', 0)} sujetos")

            # F8: Potencia post-hoc
            if result.get('power'):
                pw = result['power']
                power_pct = pw['power'] * 100
                _pw_color = "success" if pw['power'] >= 0.8 else "warning" if pw['power'] >= 0.5 else "error"
                if _pw_color == "success":
                    st.success(f"Potencia estadistica: {power_pct:.0f}% (adecuada). "
                               f"n por grupo para 80%: {pw.get('n_for_80', '—')}")
                elif _pw_color == "warning":
                    st.warning(f"Potencia estadistica: {power_pct:.0f}% (baja). "
                               f"n por grupo para 80%: {pw.get('n_for_80', '—')}")
                else:
                    st.error(f"Potencia estadistica: {power_pct:.0f}% (insuficiente). "
                             f"n por grupo para 80%: {pw.get('n_for_80', '—')}")

            # P1: Interpretacion en lenguaje natural
            interpretation = generate_interpretation(result)
            if interpretation:
                st.markdown("**Para tu publicacion:**")
                st.code(interpretation, language=None)

            # P5: Figura automatica
            _auto_fig_map = {
                Q_DIFF_GROUPS: "boxplot",
                Q_CORRELATION: "scatter",
                Q_ASSOCIATION: None,
                Q_AGREEMENT: "bland_altman",
                Q_PREDICTION: "roc",
                Q_SURVIVAL: "kaplan_meier",
                Q_RISK: None,
                Q_RELIABILITY: None,
            }
            _auto_fig_type = _auto_fig_map.get(analysis_type)
            if _auto_fig_type:
                _auto_groups = selected_groups if analysis_type == Q_DIFF_GROUPS else None
                auto_fig = generate_figure(
                    _auto_fig_type, df, var_dep, var_group,
                    _auto_groups, result, options=fig_options)
                st.pyplot(auto_fig)
                st.session_state.figures.append(auto_fig)

            with st.expander("Detalle completo"):
                st.text(format_result_text(result))

            # U2: Tabla de contingencia visual
            if result.get('contingency_table'):
                with st.expander("Tabla de contingencia", expanded=True):
                    ct_df = pd.DataFrame(result['contingency_table'])
                    st.dataframe(ct_df, use_container_width=True)

            if result.get('posthoc'):
                with st.expander(f"Post-hoc: {result['posthoc_name']}", expanded=True):
                    ph_df = pd.DataFrame(result['posthoc'])
                    st.dataframe(ph_df.round(4), use_container_width=True)
        else:
            st.error(f"Error en el analisis: {result.get('error', 'Desconocido')}")

    # --- PASO 4: Figuras ------------------------------------------------------
    st.header("4. Figuras")

    available_figs = {}
    if analysis_type == Q_DIFF_GROUPS:
        available_figs = {
            "Box plot + puntos": "boxplot",
            "Violin plot": "violin",
            "Barras + error (SEM)": "bar_error",
            "Datos pareados": "paired",
            "Histograma por grupo": "histogram",
        }
    elif analysis_type == Q_CORRELATION:
        available_figs = {
            "Scatter + regresion": "scatter",
            "Histograma": "histogram",
        }
    elif analysis_type == Q_AGREEMENT:
        available_figs = {
            "Bland-Altman": "bland_altman",
        }
    elif analysis_type == Q_PREDICTION:
        available_figs = {
            "Curva ROC": "roc",
        }
    elif analysis_type == Q_SURVIVAL:
        available_figs = {
            "Kaplan-Meier": "kaplan_meier",
        }
    else:
        available_figs = {
            "Histograma": "histogram",
        }

    selected_figs = st.multiselect("Selecciona las figuras a generar",
                                   list(available_figs.keys()))

    if selected_figs and st.button("Generar figuras", use_container_width=True):
        _close_figures()
        st.session_state.figures = []
        last_result = st.session_state.results[-1] if st.session_state.results else None

        for fig_name in selected_figs:
            fig_type = available_figs[fig_name]
            fig = generate_figure(
                fig_type, df, var_dep, var_group,
                selected_groups if analysis_type == Q_DIFF_GROUPS else None,
                last_result,
                options=fig_options,
            )
            st.session_state.figures.append(fig)
            st.pyplot(fig)

            # G1: Descarga en PNG y SVG
            dl_cols = st.columns(2)
            buf_png = BytesIO()
            fig.savefig(buf_png, format='png', dpi=300, bbox_inches='tight')
            buf_png.seek(0)
            with dl_cols[0]:
                st.download_button(
                    f"PNG 300dpi — {fig_name}",
                    data=buf_png,
                    file_name=f"statlab_{fig_type}.png",
                    mime="image/png",
                    use_container_width=True,
                )
            buf_svg = BytesIO()
            fig.savefig(buf_svg, format='svg', bbox_inches='tight')
            buf_svg.seek(0)
            with dl_cols[1]:
                st.download_button(
                    f"SVG — {fig_name}",
                    data=buf_svg,
                    file_name=f"statlab_{fig_type}.svg",
                    mime="image/svg+xml",
                    use_container_width=True,
                )

    # --- S4: Tabla resumen de todos los analisis --------------------------------
    valid_results = [r for r in st.session_state.results if r.get('success')]

    if len(valid_results) > 1:
        st.header("5. Resumen de analisis")
        _summary_rows = []
        for r in valid_results:
            row = {
                'Test': r.get('test_name', 'N/A'),
                'Variable': r.get('var_dep', ''),
                'Grupo/Predictor': r.get('var_group', ''),
            }
            p = r.get('p_value')
            row['p-valor'] = f"{p:.4f}" if isinstance(p, (int, float)) else "—"
            sig = r.get('significant')
            row['Sig.'] = "Si" if sig is True else "No" if sig is False else "—"

            if r.get('ci_lower') is not None:
                row['IC 95%'] = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
            else:
                row['IC 95%'] = "—"

            if r.get('cohens_d') is not None:
                row['Efecto'] = f"d={r['cohens_d']:.2f}"
            elif r.get('eta_squared') is not None:
                row['Efecto'] = f"eta2={r['eta_squared']:.3f}"
            elif r.get('r_squared') is not None:
                row['Efecto'] = f"R2={r['r_squared']:.3f}"
            elif r.get('auc') is not None:
                row['Efecto'] = f"AUC={r['auc']:.3f}"
            else:
                row['Efecto'] = "—"

            _summary_rows.append(row)

        st.dataframe(pd.DataFrame(_summary_rows), use_container_width=True, hide_index=True)

    # --- PASO 6: Informe ------------------------------------------------------
    st.header("6. Descargar informe" if len(valid_results) > 1 else "5. Descargar informe")

    if valid_results:
        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            full_text = "STATLAB - INFORME DE ANALISIS ESTADISTICO\n"
            full_text += f"{'='*60}\n\n"
            for r in valid_results:
                full_text += format_result_text(r) + "\n"

            st.download_button(
                "Informe TXT",
                data=full_text,
                file_name="statlab_informe.txt",
                mime="text/plain",
                use_container_width=True
            )

        with col_dl2:
            if st.button("Generar informe PDF", use_container_width=True):
                progress = st.progress(0, text="Preparando informe...")
                progress.progress(10, text="Generando resultados...")
                pdf_buf = generate_pdf_report(valid_results, st.session_state.figures)
                progress.progress(90, text="Finalizando...")
                st.download_button(
                    "Descargar PDF",
                    data=pdf_buf,
                    file_name="statlab_informe.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                progress.progress(100, text="Listo")
    else:
        st.info("Ejecuta al menos un analisis para poder generar el informe.")

    # --- Historial ------------------------------------------------------------
    if st.session_state.results:
        with st.expander(f"Historial de analisis ({len(valid_results)} realizados)"):
            for i, r in enumerate(valid_results):
                p_val = r.get('p_value')
                if isinstance(p_val, float):
                    st.markdown(f"**{i+1}. {r.get('test_name', 'N/A')}** — "
                                f"{r['var_dep']} x {r['var_group']} — "
                                f"p = {p_val:.4f}")
                else:
                    st.markdown(f"**{i+1}.** Error")
            # U1: Boton para limpiar historial
            if st.button("Limpiar historial", type="secondary"):
                _close_figures()
                st.session_state.results = []
                st.session_state.figures = []
                st.rerun()

else:
    st.markdown("---")
    st.markdown("""
    ### Como funciona?

    **1.** Sube un archivo CSV o Excel con tus datos
    **2.** Define que variables son continuas y cuales categoricas
    **3.** Elige que quieres analizar y StatLab te sugerira el test adecuado
    **4.** Genera figuras de calidad publicacion
    **5.** Descarga el informe con todos los resultados

    ### Tests disponibles

    **Comparacion de 2 grupos:** T-test, Welch, Mann-Whitney, Wilcoxon
    **Comparacion de >2 grupos:** ANOVA + Tukey, Kruskal-Wallis + Dunn
    **Correlacion:** Pearson, Spearman, regresion lineal
    **Categoricas:** Chi-cuadrado, Fisher
    **Concordancia:** Bland-Altman
    **Discriminacion:** Curva ROC (AUC, corte optimo)
    **Supervivencia:** Kaplan-Meier, log-rank
    """)
