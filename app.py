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
from stats.tests import check_normality, suggest_test, run_test
from charts.figures import generate_figure
from reports.text import format_result_text
from reports.pdf import generate_pdf_report

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
    st.info("StatLab infiere automaticamente el tipo, pero puedes corregirlo.")

    cols = df.columns.tolist()
    var_types = {}
    col_grid = st.columns(3)
    for i, col_name in enumerate(cols):
        inferred = infer_variable_type(df[col_name])
        with col_grid[i % 3]:
            var_types[col_name] = st.selectbox(
                f"`{col_name}`",
                ['Continua', 'Categorica'],
                index=0 if inferred == 'Continua' else 1,
                key=f"vtype_{col_name}"
            )
    st.session_state.var_types = var_types

    # --- PASO 3: Configuracion del analisis -----------------------------------
    st.header("3. Analisis estadistico")

    continuous_vars = [c for c, t in var_types.items() if t == 'Continua']
    categorical_vars = [c for c, t in var_types.items() if t == 'Categorica']

    analysis_type = st.radio(
        "Tipo de analisis",
        ["Comparacion de grupos", "Correlacion / Regresion", "Tabla de contingencia"],
        horizontal=True
    )

    col_left, col_right = st.columns(2)

    if analysis_type == "Comparacion de grupos":
        with col_left:
            if not continuous_vars:
                st.warning("No hay variables continuas definidas")
                st.stop()
            var_dep = st.selectbox("Variable dependiente (continua)", continuous_vars)
        with col_right:
            if not categorical_vars:
                st.warning("No hay variables categoricas definidas")
                st.stop()
            var_group = st.selectbox("Variable de agrupacion (categorica)", categorical_vars)

        # A3: Validar que la variable continua sea numerica
        ok, err = validate_continuous(df, var_dep)
        if not ok:
            st.error(err)
            st.stop()

        all_groups = sorted(df[var_group].dropna().unique())
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

        # Normalidad
        st.subheader("Comprobacion de normalidad")
        norm_cols = st.columns(min(n_groups, 4))
        all_normal = True
        for i, g in enumerate(selected_groups):
            gdata = df[df[var_group] == g][var_dep].dropna()
            stat, p = check_normality(gdata)
            is_normal = p > 0.05 if p is not None else None
            if is_normal is False:
                all_normal = False
            with norm_cols[i % min(n_groups, 4)]:
                if p is not None:
                    emoji = "✅" if is_normal else "⚠️"
                    st.metric(f"{g}", f"p = {p:.4f} {emoji}")
                else:
                    st.metric(f"{g}", "N/A")

        paired = st.checkbox("Datos pareados?", value=False)

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

        suggestions = suggest_test('Continua', 'Categorica', n_groups, paired, all_normal)

        if suggestions:
            st.subheader("Test sugerido")
            if all_normal:
                st.success("Datos normales -> se sugiere test parametrico")
            else:
                st.warning("Datos no normales -> se sugiere test no parametrico")

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

            selected_test_name = st.selectbox(
                "Elige el test",
                options=list(all_tests.keys()),
                index=list(all_tests.keys()).index(suggestions[0][0]) if suggestions[0][0] in all_tests else 0
            )
            selected_test_id = all_tests[selected_test_name]
        else:
            st.error("No se encontro un test adecuado para esta configuracion.")
            st.stop()

    elif analysis_type == "Correlacion / Regresion":
        with col_left:
            if len(continuous_vars) < 2:
                st.warning("Necesitas al menos 2 variables continuas")
                st.stop()
            var_dep = st.selectbox("Variable Y", continuous_vars)
        with col_right:
            remaining = [c for c in continuous_vars if c != var_dep]
            var_group = st.selectbox("Variable X", remaining)

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

    elif analysis_type == "Tabla de contingencia":
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

    # Alpha
    alpha = st.slider("Nivel de significancia (alpha)", 0.001, 0.10, 0.05, 0.005)

    # --- Ejecutar test --------------------------------------------------------
    if st.button("Ejecutar analisis", type="primary", use_container_width=True):
        with st.spinner("Calculando..."):
            _id_col = paired_id_col if (analysis_type == "Comparacion de grupos"
                                        and paired and 'paired_id_col' in dir()) else None
            result = run_test(selected_test_id, df, var_dep, var_group,
                              selected_groups, alpha, paired_id_col=_id_col)
            st.session_state.results.append(result)

        if result.get('success'):
            st.subheader("Resultados")

            rcols = st.columns(3)
            with rcols[0]:
                st.metric("Estadistico", f"{result.get('statistic', 0):.4f}")
            with rcols[1]:
                p = result.get('p_value', 1)
                st.metric("p-valor", f"{p:.6f}")
            with rcols[2]:
                if result['significant']:
                    st.markdown("### Significativo")
                else:
                    st.markdown("### No significativo")

            if result.get('warning'):
                st.warning(result['warning'])
            if result.get('posthoc_error'):
                st.warning(result['posthoc_error'])

            if result.get('cohens_d'):
                st.info(f"Tamano del efecto: d de Cohen = {result['cohens_d']:.3f}")
            if result.get('eta_squared'):
                st.info(f"Tamano del efecto: eta2 = {result['eta_squared']:.3f}")
            if result.get('r_squared') is not None:
                st.info(f"R2 = {result['r_squared']:.3f}")

            with st.expander("Detalle completo", expanded=True):
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
    if analysis_type == "Comparacion de grupos":
        available_figs = {
            "Box plot + puntos": "boxplot",
            "Violin plot": "violin",
            "Barras + error (SEM)": "bar_error",
            "Datos pareados": "paired",
            "Histograma por grupo": "histogram",
        }
    elif analysis_type == "Correlacion / Regresion":
        available_figs = {
            "Scatter + regresion": "scatter",
            "Histograma": "histogram",
        }
    else:
        available_figs = {
            "Histograma": "histogram",
        }

    selected_figs = st.multiselect("Selecciona las figuras a generar",
                                   list(available_figs.keys()))

    if selected_figs and st.button("Generar figuras", use_container_width=True):
        st.session_state.figures = []
        last_result = st.session_state.results[-1] if st.session_state.results else None

        for fig_name in selected_figs:
            fig_type = available_figs[fig_name]
            fig = generate_figure(
                fig_type, df, var_dep, var_group,
                selected_groups if analysis_type == "Comparacion de grupos" else None,
                last_result
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

    # --- PASO 5: Informe ------------------------------------------------------
    st.header("5. Descargar informe")

    valid_results = [r for r in st.session_state.results if r.get('success')]

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
    """)
