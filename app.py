"""
StatLab - Análisis Estadístico y Generación de Figuras
Prototipo MVP - Guzmán
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# ─── Configuración general ───────────────────────────────────────────────────
st.set_page_config(
    page_title="StatLab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo
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

# ─── Estado de la sesión ─────────────────────────────────────────────────────
if 'df' not in st.session_state:
    st.session_state.df = None
if 'var_types' not in st.session_state:
    st.session_state.var_types = {}
if 'results' not in st.session_state:
    st.session_state.results = []
if 'figures' not in st.session_state:
    st.session_state.figures = []

# ─── Funciones auxiliares ────────────────────────────────────────────────────

def infer_variable_type(series):
    """Infiere si una variable es categórica o continua."""
    if series.dtype == 'object' or series.dtype.name == 'category':
        return 'Categórica'
    if series.nunique() <= 10 and series.dtype in ['int64', 'int32']:
        return 'Categórica'
    return 'Continua'


def check_normality(data, alpha=0.05):
    """Test de Shapiro-Wilk para normalidad."""
    if len(data) < 3 or len(data) > 5000:
        return None, None
    clean = data.dropna()
    if len(clean) < 3:
        return None, None
    stat, p = stats.shapiro(clean)
    return stat, p


def suggest_test(var_dep_type, var_group_type, n_groups, paired=False, normal=True):
    """Sugiere el test estadístico más adecuado."""
    suggestions = []
    if var_dep_type == 'Continua' and var_group_type == 'Categórica':
        if n_groups == 2:
            if normal:
                if paired:
                    suggestions.append(("T-test pareado", "t_paired"))
                else:
                    suggestions.append(("T-test independiente", "t_independent"))
                    suggestions.append(("T-test Welch", "t_welch"))
            else:
                if paired:
                    suggestions.append(("Wilcoxon signed-rank", "wilcoxon"))
                else:
                    suggestions.append(("Mann-Whitney U", "mann_whitney"))
        elif n_groups > 2:
            if normal:
                if paired:
                    suggestions.append(("ANOVA medidas repetidas", "rm_anova"))
                else:
                    suggestions.append(("ANOVA one-way", "anova"))
            else:
                if paired:
                    suggestions.append(("Friedman", "friedman"))
                else:
                    suggestions.append(("Kruskal-Wallis", "kruskal"))
    elif var_dep_type == 'Categórica' and var_group_type == 'Categórica':
        suggestions.append(("Chi-cuadrado", "chi2"))
        suggestions.append(("Test exacto de Fisher", "fisher"))
    elif var_dep_type == 'Continua' and var_group_type == 'Continua':
        if normal:
            suggestions.append(("Correlación de Pearson", "pearson"))
        else:
            suggestions.append(("Correlación de Spearman", "spearman"))
        suggestions.append(("Regresión lineal", "linear_reg"))
    return suggestions


def run_test(test_id, df, var_dep, var_group, groups=None, alpha=0.05):
    """Ejecuta el test estadístico seleccionado."""
    result = {"test": test_id, "var_dep": var_dep, "var_group": var_group, "alpha": alpha}
    
    try:
        if test_id in ['t_independent', 't_welch', 'mann_whitney', 'wilcoxon', 't_paired']:
            if groups and len(groups) == 2:
                g1 = df[df[var_group] == groups[0]][var_dep].dropna()
                g2 = df[df[var_group] == groups[1]][var_dep].dropna()
            else:
                unique_groups = df[var_group].dropna().unique()[:2]
                g1 = df[df[var_group] == unique_groups[0]][var_dep].dropna()
                g2 = df[df[var_group] == unique_groups[1]][var_dep].dropna()
                groups = list(unique_groups[:2])
            
            result["groups"] = [str(g) for g in groups]
            result["n"] = [len(g1), len(g2)]
            result["mean"] = [float(g1.mean()), float(g2.mean())]
            result["std"] = [float(g1.std()), float(g2.std())]
            result["median"] = [float(g1.median()), float(g2.median())]
            
            if test_id == 't_independent':
                stat, p = stats.ttest_ind(g1, g2, equal_var=True)
                result["test_name"] = "T-test independiente (Student)"
            elif test_id == 't_welch':
                stat, p = stats.ttest_ind(g1, g2, equal_var=False)
                result["test_name"] = "T-test de Welch"
            elif test_id == 't_paired':
                min_len = min(len(g1), len(g2))
                stat, p = stats.ttest_rel(g1.values[:min_len], g2.values[:min_len])
                result["test_name"] = "T-test pareado"
            elif test_id == 'mann_whitney':
                stat, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                result["test_name"] = "Mann-Whitney U"
            elif test_id == 'wilcoxon':
                min_len = min(len(g1), len(g2))
                stat, p = stats.wilcoxon(g1.values[:min_len], g2.values[:min_len])
                result["test_name"] = "Wilcoxon signed-rank"
            
            result["statistic"] = float(stat)
            result["p_value"] = float(p)
            result["significant"] = p < alpha
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((g1.std()**2 + g2.std()**2) / 2)
            if pooled_std > 0:
                result["cohens_d"] = float(abs(g1.mean() - g2.mean()) / pooled_std)
            
        elif test_id in ['anova', 'kruskal']:
            group_data = []
            group_names = groups if groups else sorted(df[var_group].dropna().unique())
            for g in group_names:
                gd = df[df[var_group] == g][var_dep].dropna()
                if len(gd) > 0:
                    group_data.append(gd)
            
            result["groups"] = [str(g) for g in group_names]
            result["n"] = [len(gd) for gd in group_data]
            result["mean"] = [float(gd.mean()) for gd in group_data]
            result["std"] = [float(gd.std()) for gd in group_data]
            
            if test_id == 'anova':
                stat, p = stats.f_oneway(*group_data)
                result["test_name"] = "ANOVA one-way"
                # Post-hoc Tukey
                if p < alpha and len(group_data) > 2:
                    all_data = pd.concat([pd.DataFrame({var_dep: gd, var_group: name}) 
                                         for gd, name in zip(group_data, group_names)])
                    try:
                        posthoc = sp.posthoc_tukey(all_data, val_col=var_dep, group_col=var_group)
                        result["posthoc"] = posthoc.to_dict()
                        result["posthoc_name"] = "Tukey HSD"
                    except:
                        pass
            else:
                stat, p = stats.kruskal(*group_data)
                result["test_name"] = "Kruskal-Wallis"
                # Post-hoc Dunn
                if p < alpha and len(group_data) > 2:
                    all_data = pd.concat([pd.DataFrame({var_dep: gd, var_group: name}) 
                                         for gd, name in zip(group_data, group_names)])
                    try:
                        posthoc = sp.posthoc_dunn(all_data, val_col=var_dep, group_col=var_group, 
                                                   p_adjust='bonferroni')
                        result["posthoc"] = posthoc.to_dict()
                        result["posthoc_name"] = "Dunn (Bonferroni)"
                    except:
                        pass
            
            result["statistic"] = float(stat)
            result["p_value"] = float(p)
            result["significant"] = p < alpha
            
            # Effect size (eta squared for ANOVA)
            if test_id == 'anova':
                all_vals = pd.concat(group_data)
                ss_total = ((all_vals - all_vals.mean())**2).sum()
                ss_between = sum(len(gd) * (gd.mean() - all_vals.mean())**2 for gd in group_data)
                if ss_total > 0:
                    result["eta_squared"] = float(ss_between / ss_total)
        
        elif test_id in ['pearson', 'spearman']:
            clean = df[[var_dep, var_group]].dropna()
            x, y = clean[var_dep], clean[var_group]
            
            if test_id == 'pearson':
                stat, p = stats.pearsonr(x, y)
                result["test_name"] = "Correlación de Pearson"
            else:
                stat, p = stats.spearmanr(x, y)
                result["test_name"] = "Correlación de Spearman"
            
            result["statistic"] = float(stat)
            result["p_value"] = float(p)
            result["significant"] = p < alpha
            result["r_squared"] = float(stat**2)
            result["n"] = len(clean)
        
        elif test_id == 'linear_reg':
            clean = df[[var_dep, var_group]].dropna()
            slope, intercept, r, p, se = stats.linregress(clean[var_group], clean[var_dep])
            result["test_name"] = "Regresión lineal"
            result["slope"] = float(slope)
            result["intercept"] = float(intercept)
            result["statistic"] = float(r)
            result["r_squared"] = float(r**2)
            result["p_value"] = float(p)
            result["std_error"] = float(se)
            result["significant"] = p < alpha
            result["n"] = len(clean)
        
        elif test_id in ['chi2', 'fisher']:
            ct = pd.crosstab(df[var_dep], df[var_group])
            result["contingency_table"] = ct.to_dict()
            
            if test_id == 'chi2':
                stat, p, dof, expected = stats.chi2_contingency(ct)
                result["test_name"] = "Chi-cuadrado"
                result["dof"] = int(dof)
            else:
                if ct.shape == (2, 2):
                    stat, p = stats.fisher_exact(ct)
                    result["test_name"] = "Test exacto de Fisher"
                else:
                    stat, p, dof, expected = stats.chi2_contingency(ct)
                    result["test_name"] = "Chi-cuadrado (Fisher no aplicable >2x2)"
                    result["dof"] = int(dof)
            
            result["statistic"] = float(stat)
            result["p_value"] = float(p)
            result["significant"] = p < alpha
        
        result["success"] = True
        
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
    
    return result


def generate_figure(fig_type, df, var_dep, var_group, groups=None, result=None):
    """Genera una figura según el tipo seleccionado."""
    # Paleta médica profesional
    palette = ['#3182ce', '#e53e3e', '#38a169', '#d69e2e', '#805ad5', '#dd6b20']
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if groups:
        plot_df = df[df[var_group].isin(groups)].copy()
    else:
        plot_df = df.copy()
    
    try:
        if fig_type == 'boxplot':
            sns.boxplot(data=plot_df, x=var_group, y=var_dep, palette=palette, 
                       width=0.6, ax=ax, showfliers=True)
            # Overlay individual points
            sns.stripplot(data=plot_df, x=var_group, y=var_dep, color='black', 
                         alpha=0.4, size=4, jitter=True, ax=ax)
            ax.set_title(f'{var_dep} por {var_group}', fontweight='bold', pad=15)
            
        elif fig_type == 'violin':
            sns.violinplot(data=plot_df, x=var_group, y=var_dep, palette=palette, 
                          inner='box', ax=ax)
            sns.stripplot(data=plot_df, x=var_group, y=var_dep, color='black', 
                         alpha=0.3, size=3, jitter=True, ax=ax)
            ax.set_title(f'{var_dep} por {var_group}', fontweight='bold', pad=15)
            
        elif fig_type == 'bar_error':
            grouped = plot_df.groupby(var_group)[var_dep]
            means = grouped.mean()
            sems = grouped.sem()
            order = groups if groups else means.index
            x_pos = range(len(order))
            bars = ax.bar(x_pos, [means[g] for g in order], 
                         yerr=[sems[g] for g in order],
                         capsize=5, color=palette[:len(order)], 
                         edgecolor='black', linewidth=0.5, alpha=0.85, width=0.6)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(order)
            ax.set_ylabel(var_dep)
            ax.set_xlabel(var_group)
            ax.set_title(f'{var_dep} por {var_group} (media ± SEM)', fontweight='bold', pad=15)
            
        elif fig_type == 'scatter':
            ax.scatter(plot_df[var_group], plot_df[var_dep], alpha=0.6, 
                      color=palette[0], edgecolors='white', s=60)
            # Línea de regresión
            clean = plot_df[[var_dep, var_group]].dropna()
            if len(clean) > 2:
                z = np.polyfit(clean[var_group], clean[var_dep], 1)
                p_line = np.poly1d(z)
                x_line = np.linspace(clean[var_group].min(), clean[var_group].max(), 100)
                ax.plot(x_line, p_line(x_line), '--', color=palette[1], linewidth=2)
                r, pval = stats.pearsonr(clean[var_group], clean[var_dep])
                ax.text(0.05, 0.95, f'r = {r:.3f}, p = {pval:.4f}', 
                       transform=ax.transAxes, fontsize=11, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_xlabel(var_group)
            ax.set_ylabel(var_dep)
            ax.set_title(f'{var_dep} vs {var_group}', fontweight='bold', pad=15)
            
        elif fig_type == 'paired':
            unique_groups = groups if groups else sorted(plot_df[var_group].dropna().unique())[:2]
            if len(unique_groups) == 2:
                g1_data = plot_df[plot_df[var_group] == unique_groups[0]][var_dep].dropna().values
                g2_data = plot_df[plot_df[var_group] == unique_groups[1]][var_dep].dropna().values
                min_n = min(len(g1_data), len(g2_data))
                for i in range(min_n):
                    color = palette[1] if g2_data[i] > g1_data[i] else palette[0]
                    ax.plot([0, 1], [g1_data[i], g2_data[i]], '-o', color=color, 
                           alpha=0.4, markersize=6)
                ax.set_xticks([0, 1])
                ax.set_xticklabels(unique_groups)
                ax.set_ylabel(var_dep)
                ax.set_title(f'{var_dep}: {unique_groups[0]} → {unique_groups[1]}', 
                           fontweight='bold', pad=15)
        
        elif fig_type == 'histogram':
            if var_group and plot_df[var_group].dtype == 'object':
                unique_groups = groups if groups else sorted(plot_df[var_group].dropna().unique())
                for i, g in enumerate(unique_groups):
                    gd = plot_df[plot_df[var_group] == g][var_dep].dropna()
                    ax.hist(gd, bins='auto', alpha=0.6, color=palette[i % len(palette)], 
                           label=str(g), edgecolor='white')
                ax.legend()
            else:
                ax.hist(plot_df[var_dep].dropna(), bins='auto', alpha=0.7, 
                       color=palette[0], edgecolor='white')
            ax.set_xlabel(var_dep)
            ax.set_ylabel('Frecuencia')
            ax.set_title(f'Distribución de {var_dep}', fontweight='bold', pad=15)
        
        # Añadir significancia si hay resultado
        if result and result.get('p_value') is not None:
            p = result['p_value']
            if p < 0.001:
                sig_text = '***'
            elif p < 0.01:
                sig_text = '**'
            elif p < 0.05:
                sig_text = '*'
            else:
                sig_text = 'ns'
            ax.text(0.95, 0.95, f'p = {p:.4f} ({sig_text})', transform=ax.transAxes,
                   fontsize=11, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
    
    return fig


def format_result_text(result):
    """Formatea un resultado como texto para el informe."""
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"TEST: {result.get('test_name', result.get('test', 'N/A'))}")
    lines.append(f"{'='*60}")
    lines.append(f"Variable dependiente: {result.get('var_dep', 'N/A')}")
    lines.append(f"Variable de agrupación: {result.get('var_group', 'N/A')}")
    lines.append(f"Nivel de significancia: α = {result.get('alpha', 0.05)}")
    lines.append("")
    
    if result.get('groups'):
        lines.append("GRUPOS:")
        for i, g in enumerate(result['groups']):
            n = result['n'][i] if isinstance(result.get('n'), list) else result.get('n', '')
            line = f"  {g}: n = {n}"
            if result.get('mean') and isinstance(result['mean'], list):
                line += f", media = {result['mean'][i]:.4f}, DE = {result['std'][i]:.4f}"
            if result.get('median') and isinstance(result['median'], list):
                line += f", mediana = {result['median'][i]:.4f}"
            lines.append(line)
        lines.append("")
    
    if result.get('n') and not isinstance(result['n'], list):
        lines.append(f"N: {result['n']}")
    
    lines.append("RESULTADOS:")
    lines.append(f"  Estadístico = {result.get('statistic', 'N/A'):.4f}" if isinstance(result.get('statistic'), (int, float)) else f"  Estadístico = {result.get('statistic', 'N/A')}")
    
    if result.get('dof') is not None:
        lines.append(f"  Grados de libertad = {result['dof']}")
    
    p = result.get('p_value')
    if p is not None:
        lines.append(f"  p-valor = {p:.6f}")
        if p < 0.001:
            lines.append(f"  Significancia: *** (p < 0.001)")
        elif p < 0.01:
            lines.append(f"  Significancia: ** (p < 0.01)")
        elif p < 0.05:
            lines.append(f"  Significancia: * (p < 0.05)")
        else:
            lines.append(f"  Significancia: ns (p ≥ 0.05)")
    
    if result.get('cohens_d') is not None:
        d = result['cohens_d']
        effect = "pequeño" if d < 0.5 else "mediano" if d < 0.8 else "grande"
        lines.append(f"  d de Cohen = {d:.4f} (efecto {effect})")
    
    if result.get('eta_squared') is not None:
        eta = result['eta_squared']
        effect = "pequeño" if eta < 0.06 else "mediano" if eta < 0.14 else "grande"
        lines.append(f"  η² = {eta:.4f} (efecto {effect})")
    
    if result.get('r_squared') is not None:
        lines.append(f"  R² = {result['r_squared']:.4f}")
    
    if result.get('slope') is not None:
        lines.append(f"  Pendiente = {result['slope']:.4f} ± {result.get('std_error', 0):.4f}")
        lines.append(f"  Intercepto = {result['intercept']:.4f}")
    
    if result.get('posthoc'):
        lines.append(f"\n  POST-HOC ({result.get('posthoc_name', 'N/A')}):")
        ph = pd.DataFrame(result['posthoc'])
        lines.append(f"  {ph.to_string()}")
    
    lines.append("")
    sig = result.get('significant', None)
    if sig is not None:
        if sig:
            lines.append("CONCLUSIÓN: Se rechaza H₀. La diferencia/asociación ES estadísticamente significativa.")
        else:
            lines.append("CONCLUSIÓN: No se rechaza H₀. La diferencia/asociación NO es estadísticamente significativa.")
    
    lines.append("")
    return "\n".join(lines)


def generate_pdf_report(results, figures):
    """Genera un informe PDF con resultados y figuras."""
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        # Portada
        fig_cover = plt.figure(figsize=(8.5, 11))
        fig_cover.text(0.5, 0.65, 'StatLab', fontsize=36, ha='center', fontweight='bold', color='#1a365d')
        fig_cover.text(0.5, 0.58, 'Informe de Análisis Estadístico', fontsize=16, ha='center', color='#4a5568')
        fig_cover.text(0.5, 0.50, f'Generado con StatLab', fontsize=11, ha='center', color='#718096')
        import datetime
        fig_cover.text(0.5, 0.44, f'Fecha: {datetime.datetime.now().strftime("%d/%m/%Y %H:%M")}', 
                      fontsize=11, ha='center', color='#718096')
        fig_cover.text(0.5, 0.38, f'Número de análisis: {len(results)}', 
                      fontsize=11, ha='center', color='#718096')
        plt.axis('off')
        pdf.savefig(fig_cover)
        plt.close(fig_cover)
        
        # Resultados
        for i, result in enumerate(results):
            if not result.get('success'):
                continue
            text = format_result_text(result)
            fig_res = plt.figure(figsize=(8.5, 11))
            fig_res.text(0.07, 0.95, text, fontsize=9, fontfamily='monospace',
                        verticalalignment='top', wrap=True)
            plt.axis('off')
            pdf.savefig(fig_res)
            plt.close(fig_res)
        
        # Figuras
        for fig in figures:
            pdf.savefig(fig, bbox_inches='tight')
    
    buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════════════════════════════
# INTERFAZ PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

st.title("📊 StatLab")
st.caption("Análisis estadístico y generación de figuras")

# ─── PASO 1: Carga de datos ──────────────────────────────────────────────────
st.header("1. Datos")

uploaded_file = st.file_uploader("Sube tu archivo", type=['csv', 'xlsx', 'xls'], 
                                  help="Formatos aceptados: CSV, XLSX, XLS")

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            # Intentar detectar separador
            sample = uploaded_file.read(4096).decode('utf-8', errors='ignore')
            uploaded_file.seek(0)
            sep = ',' if sample.count(',') > sample.count(';') else ';'
            st.session_state.df = pd.read_csv(uploaded_file, sep=sep)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)
        st.success(f"✅ Cargado: {uploaded_file.name} — {st.session_state.df.shape[0]} filas × {st.session_state.df.shape[1]} columnas")
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")

df = st.session_state.df

if df is not None:
    # Preview
    with st.expander("👀 Vista previa de los datos", expanded=True):
        st.dataframe(df.head(20), use_container_width=True)
    
    # Resumen rápido
    with st.expander("📋 Resumen descriptivo"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Filas", df.shape[0])
        col2.metric("Columnas", df.shape[1])
        col3.metric("Valores faltantes", int(df.isna().sum().sum()))
        st.dataframe(df.describe(include='all').round(3), use_container_width=True)
    
    # ─── PASO 2: Definición de variables ──────────────────────────────────────
    st.header("2. Definición de variables")
    
    st.info("💡 StatLab infiere automáticamente el tipo, pero puedes corregirlo.")
    
    cols = df.columns.tolist()
    var_types = {}
    
    col_grid = st.columns(3)
    for i, col_name in enumerate(cols):
        inferred = infer_variable_type(df[col_name])
        with col_grid[i % 3]:
            var_types[col_name] = st.selectbox(
                f"`{col_name}`",
                ['Continua', 'Categórica'],
                index=0 if inferred == 'Continua' else 1,
                key=f"vtype_{col_name}"
            )
    
    st.session_state.var_types = var_types
    
    # ─── PASO 3: Configuración del análisis ───────────────────────────────────
    st.header("3. Análisis estadístico")
    
    continuous_vars = [c for c, t in var_types.items() if t == 'Continua']
    categorical_vars = [c for c, t in var_types.items() if t == 'Categórica']
    
    analysis_type = st.radio(
        "Tipo de análisis",
        ["Comparación de grupos", "Correlación / Regresión", "Tabla de contingencia"],
        horizontal=True
    )
    
    col_left, col_right = st.columns(2)
    
    if analysis_type == "Comparación de grupos":
        with col_left:
            if not continuous_vars:
                st.warning("No hay variables continuas definidas")
                st.stop()
            var_dep = st.selectbox("Variable dependiente (continua)", continuous_vars)
        with col_right:
            if not categorical_vars:
                st.warning("No hay variables categóricas definidas")
                st.stop()
            var_group = st.selectbox("Variable de agrupación (categórica)", categorical_vars)
        
        # Selección de grupos
        all_groups = sorted(df[var_group].dropna().unique())
        selected_groups = st.multiselect(
            f"Grupos a comparar (de `{var_group}`)",
            options=all_groups,
            default=all_groups[:min(4, len(all_groups))]
        )
        
        if len(selected_groups) < 2:
            st.warning("Selecciona al menos 2 grupos para comparar.")
            st.stop()
        
        n_groups = len(selected_groups)
        
        # Normalidad
        st.subheader("Comprobación de normalidad")
        norm_results = {}
        norm_cols = st.columns(min(n_groups, 4))
        all_normal = True
        for i, g in enumerate(selected_groups):
            gdata = df[df[var_group] == g][var_dep].dropna()
            stat, p = check_normality(gdata)
            is_normal = p > 0.05 if p is not None else None
            if is_normal is False:
                all_normal = False
            norm_results[g] = {"stat": stat, "p": p, "normal": is_normal}
            with norm_cols[i % min(n_groups, 4)]:
                if p is not None:
                    emoji = "✅" if is_normal else "⚠️"
                    st.metric(f"{g}", f"p = {p:.4f} {emoji}")
                else:
                    st.metric(f"{g}", "N/A")
        
        paired = st.checkbox("¿Datos pareados?", value=False)
        
        # Sugerencias
        suggestions = suggest_test('Continua', 'Categórica', n_groups, paired, all_normal)
        
        if suggestions:
            st.subheader("Test sugerido")
            if all_normal:
                st.success("✅ Datos normales → se sugiere test paramétrico")
            else:
                st.warning("⚠️ Datos no normales → se sugiere test no paramétrico")
            
            test_options = {name: tid for name, tid in suggestions}
            
            # Add all valid tests for user override
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
            st.error("No se encontró un test adecuado para esta configuración.")
            st.stop()
        
    elif analysis_type == "Correlación / Regresión":
        with col_left:
            if len(continuous_vars) < 2:
                st.warning("Necesitas al menos 2 variables continuas")
                st.stop()
            var_dep = st.selectbox("Variable Y", continuous_vars)
        with col_right:
            remaining = [c for c in continuous_vars if c != var_dep]
            var_group = st.selectbox("Variable X", remaining)
        
        selected_groups = None
        all_tests = {
            "Correlación de Pearson": "pearson",
            "Correlación de Spearman": "spearman",
            "Regresión lineal": "linear_reg",
        }
        selected_test_name = st.selectbox("Elige el test", list(all_tests.keys()))
        selected_test_id = all_tests[selected_test_name]
        
    elif analysis_type == "Tabla de contingencia":
        with col_left:
            if len(categorical_vars) < 2:
                st.warning("Necesitas al menos 2 variables categóricas")
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
    alpha = st.slider("Nivel de significancia (α)", 0.001, 0.10, 0.05, 0.005)
    
    # ─── Ejecutar test ────────────────────────────────────────────────────────
    if st.button("🚀 Ejecutar análisis", type="primary", use_container_width=True):
        with st.spinner("Calculando..."):
            result = run_test(selected_test_id, df, var_dep, var_group, selected_groups, alpha)
            st.session_state.results.append(result)
        
        if result.get('success'):
            st.subheader("Resultados")
            
            # Métricas principales
            rcols = st.columns(3)
            with rcols[0]:
                st.metric("Estadístico", f"{result.get('statistic', 0):.4f}")
            with rcols[1]:
                p = result.get('p_value', 1)
                st.metric("p-valor", f"{p:.6f}")
            with rcols[2]:
                if result['significant']:
                    st.markdown("### 🔴 Significativo")
                else:
                    st.markdown("### 🟢 No significativo")
            
            # Effect size
            if result.get('cohens_d'):
                st.info(f"📏 Tamaño del efecto: d de Cohen = {result['cohens_d']:.3f}")
            if result.get('eta_squared'):
                st.info(f"📏 Tamaño del efecto: η² = {result['eta_squared']:.3f}")
            if result.get('r_squared') is not None:
                st.info(f"📏 R² = {result['r_squared']:.3f}")
            
            # Detalle completo
            with st.expander("📄 Detalle completo", expanded=True):
                st.text(format_result_text(result))
            
            # Post-hoc
            if result.get('posthoc'):
                with st.expander(f"🔍 Post-hoc: {result['posthoc_name']}", expanded=True):
                    ph_df = pd.DataFrame(result['posthoc'])
                    st.dataframe(ph_df.round(4), use_container_width=True)
        else:
            st.error(f"Error en el análisis: {result.get('error', 'Desconocido')}")
    
    # ─── PASO 4: Figuras ─────────────────────────────────────────────────────
    st.header("4. Figuras")
    
    available_figs = {}
    if analysis_type == "Comparación de grupos":
        available_figs = {
            "Box plot + puntos": "boxplot",
            "Violin plot": "violin",
            "Barras + error (SEM)": "bar_error",
            "Datos pareados": "paired",
            "Histograma por grupo": "histogram",
        }
    elif analysis_type == "Correlación / Regresión":
        available_figs = {
            "Scatter + regresión": "scatter",
            "Histograma": "histogram",
        }
    else:
        available_figs = {
            "Histograma": "histogram",
        }
    
    selected_figs = st.multiselect("Selecciona las figuras a generar", list(available_figs.keys()))
    
    if selected_figs and st.button("🎨 Generar figuras", use_container_width=True):
        st.session_state.figures = []
        last_result = st.session_state.results[-1] if st.session_state.results else None
        
        for fig_name in selected_figs:
            fig_type = available_figs[fig_name]
            fig = generate_figure(fig_type, df, var_dep, var_group, 
                                selected_groups if analysis_type == "Comparación de grupos" else None,
                                last_result)
            st.session_state.figures.append(fig)
            st.pyplot(fig)
            
            # Botón de descarga individual
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            st.download_button(
                f"⬇️ Descargar {fig_name} (PNG, 300dpi)",
                data=buf,
                file_name=f"statlab_{fig_type}.png",
                mime="image/png"
            )
    
    # ─── PASO 5: Informe ─────────────────────────────────────────────────────
    st.header("5. Descargar informe")
    
    valid_results = [r for r in st.session_state.results if r.get('success')]
    
    if valid_results:
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            # Texto plano
            full_text = "STATLAB - INFORME DE ANÁLISIS ESTADÍSTICO\n"
            full_text += f"{'='*60}\n\n"
            for r in valid_results:
                full_text += format_result_text(r) + "\n"
            
            st.download_button(
                "📄 Informe TXT",
                data=full_text,
                file_name="statlab_informe.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col_dl2:
            # PDF
            if st.button("📑 Generar informe PDF", use_container_width=True):
                with st.spinner("Generando PDF..."):
                    pdf_buf = generate_pdf_report(valid_results, st.session_state.figures)
                    st.download_button(
                        "⬇️ Descargar PDF",
                        data=pdf_buf,
                        file_name="statlab_informe.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
    else:
        st.info("Ejecuta al menos un análisis para poder generar el informe.")

    # ─── Historial ────────────────────────────────────────────────────────────
    if len(st.session_state.results) > 1:
        with st.expander(f"📚 Historial de análisis ({len(valid_results)} realizados)"):
            for i, r in enumerate(valid_results):
                st.markdown(f"**{i+1}. {r.get('test_name', 'N/A')}** — "
                          f"{r['var_dep']} × {r['var_group']} — "
                          f"p = {r.get('p_value', 'N/A'):.4f}" if isinstance(r.get('p_value'), float) else f"**{i+1}.** Error")

else:
    # Estado vacío
    st.markdown("---")
    st.markdown("""
    ### ¿Cómo funciona?
    
    **1.** Sube un archivo CSV o Excel con tus datos  
    **2.** Define qué variables son continuas y cuáles categóricas  
    **3.** Elige qué quieres analizar y StatLab te sugerirá el test adecuado  
    **4.** Genera figuras de calidad publicación  
    **5.** Descarga el informe con todos los resultados  
    
    ### Tests disponibles
    
    **Comparación de 2 grupos:** T-test, Welch, Mann-Whitney, Wilcoxon  
    **Comparación de >2 grupos:** ANOVA + Tukey, Kruskal-Wallis + Dunn  
    **Correlación:** Pearson, Spearman, regresión lineal  
    **Categóricas:** Chi-cuadrado, Fisher  
    """)
