"""Tests estadisticos: ejecucion, normalidad y sugerencia automatica."""

import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp


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
    """Sugiere el test estadistico mas adecuado."""
    suggestions = []
    if var_dep_type == 'Continua' and var_group_type == 'Categorica':
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
    elif var_dep_type == 'Categorica' and var_group_type == 'Categorica':
        suggestions.append(("Chi-cuadrado", "chi2"))
        suggestions.append(("Test exacto de Fisher", "fisher"))
    elif var_dep_type == 'Continua' and var_group_type == 'Continua':
        if normal:
            suggestions.append(("Correlacion de Pearson", "pearson"))
        else:
            suggestions.append(("Correlacion de Spearman", "spearman"))
        suggestions.append(("Regresion lineal", "linear_reg"))
    return suggestions


def _run_two_groups(test_id, df, var_dep, var_group, groups, alpha, paired_id_col=None):
    """Ejecuta tests de 2 grupos (t-test, Welch, Mann-Whitney, Wilcoxon)."""
    result = {}

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
    elif test_id in ('t_paired', 'wilcoxon'):
        # U3: Emparejar por columna ID si se proporciona
        if paired_id_col and paired_id_col in df.columns:
            df1 = df[df[var_group] == groups[0]][[paired_id_col, var_dep]].dropna()
            df2 = df[df[var_group] == groups[1]][[paired_id_col, var_dep]].dropna()
            merged = df1.merge(df2, on=paired_id_col, suffixes=('_1', '_2'))
            paired_g1 = merged[f'{var_dep}_1'].values
            paired_g2 = merged[f'{var_dep}_2'].values
            if len(merged) < len(df1) or len(merged) < len(df2):
                result["warning"] = (f"Emparejados por '{paired_id_col}': "
                                     f"{len(merged)} pares de {len(df1)}/{len(df2)} sujetos.")
        else:
            min_len = min(len(g1), len(g2))
            if len(g1) != len(g2):
                result["warning"] = (f"Grupos con distinto n ({len(g1)} vs {len(g2)}). "
                                     f"Se usaron los primeros {min_len} de cada grupo.")
            paired_g1 = g1.values[:min_len]
            paired_g2 = g2.values[:min_len]
        if test_id == 't_paired':
            stat, p = stats.ttest_rel(paired_g1, paired_g2)
            result["test_name"] = "T-test pareado"
        else:
            stat, p = stats.wilcoxon(paired_g1, paired_g2)
            result["test_name"] = "Wilcoxon signed-rank"
    elif test_id == 'mann_whitney':
        stat, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        result["test_name"] = "Mann-Whitney U"

    result["statistic"] = float(stat)
    result["p_value"] = float(p)
    result["significant"] = p < alpha

    # Effect size (Cohen's d) — pooled std ponderado por n
    n1, n2 = len(g1), len(g2)
    pooled_std = np.sqrt(((n1 - 1) * g1.std()**2 + (n2 - 1) * g2.std()**2) / (n1 + n2 - 2))
    if pooled_std > 0:
        result["cohens_d"] = float(abs(g1.mean() - g2.mean()) / pooled_std)

    return result


def _run_multi_groups(test_id, df, var_dep, var_group, groups, alpha):
    """Ejecuta tests de >2 grupos (ANOVA, Kruskal-Wallis) con post-hoc."""
    result = {}
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
        if p < alpha and len(group_data) > 2:
            all_data = pd.concat([pd.DataFrame({var_dep: gd, var_group: name})
                                  for gd, name in zip(group_data, group_names)])
            try:
                posthoc = sp.posthoc_tukey(all_data, val_col=var_dep, group_col=var_group)
                result["posthoc"] = posthoc.to_dict()
                result["posthoc_name"] = "Tukey HSD"
            except Exception as e:
                result["posthoc_error"] = f"Tukey HSD fallo: {str(e)}"
    else:
        stat, p = stats.kruskal(*group_data)
        result["test_name"] = "Kruskal-Wallis"
        if p < alpha and len(group_data) > 2:
            all_data = pd.concat([pd.DataFrame({var_dep: gd, var_group: name})
                                  for gd, name in zip(group_data, group_names)])
            try:
                posthoc = sp.posthoc_dunn(all_data, val_col=var_dep, group_col=var_group,
                                          p_adjust='bonferroni')
                result["posthoc"] = posthoc.to_dict()
                result["posthoc_name"] = "Dunn (Bonferroni)"
            except Exception as e:
                result["posthoc_error"] = f"Dunn fallo: {str(e)}"

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

    return result


def _run_correlation(test_id, df, var_dep, var_group, alpha):
    """Ejecuta correlacion de Pearson o Spearman."""
    result = {}
    clean = df[[var_dep, var_group]].dropna()
    x, y = clean[var_dep], clean[var_group]

    if test_id == 'pearson':
        stat, p = stats.pearsonr(x, y)
        result["test_name"] = "Correlacion de Pearson"
    else:
        stat, p = stats.spearmanr(x, y)
        result["test_name"] = "Correlacion de Spearman"

    result["statistic"] = float(stat)
    result["p_value"] = float(p)
    result["significant"] = p < alpha
    result["r_squared"] = float(stat**2)
    result["n"] = len(clean)
    return result


def _run_regression(df, var_dep, var_group, alpha):
    """Ejecuta regresion lineal simple."""
    result = {}
    clean = df[[var_dep, var_group]].dropna()
    slope, intercept, r, p, se = stats.linregress(clean[var_group], clean[var_dep])
    result["test_name"] = "Regresion lineal"
    result["slope"] = float(slope)
    result["intercept"] = float(intercept)
    result["statistic"] = float(r)
    result["r_squared"] = float(r**2)
    result["p_value"] = float(p)
    result["std_error"] = float(se)
    result["significant"] = p < alpha
    result["n"] = len(clean)
    return result


def _run_categorical(test_id, df, var_dep, var_group, alpha):
    """Ejecuta chi-cuadrado o Fisher."""
    result = {}
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
    return result


def run_test(test_id, df, var_dep, var_group, groups=None, alpha=0.05, paired_id_col=None):
    """Ejecuta el test estadistico seleccionado."""
    result = {"test": test_id, "var_dep": var_dep, "var_group": var_group, "alpha": alpha}

    try:
        if test_id in ('t_independent', 't_welch', 'mann_whitney', 'wilcoxon', 't_paired'):
            result.update(_run_two_groups(test_id, df, var_dep, var_group, groups, alpha,
                                          paired_id_col=paired_id_col))
        elif test_id in ('anova', 'kruskal'):
            result.update(_run_multi_groups(test_id, df, var_dep, var_group, groups, alpha))
        elif test_id in ('pearson', 'spearman'):
            result.update(_run_correlation(test_id, df, var_dep, var_group, alpha))
        elif test_id == 'linear_reg':
            result.update(_run_regression(df, var_dep, var_group, alpha))
        elif test_id in ('chi2', 'fisher'):
            result.update(_run_categorical(test_id, df, var_dep, var_group, alpha))

        result["success"] = True
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)

    return result
