"""Tests estadisticos: ejecucion, normalidad y sugerencia automatica."""

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats


def check_normality(data, alpha=0.05):
    """Test de Shapiro-Wilk para normalidad."""
    if len(data) < 3 or len(data) > 5000:
        return None, None
    clean = data.dropna()
    if len(clean) < 3:
        return None, None
    stat, p = stats.shapiro(clean)
    return stat, p


def check_homogeneity(groups_data):
    """Test de Levene para homogeneidad de varianzas.

    Retorna (statistic, p_value) o (None, None) si no es posible.
    """
    valid = [g.dropna() for g in groups_data if len(g.dropna()) >= 2]
    if len(valid) < 2:
        return None, None
    stat, p = stats.levene(*valid)
    return stat, p


def suggest_test(var_dep_type, var_group_type, n_groups, paired=False, normal=True,
                 equal_var=True):
    """Sugiere el test estadistico mas adecuado."""
    suggestions = []
    if var_dep_type == 'Continua' and var_group_type == 'Categorica':
        if n_groups == 2:
            if normal:
                if paired:
                    suggestions.append(("T-test pareado", "t_paired"))
                elif equal_var:
                    suggestions.append(("T-test independiente", "t_independent"))
                    suggestions.append(("T-test Welch", "t_welch"))
                else:
                    suggestions.append(("T-test de Welch", "t_welch"))
                    suggestions.append(("T-test independiente", "t_independent"))
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

    # IC 95% para la diferencia de medias
    mean_diff = float(g1.mean() - g2.mean())
    se_diff = np.sqrt(g1.var(ddof=1) / n1 + g2.var(ddof=1) / n2)
    dof_welch = (g1.var(ddof=1) / n1 + g2.var(ddof=1) / n2)**2 / (
        (g1.var(ddof=1) / n1)**2 / (n1 - 1) + (g2.var(ddof=1) / n2)**2 / (n2 - 1))
    t_crit = stats.t.ppf(1 - alpha / 2, dof_welch)
    result["ci_lower"] = float(mean_diff - t_crit * se_diff)
    result["ci_upper"] = float(mean_diff + t_crit * se_diff)
    result["mean_diff"] = mean_diff

    return result


# F6: Metodos post-hoc disponibles
POSTHOC_METHODS = {
    'anova': {
        'tukey': ('Tukey HSD', 'Controla FWER. El mas usado para ANOVA.'),
        'scheffe': ('Scheffe', 'Mas conservador. Util para comparaciones no planificadas.'),
        'bonferroni_t': ('Bonferroni (t-test)', 'T-tests pareados con correccion Bonferroni.'),
        'holm_t': ('Holm (t-test)', 'Menos conservador que Bonferroni, mas potencia.'),
    },
    'kruskal': {
        'dunn_bonferroni': ('Dunn (Bonferroni)', 'Correccion Bonferroni. Conservador.'),
        'dunn_holm': ('Dunn (Holm)', 'Menos conservador, mas potencia.'),
        'dunn_bh': ('Dunn (Benjamini-Hochberg)', 'Controla FDR en vez de FWER. Menos conservador.'),
        'conover_bonferroni': ('Conover (Bonferroni)', 'Mas potente que Dunn, usa t-aproximacion.'),
    },
}


def _run_posthoc(test_id, posthoc_method, all_data, var_dep, var_group):
    """Ejecuta el analisis post-hoc seleccionado."""
    if test_id == 'anova':
        if posthoc_method == 'scheffe':
            ph = sp.posthoc_scheffe(all_data, val_col=var_dep, group_col=var_group)
            return ph, 'Scheffe'
        elif posthoc_method == 'bonferroni_t':
            ph = sp.posthoc_ttest(all_data, val_col=var_dep, group_col=var_group,
                                  p_adjust='bonferroni')
            return ph, 'Bonferroni (t-test)'
        elif posthoc_method == 'holm_t':
            ph = sp.posthoc_ttest(all_data, val_col=var_dep, group_col=var_group,
                                  p_adjust='holm')
            return ph, 'Holm (t-test)'
        else:  # tukey (default)
            ph = sp.posthoc_tukey(all_data, val_col=var_dep, group_col=var_group)
            return ph, 'Tukey HSD'
    else:
        if posthoc_method == 'dunn_holm':
            ph = sp.posthoc_dunn(all_data, val_col=var_dep, group_col=var_group,
                                 p_adjust='holm')
            return ph, 'Dunn (Holm)'
        elif posthoc_method == 'dunn_bh':
            ph = sp.posthoc_dunn(all_data, val_col=var_dep, group_col=var_group,
                                 p_adjust='fdr_bh')
            return ph, 'Dunn (Benjamini-Hochberg)'
        elif posthoc_method == 'conover_bonferroni':
            ph = sp.posthoc_conover(all_data, val_col=var_dep, group_col=var_group,
                                    p_adjust='bonferroni')
            return ph, 'Conover (Bonferroni)'
        else:  # dunn_bonferroni (default)
            ph = sp.posthoc_dunn(all_data, val_col=var_dep, group_col=var_group,
                                 p_adjust='bonferroni')
            return ph, 'Dunn (Bonferroni)'


def _run_multi_groups(test_id, df, var_dep, var_group, groups, alpha,
                      posthoc_method=None):
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
    else:
        stat, p = stats.kruskal(*group_data)
        result["test_name"] = "Kruskal-Wallis"

    result["statistic"] = float(stat)
    result["p_value"] = float(p)
    result["significant"] = p < alpha

    # Post-hoc si es significativo y hay >2 grupos
    if p < alpha and len(group_data) > 2:
        all_data = pd.concat([pd.DataFrame({var_dep: gd, var_group: name})
                              for gd, name in zip(group_data, group_names)])
        # Default post-hoc si no se especifica
        if posthoc_method is None:
            posthoc_method = 'tukey' if test_id == 'anova' else 'dunn_bonferroni'
        try:
            posthoc, posthoc_name = _run_posthoc(test_id, posthoc_method,
                                                  all_data, var_dep, var_group)
            result["posthoc"] = posthoc.to_dict()
            result["posthoc_name"] = posthoc_name
        except Exception as e:
            result["posthoc_error"] = f"Post-hoc fallo: {str(e)}"

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

    # IC 95% para r (transformacion z de Fisher)
    n = len(clean)
    if n > 3:
        z = np.arctanh(stat)
        se_z = 1 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci_z_lo, ci_z_hi = z - z_crit * se_z, z + z_crit * se_z
        result["ci_lower"] = float(np.tanh(ci_z_lo))
        result["ci_upper"] = float(np.tanh(ci_z_hi))

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

    # IC 95% para la pendiente
    n = len(clean)
    t_crit = stats.t.ppf(1 - alpha / 2, n - 2)
    result["ci_lower"] = float(slope - t_crit * se)
    result["ci_upper"] = float(slope + t_crit * se)

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


def _run_bland_altman(df, var_dep, var_group):
    """Bland-Altman: concordancia entre dos metodos de medicion."""
    result = {}
    clean = df[[var_dep, var_group]].dropna()
    m1, m2 = clean[var_dep].values, clean[var_group].values
    diff = m1 - m2
    mean = (m1 + m2) / 2

    bias = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    loa_upper = bias + 1.96 * sd_diff
    loa_lower = bias - 1.96 * sd_diff

    result["test_name"] = "Bland-Altman"
    result["bias"] = bias
    result["sd_diff"] = sd_diff
    result["loa_upper"] = float(loa_upper)
    result["loa_lower"] = float(loa_lower)
    result["n"] = len(clean)
    result["means"] = mean.tolist()
    result["diffs"] = diff.tolist()

    # IC 95% para el sesgo
    n = len(clean)
    se_bias = sd_diff / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, n - 1)
    result["ci_lower"] = float(bias - t_crit * se_bias)
    result["ci_upper"] = float(bias + t_crit * se_bias)

    # Test de sesgo (bias != 0)
    stat, p = stats.ttest_1samp(diff, 0)
    result["statistic"] = float(stat)
    result["p_value"] = float(p)
    result["significant"] = p < 0.05
    return result


def _run_roc(df, var_dep, var_group, positive_label=None):
    """Curva ROC para variable binaria vs predictor continuo."""
    result = {}
    clean = df[[var_dep, var_group]].dropna()

    # Binarizar variable de resultado
    labels = sorted(clean[var_dep].unique())
    if len(labels) != 2:
        raise ValueError(f"La variable '{var_dep}' debe tener exactamente 2 categorias, "
                         f"tiene {len(labels)}: {labels}")

    pos = positive_label if positive_label else labels[1]
    y_true = (clean[var_dep] == pos).astype(int).values
    y_score = clean[var_group].values

    # Calcular ROC manualmente (sin sklearn)
    thresholds = np.sort(np.unique(y_score))[::-1]
    tpr_list, fpr_list = [0.0], [0.0]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    for t in thresholds:
        predicted = (y_score >= t).astype(int)
        tp = ((predicted == 1) & (y_true == 1)).sum()
        fp = ((predicted == 1) & (y_true == 0)).sum()
        tpr_list.append(tp / n_pos if n_pos > 0 else 0)
        fpr_list.append(fp / n_neg if n_neg > 0 else 0)

    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)

    # Ordenar por fpr
    order = np.argsort(fpr)
    fpr, tpr = fpr[order], tpr[order]

    # AUC (trapezoidal)
    auc = float(np.trapezoid(tpr, fpr))

    # Youden's J para cutoff optimo
    j_scores = tpr_list[1:] - np.array(fpr_list[1:])
    best_idx = np.argmax(j_scores)
    best_threshold = float(thresholds[best_idx])
    best_sens = float(tpr_list[best_idx + 1])
    best_spec = float(1 - fpr_list[best_idx + 1])

    result["test_name"] = "Curva ROC"
    result["auc"] = auc
    result["fpr"] = fpr.tolist()
    result["tpr"] = tpr.tolist()
    result["best_threshold"] = best_threshold
    result["sensitivity"] = best_sens
    result["specificity"] = best_spec
    result["positive_label"] = str(pos)
    result["n"] = len(clean)
    result["statistic"] = auc
    result["p_value"] = None
    result["significant"] = None
    return result


def _run_kaplan_meier(df, var_dep, var_group, group_col=None, groups=None, alpha=0.05):
    """Kaplan-Meier: analisis de supervivencia."""
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    result = {}
    clean = df[[var_dep, var_group]].dropna()
    if group_col:
        clean = df[[var_dep, var_group, group_col]].dropna()

    durations = clean[var_dep].values
    events = clean[var_group].values

    # Validar eventos binarios
    unique_events = sorted(pd.Series(events).unique())
    if not set(unique_events).issubset({0, 1, 0.0, 1.0, True, False}):
        raise ValueError(f"La variable de evento debe ser binaria (0/1), "
                         f"valores encontrados: {unique_events}")
    events = events.astype(int)

    result["test_name"] = "Kaplan-Meier"
    result["n"] = len(clean)
    result["curves"] = {}

    if group_col and group_col in clean.columns:
        group_labels = groups if groups else sorted(clean[group_col].unique())
        for g in group_labels:
            mask = clean[group_col] == g
            kmf = KaplanMeierFitter()
            kmf.fit(durations[mask], event_observed=events[mask], label=str(g))
            result["curves"][str(g)] = {
                "timeline": kmf.survival_function_.index.tolist(),
                "survival": kmf.survival_function_.iloc[:, 0].tolist(),
                "median": float(kmf.median_survival_time_) if np.isfinite(kmf.median_survival_time_) else None,
                "n": int(mask.sum()),
            }

        # Log-rank test si hay 2+ grupos
        if len(group_labels) >= 2:
            g1_mask = clean[group_col] == group_labels[0]
            g2_mask = clean[group_col] == group_labels[1]
            lr = logrank_test(durations[g1_mask], durations[g2_mask],
                              event_observed_A=events[g1_mask],
                              event_observed_B=events[g2_mask])
            result["statistic"] = float(lr.test_statistic)
            result["p_value"] = float(lr.p_value)
            result["significant"] = lr.p_value < alpha
            result["logrank_name"] = f"Log-rank: {group_labels[0]} vs {group_labels[1]}"
    else:
        kmf = KaplanMeierFitter()
        kmf.fit(durations, event_observed=events, label="Global")
        result["curves"]["Global"] = {
            "timeline": kmf.survival_function_.index.tolist(),
            "survival": kmf.survival_function_.iloc[:, 0].tolist(),
            "median": float(kmf.median_survival_time_) if np.isfinite(kmf.median_survival_time_) else None,
            "n": int(len(durations)),
        }
        result["statistic"] = None
        result["p_value"] = None
        result["significant"] = None

    return result


def _run_logistic(df, var_dep, var_group, alpha=0.05):
    """F7: Regresion logistica simple."""
    import statsmodels.api as sm

    result = {}
    clean = df[[var_dep, var_group]].dropna()

    labels = sorted(clean[var_dep].unique())
    if len(labels) != 2:
        raise ValueError(f"La variable '{var_dep}' debe tener exactamente 2 categorias, "
                         f"tiene {len(labels)}: {labels}")

    y = (clean[var_dep] == labels[1]).astype(int).values
    X = sm.add_constant(clean[var_group].values.astype(float))

    model = sm.Logit(y, X).fit(disp=0)

    result["test_name"] = "Regresion logistica"
    result["n"] = len(clean)
    result["positive_label"] = str(labels[1])

    # Coeficientes (intercepto + predictor)
    coef = float(model.params[1])
    se = float(model.bse[1])
    p = float(model.pvalues[1])
    with np.errstate(over='ignore'):
        or_val = float(np.exp(coef))
        or_ci_lower = float(np.exp(coef - stats.norm.ppf(1 - alpha / 2) * se))
        or_ci_upper = float(np.exp(coef + stats.norm.ppf(1 - alpha / 2) * se))

    result["statistic"] = coef
    result["p_value"] = p
    result["significant"] = p < alpha
    result["odds_ratio"] = or_val
    result["or_ci_lower"] = or_ci_lower
    result["or_ci_upper"] = or_ci_upper
    result["ci_lower"] = or_ci_lower
    result["ci_upper"] = or_ci_upper
    result["coef"] = coef
    result["std_error"] = se

    # Pseudo R2 (McFadden)
    result["pseudo_r2"] = float(model.prsquared)

    # AIC
    result["aic"] = float(model.aic)

    return result


def _run_power(result):
    """F8: Calculo de potencia post-hoc basado en el resultado de un test."""
    from statsmodels.stats.power import FTestAnovaPower, TTestIndPower, TTestPower

    test_id = result.get('test', '')
    power_info = {}

    if test_id in ('t_independent', 't_welch', 'mann_whitney'):
        d = result.get('cohens_d')
        ns = result.get('n', [])
        if d and len(ns) >= 2 and min(ns) >= 2:
            analysis = TTestIndPower()
            n_avg = sum(ns) / len(ns)
            ratio = ns[1] / ns[0] if ns[0] > 0 else 1
            power = analysis.power(effect_size=d, nobs1=ns[0], ratio=ratio,
                                   alpha=result.get('alpha', 0.05))
            power_info["power"] = float(power)
            # n necesario para 80% de potencia
            n_needed = analysis.solve_power(effect_size=d, power=0.8,
                                            ratio=ratio,
                                            alpha=result.get('alpha', 0.05))
            power_info["n_for_80"] = int(np.ceil(n_needed))

    elif test_id in ('t_paired', 'wilcoxon'):
        d = result.get('cohens_d')
        ns = result.get('n', [])
        if d and len(ns) >= 2 and min(ns) >= 2:
            analysis = TTestPower()
            n_used = min(ns)
            power = analysis.power(effect_size=d, nobs=n_used,
                                   alpha=result.get('alpha', 0.05))
            power_info["power"] = float(power)
            n_needed = analysis.solve_power(effect_size=d, power=0.8,
                                            alpha=result.get('alpha', 0.05))
            power_info["n_for_80"] = int(np.ceil(n_needed))

    elif test_id == 'anova':
        eta2 = result.get('eta_squared')
        ns = result.get('n', [])
        k = len(ns)
        if eta2 and eta2 < 1 and k >= 2:
            f_effect = np.sqrt(eta2 / (1 - eta2))
            n_avg = sum(ns) / k
            analysis = FTestAnovaPower()
            power = analysis.power(effect_size=f_effect, nobs=n_avg,
                                   k_groups=k,
                                   alpha=result.get('alpha', 0.05))
            power_info["power"] = float(power)
            n_needed = analysis.solve_power(effect_size=f_effect, power=0.8,
                                            k_groups=k,
                                            alpha=result.get('alpha', 0.05))
            power_info["n_for_80"] = int(np.ceil(n_needed))

    return power_info if power_info else None


def _run_icc(df, var_dep, var_group, alpha=0.05):
    """F9: Coeficiente de correlacion intraclase (ICC).

    Calcula ICC(3,1) — two-way mixed, single measures, consistency.
    var_dep = medicion, var_group = evaluador/metodo.
    Requiere una columna de sujeto (extra['subject_col']).
    """
    result = {}
    result["test_name"] = "ICC"

    clean = df.dropna(subset=[var_dep, var_group])
    raters = sorted(clean[var_group].unique())
    k = len(raters)

    if k < 2:
        raise ValueError("Se necesitan al menos 2 evaluadores/metodos para calcular ICC.")

    # Pivotar: filas = sujetos, columnas = evaluadores
    # Necesitamos un sujeto implícito: asumimos filas balanceadas
    n_per_rater = [len(clean[clean[var_group] == r]) for r in raters]
    if len(set(n_per_rater)) != 1:
        raise ValueError(f"Datos no balanceados: cada evaluador debe tener el mismo n. "
                         f"Encontrados: {dict(zip(raters, n_per_rater))}")

    n = n_per_rater[0]  # sujetos por evaluador
    # Construir matriz: cada columna = un evaluador
    matrix = np.column_stack([
        clean[clean[var_group] == r][var_dep].values for r in raters
    ])

    grand_mean = matrix.mean()
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)

    # Sumas de cuadrados
    ss_total = ((matrix - grand_mean) ** 2).sum()
    ss_rows = k * ((row_means - grand_mean) ** 2).sum()  # Between subjects
    ss_cols = n * ((col_means - grand_mean) ** 2).sum()  # Between raters
    ss_error = ss_total - ss_rows - ss_cols

    # Mean squares
    ms_rows = ss_rows / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 0

    # ICC(3,1) — two-way mixed, consistency, single measures
    icc = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error) if (ms_rows + (k - 1) * ms_error) > 0 else 0

    # IC 95% para ICC (basado en F)
    f_value = ms_rows / ms_error if ms_error > 0 else 0
    df1 = n - 1
    df2 = (n - 1) * (k - 1)

    if f_value > 0 and df1 > 0 and df2 > 0:
        f_lo = stats.f.ppf(alpha / 2, df1, df2)
        f_hi = stats.f.ppf(1 - alpha / 2, df1, df2)
        ci_lower = (f_value / f_hi - 1) / (f_value / f_hi + k - 1) if f_hi > 0 else 0
        ci_upper = (f_value / f_lo - 1) / (f_value / f_lo + k - 1) if f_lo > 0 else 1
        result["ci_lower"] = float(max(0, ci_lower))
        result["ci_upper"] = float(min(1, ci_upper))

    result["statistic"] = float(icc)
    result["icc"] = float(icc)
    result["n_subjects"] = n
    result["n_raters"] = k
    result["raters"] = [str(r) for r in raters]
    result["p_value"] = None
    result["significant"] = None

    # Calificacion
    if icc < 0.5:
        result["quality"] = "pobre"
    elif icc < 0.75:
        result["quality"] = "moderada"
    elif icc < 0.9:
        result["quality"] = "buena"
    else:
        result["quality"] = "excelente"

    return result


def run_test(test_id, df, var_dep, var_group, groups=None, alpha=0.05, paired_id_col=None,
             extra=None):
    """Ejecuta el test estadistico seleccionado."""
    result = {"test": test_id, "var_dep": var_dep, "var_group": var_group, "alpha": alpha}

    try:
        if test_id in ('t_independent', 't_welch', 'mann_whitney', 'wilcoxon', 't_paired'):
            result.update(_run_two_groups(test_id, df, var_dep, var_group, groups, alpha,
                                          paired_id_col=paired_id_col))
        elif test_id in ('anova', 'kruskal'):
            posthoc_method = extra.get('posthoc_method') if extra else None
            result.update(_run_multi_groups(test_id, df, var_dep, var_group, groups, alpha,
                                            posthoc_method=posthoc_method))
        elif test_id in ('pearson', 'spearman'):
            result.update(_run_correlation(test_id, df, var_dep, var_group, alpha))
        elif test_id == 'linear_reg':
            result.update(_run_regression(df, var_dep, var_group, alpha))
        elif test_id in ('chi2', 'fisher'):
            result.update(_run_categorical(test_id, df, var_dep, var_group, alpha))
        elif test_id == 'bland_altman':
            result.update(_run_bland_altman(df, var_dep, var_group))
        elif test_id == 'roc':
            positive_label = extra.get('positive_label') if extra else None
            result.update(_run_roc(df, var_dep, var_group, positive_label))
        elif test_id == 'kaplan_meier':
            group_col = extra.get('group_col') if extra else None
            km_groups = extra.get('groups') if extra else None
            result.update(_run_kaplan_meier(df, var_dep, var_group,
                                           group_col=group_col, groups=km_groups,
                                           alpha=alpha))
        elif test_id == 'logistic':
            result.update(_run_logistic(df, var_dep, var_group, alpha))
        elif test_id == 'icc':
            result.update(_run_icc(df, var_dep, var_group, alpha))

        result["success"] = True

        # F8: Potencia post-hoc (si aplica)
        try:
            power_info = _run_power(result)
            if power_info:
                result["power"] = power_info
        except Exception:
            pass  # Potencia es informativa, no bloquea el resultado
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)

    return result
