"""Formateo de resultados en texto plano e interpretacion para publicacion."""

import pandas as pd


def format_result_text(result):
    """Formatea un resultado como texto para el informe."""
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"TEST: {result.get('test_name', result.get('test', 'N/A'))}")
    lines.append(f"{'='*60}")
    lines.append(f"Variable dependiente: {result.get('var_dep', 'N/A')}")
    lines.append(f"Variable de agrupacion: {result.get('var_group', 'N/A')}")
    lines.append(f"Nivel de significancia: alpha = {result.get('alpha', 0.05)}")
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
    if isinstance(result.get('statistic'), (int, float)):
        lines.append(f"  Estadistico = {result['statistic']:.4f}")
    else:
        lines.append(f"  Estadistico = {result.get('statistic', 'N/A')}")

    if result.get('dof') is not None:
        lines.append(f"  Grados de libertad = {result['dof']}")

    p = result.get('p_value')
    if p is not None:
        lines.append(f"  p-valor = {p:.6f}")
        if p < 0.001:
            lines.append("  Significancia: *** (p < 0.001)")
        elif p < 0.01:
            lines.append("  Significancia: ** (p < 0.01)")
        elif p < 0.05:
            lines.append("  Significancia: * (p < 0.05)")
        else:
            lines.append("  Significancia: ns (p >= 0.05)")

    if result.get('ci_lower') is not None and result.get('ci_upper') is not None:
        lines.append(f"  IC 95% = [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")

    if result.get('cohens_d') is not None:
        d = result['cohens_d']
        lines.append(f"  d de Cohen = {d:.4f} (efecto {_effect_label(d)})")

    if result.get('eta_squared') is not None:
        eta = result['eta_squared']
        lines.append(f"  eta2 = {eta:.4f} (efecto {_eta_label(eta)})")

    if result.get('r_squared') is not None:
        lines.append(f"  R2 = {result['r_squared']:.4f}")

    if result.get('slope') is not None:
        lines.append(f"  Pendiente = {result['slope']:.4f} +/- {result.get('std_error', 0):.4f}")
        lines.append(f"  Intercepto = {result['intercept']:.4f}")

    if result.get('bias') is not None:
        lines.append(f"  Sesgo (bias) = {result['bias']:.4f}")
        lines.append(f"  DE de diferencias = {result['sd_diff']:.4f}")
        lines.append(f"  Limite superior (+1.96 DE) = {result['loa_upper']:.4f}")
        lines.append(f"  Limite inferior (-1.96 DE) = {result['loa_lower']:.4f}")

    if result.get('auc') is not None:
        lines.append(f"  AUC = {result['auc']:.4f}")
        if result.get('best_threshold') is not None:
            lines.append(f"  Corte optimo = {result['best_threshold']:.4f}")
            lines.append(f"  Sensibilidad = {result['sensitivity']:.4f}")
            lines.append(f"  Especificidad = {result['specificity']:.4f}")

    if result.get('curves'):
        lines.append("  CURVAS DE SUPERVIVENCIA:")
        for label, data in result['curves'].items():
            median = data.get('median')
            med_str = f"{median:.1f}" if median is not None else "no alcanzada"
            lines.append(f"    {label}: n={data['n']}, mediana={med_str}")
        if result.get('logrank_name'):
            lines.append(f"  {result['logrank_name']}")

    if result.get('posthoc'):
        lines.append(f"\n  POST-HOC ({result.get('posthoc_name', 'N/A')}):")
        ph = pd.DataFrame(result['posthoc'])
        lines.append(f"  {ph.to_string()}")

    lines.append("")
    sig = result.get('significant', None)
    if sig is not None:
        if sig:
            lines.append("CONCLUSION: Se rechaza H0. La diferencia/asociacion ES estadisticamente significativa.")
        else:
            lines.append("CONCLUSION: No se rechaza H0. La diferencia/asociacion NO es estadisticamente significativa.")

    lines.append("")
    return "\n".join(lines)


def _fmt_p(p):
    """Formatea p-valor para texto de publicacion."""
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def _effect_label(d):
    """Etiqueta del tamano del efecto para Cohen's d."""
    if d < 0.5:
        return "pequeno"
    if d < 0.8:
        return "mediano"
    return "grande"


def _eta_label(eta):
    """Etiqueta del tamano del efecto para eta cuadrado."""
    if eta < 0.06:
        return "pequeno"
    if eta < 0.14:
        return "mediano"
    return "grande"


def _auc_label(auc):
    """Calificacion de AUC."""
    if auc >= 0.9:
        return "excelente"
    if auc >= 0.8:
        return "buena"
    if auc >= 0.7:
        return "aceptable"
    if auc >= 0.6:
        return "pobre"
    return "sin capacidad discriminativa"


def generate_interpretation(result):
    """Genera un parrafo de interpretacion listo para la seccion Resultados de un paper."""
    test_id = result.get('test', '')
    var_dep = result.get('var_dep', '')
    var_group = result.get('var_group', '')
    p = result.get('p_value')
    sig = result.get('significant')

    # --- Dos grupos ---
    if test_id in ('t_independent', 't_welch', 'mann_whitney', 't_paired', 'wilcoxon'):
        groups = result.get('groups', ['Grupo 1', 'Grupo 2'])
        means = result.get('mean', [])
        stds = result.get('std', [])
        ns = result.get('n', [])
        test_name = result.get('test_name', '')
        stat = result.get('statistic', 0)

        g1_desc = f"{groups[0]} (media = {means[0]:.2f} +/- {stds[0]:.2f}, n = {ns[0]})"
        g2_desc = f"{groups[1]} (media = {means[1]:.2f} +/- {stds[1]:.2f}, n = {ns[1]})"

        if sig:
            txt = (f"Se encontro una diferencia estadisticamente significativa "
                   f"en {var_dep} entre los grupos {g1_desc} y {g2_desc}; "
                   f"{test_name}, estadistico = {stat:.2f}, {_fmt_p(p)}")
        else:
            txt = (f"No se encontraron diferencias estadisticamente significativas "
                   f"en {var_dep} entre los grupos {g1_desc} y {g2_desc}; "
                   f"{test_name}, estadistico = {stat:.2f}, {_fmt_p(p)}")

        if result.get('ci_lower') is not None:
            txt += (f", diferencia de medias = {result.get('mean_diff', 0):.2f} "
                    f"(IC 95%: [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}])")

        if result.get('cohens_d') is not None:
            d = result['cohens_d']
            txt += f", d de Cohen = {d:.2f} (efecto {_effect_label(d)})"

        txt += "."
        return txt

    # --- Multiples grupos ---
    if test_id in ('anova', 'kruskal'):
        groups = result.get('groups', [])
        test_name = result.get('test_name', '')
        stat = result.get('statistic', 0)

        group_list = ", ".join(groups)
        if sig:
            txt = (f"Se encontraron diferencias estadisticamente significativas "
                   f"en {var_dep} entre los grupos ({group_list}); "
                   f"{test_name}, estadistico = {stat:.2f}, {_fmt_p(p)}")
        else:
            txt = (f"No se encontraron diferencias estadisticamente significativas "
                   f"en {var_dep} entre los grupos ({group_list}); "
                   f"{test_name}, estadistico = {stat:.2f}, {_fmt_p(p)}")

        if result.get('eta_squared') is not None:
            eta = result['eta_squared']
            txt += f", eta2 = {eta:.3f} (efecto {_eta_label(eta)})"

        txt += "."

        if result.get('posthoc') and sig:
            ph = pd.DataFrame(result['posthoc'])
            sig_pairs = []
            for row_name in ph.index:
                for col_name in ph.columns:
                    if row_name < col_name:
                        pval = ph.loc[row_name, col_name]
                        if isinstance(pval, (int, float)) and pval < 0.05:
                            sig_pairs.append(f"{row_name} vs {col_name} ({_fmt_p(pval)})")
            if sig_pairs:
                txt += (f" El analisis post-hoc ({result.get('posthoc_name', '')}) "
                        f"revelo diferencias significativas entre: {'; '.join(sig_pairs)}.")

        return txt

    # --- Correlacion ---
    if test_id in ('pearson', 'spearman'):
        test_name = result.get('test_name', '')
        r = result.get('statistic', 0)
        n = result.get('n', 0)
        direction = "positiva" if r > 0 else "negativa"
        strength = "debil" if abs(r) < 0.3 else "moderada" if abs(r) < 0.7 else "fuerte"

        _ci_str = ""
        if result.get('ci_lower') is not None:
            _ci_str = f", IC 95%: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]"

        if sig:
            txt = (f"Se observo una correlacion {direction} {strength} estadisticamente "
                   f"significativa entre {var_dep} y {var_group} "
                   f"({test_name}, r = {r:.3f}{_ci_str}, {_fmt_p(p)}, n = {n})")
        else:
            txt = (f"No se observo una correlacion estadisticamente significativa "
                   f"entre {var_dep} y {var_group} "
                   f"({test_name}, r = {r:.3f}{_ci_str}, {_fmt_p(p)}, n = {n})")

        if result.get('r_squared') is not None:
            txt += f". {var_group} explica el {result['r_squared']*100:.1f}% de la varianza de {var_dep}"

        txt += "."
        return txt

    # --- Regresion lineal ---
    if test_id == 'linear_reg':
        slope = result.get('slope', 0)
        intercept = result.get('intercept', 0)
        se = result.get('std_error', 0)
        r2 = result.get('r_squared', 0)
        n = result.get('n', 0)

        _ci_str = ""
        if result.get('ci_lower') is not None:
            _ci_str = f", IC 95%: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]"

        if sig:
            txt = (f"{var_group} predice significativamente {var_dep} "
                   f"(pendiente = {slope:.3f} +/- {se:.3f}{_ci_str}, {_fmt_p(p)}, "
                   f"R2 = {r2:.3f}, n = {n}). "
                   f"Por cada unidad de incremento en {var_group}, "
                   f"{var_dep} cambia en {slope:.3f} unidades.")
        else:
            txt = (f"No se encontro una relacion lineal significativa entre "
                   f"{var_group} y {var_dep} "
                   f"(pendiente = {slope:.3f} +/- {se:.3f}{_ci_str}, {_fmt_p(p)}, "
                   f"R2 = {r2:.3f}, n = {n}).")
        return txt

    # --- Categoricas ---
    if test_id in ('chi2', 'fisher'):
        test_name = result.get('test_name', '')
        stat = result.get('statistic', 0)

        if sig:
            txt = (f"Se encontro una asociacion estadisticamente significativa "
                   f"entre {var_dep} y {var_group} "
                   f"({test_name}, estadistico = {stat:.2f}, {_fmt_p(p)}).")
        else:
            txt = (f"No se encontro una asociacion estadisticamente significativa "
                   f"entre {var_dep} y {var_group} "
                   f"({test_name}, estadistico = {stat:.2f}, {_fmt_p(p)}).")
        return txt

    # --- Bland-Altman ---
    if test_id == 'bland_altman':
        bias = result.get('bias', 0)
        sd = result.get('sd_diff', 0)
        loa_l = result.get('loa_lower', 0)
        loa_u = result.get('loa_upper', 0)
        n = result.get('n', 0)

        _ci_str = ""
        if result.get('ci_lower') is not None:
            _ci_str = f" (IC 95%: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}])"

        bias_sig = "estadisticamente significativo" if sig else "no significativo"
        txt = (f"El analisis de Bland-Altman (n = {n}) mostro un sesgo medio de "
               f"{bias:.3f}{_ci_str} (DE = {sd:.3f}) entre {var_dep} y {var_group}, "
               f"con limites de acuerdo de [{loa_l:.3f}, {loa_u:.3f}]. "
               f"El sesgo fue {bias_sig} ({_fmt_p(p)}).")
        return txt

    # --- ROC ---
    if test_id == 'roc':
        auc = result.get('auc', 0)
        n = result.get('n', 0)
        label = result.get('positive_label', '')
        quality = _auc_label(auc)

        txt = (f"La curva ROC (n = {n}) mostro una capacidad discriminativa "
               f"{quality} de {var_group} para predecir {var_dep} = {label} "
               f"(AUC = {auc:.3f})")

        if result.get('best_threshold') is not None:
            sens = result['sensitivity']
            spec = result['specificity']
            cut = result['best_threshold']
            txt += (f". El punto de corte optimo fue {cut:.2f} "
                    f"(sensibilidad = {sens:.2f}, especificidad = {spec:.2f})")
        txt += "."
        return txt

    # --- Kaplan-Meier ---
    if test_id == 'kaplan_meier':
        curves = result.get('curves', {})
        parts = []
        for label, data in curves.items():
            med = data.get('median')
            med_str = f"{med:.1f}" if med is not None else "no alcanzada"
            parts.append(f"{label} (n = {data['n']}, mediana = {med_str})")

        txt = f"El analisis de Kaplan-Meier mostro las siguientes curvas de supervivencia: {'; '.join(parts)}"

        if isinstance(p, (int, float)):
            if sig:
                txt += (f". La comparacion mediante log-rank test revelo diferencias "
                        f"significativas entre los grupos (estadistico = {result.get('statistic', 0):.2f}, "
                        f"{_fmt_p(p)})")
            else:
                txt += (f". La comparacion mediante log-rank test no revelo diferencias "
                        f"significativas entre los grupos (estadistico = {result.get('statistic', 0):.2f}, "
                        f"{_fmt_p(p)})")
        txt += "."
        return txt

    return ""
