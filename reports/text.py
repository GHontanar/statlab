"""Formateo de resultados en texto plano."""

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

    if result.get('cohens_d') is not None:
        d = result['cohens_d']
        effect = "pequeno" if d < 0.5 else "mediano" if d < 0.8 else "grande"
        lines.append(f"  d de Cohen = {d:.4f} (efecto {effect})")

    if result.get('eta_squared') is not None:
        eta = result['eta_squared']
        effect = "pequeno" if eta < 0.06 else "mediano" if eta < 0.14 else "grande"
        lines.append(f"  eta2 = {eta:.4f} (efecto {effect})")

    if result.get('r_squared') is not None:
        lines.append(f"  R2 = {result['r_squared']:.4f}")

    if result.get('slope') is not None:
        lines.append(f"  Pendiente = {result['slope']:.4f} +/- {result.get('std_error', 0):.4f}")
        lines.append(f"  Intercepto = {result['intercept']:.4f}")

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
