"""Calculadora de tamano muestral para distintos disenos de estudio."""

import math

import numpy as np
from scipy import stats as sp_stats
from statsmodels.stats.power import (
    FTestAnovaPower,
    NormalIndPower,
    TTestIndPower,
    TTestPower,
)


def sample_size_ttest_ind(effect_size, alpha=0.05, power=0.8, alternative='two-sided'):
    """Tamano muestral para t-test de dos grupos independientes.

    Args:
        effect_size: Cohen's d (0.2=pequeno, 0.5=mediano, 0.8=grande).
        alpha: Nivel de significancia.
        power: Potencia deseada (1 - beta).
        alternative: 'two-sided' o 'one-sided'.

    Returns:
        dict con n_per_group, n_total, y parametros usados.
    """
    if effect_size <= 0:
        return {'success': False, 'error': 'El tamano del efecto debe ser > 0'}

    try:
        analysis = TTestIndPower()
        n = analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            alternative=alternative,
        )
        n_per_group = math.ceil(n)
        return {
            'success': True,
            'n_per_group': n_per_group,
            'n_total': n_per_group * 2,
            'effect_size': effect_size,
            'alpha': alpha,
            'power': power,
            'test': 't-test independiente',
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def sample_size_ttest_paired(effect_size, alpha=0.05, power=0.8):
    """Tamano muestral para t-test pareado.

    Args:
        effect_size: Cohen's d para diferencias pareadas.
        alpha: Nivel de significancia.
        power: Potencia deseada.

    Returns:
        dict con n_pairs y parametros usados.
    """
    if effect_size <= 0:
        return {'success': False, 'error': 'El tamano del efecto debe ser > 0'}

    try:
        analysis = TTestPower()
        n = analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            alternative='two-sided',
        )
        n_pairs = math.ceil(n)
        return {
            'success': True,
            'n_per_group': n_pairs,
            'n_total': n_pairs,
            'effect_size': effect_size,
            'alpha': alpha,
            'power': power,
            'test': 't-test pareado',
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def sample_size_anova(effect_size, k_groups, alpha=0.05, power=0.8):
    """Tamano muestral por grupo para ANOVA one-way.

    Args:
        effect_size: Cohen's f (0.1=pequeno, 0.25=mediano, 0.4=grande).
        k_groups: Numero de grupos.
        alpha: Nivel de significancia.
        power: Potencia deseada.

    Returns:
        dict con n_per_group, n_total, y parametros usados.
    """
    if effect_size <= 0:
        return {'success': False, 'error': 'El tamano del efecto debe ser > 0'}
    if k_groups < 2:
        return {'success': False, 'error': 'Se necesitan al menos 2 grupos'}

    try:
        analysis = FTestAnovaPower()
        n = analysis.solve_power(
            effect_size=effect_size,
            k_groups=k_groups,
            alpha=alpha,
            power=power,
        )
        n_per_group = math.ceil(n)
        return {
            'success': True,
            'n_per_group': n_per_group,
            'n_total': n_per_group * k_groups,
            'k_groups': k_groups,
            'effect_size': effect_size,
            'alpha': alpha,
            'power': power,
            'test': f'ANOVA ({k_groups} grupos)',
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def sample_size_correlation(r, alpha=0.05, power=0.8):
    """Tamano muestral para test de correlacion.

    Usa la transformacion z de Fisher.

    Args:
        r: Correlacion esperada (0.1=pequena, 0.3=mediana, 0.5=grande).
        alpha: Nivel de significancia.
        power: Potencia deseada.

    Returns:
        dict con n_total y parametros usados.
    """
    if abs(r) <= 0 or abs(r) >= 1:
        return {'success': False, 'error': 'La correlacion debe estar entre -1 y 1 (excluidos)'}

    try:
        z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
        z_beta = sp_stats.norm.ppf(power)
        z_r = np.arctanh(r)
        n = math.ceil(((z_alpha + z_beta) / z_r) ** 2 + 3)
        return {
            'success': True,
            'n_per_group': n,
            'n_total': n,
            'r': r,
            'alpha': alpha,
            'power': power,
            'test': 'Correlacion',
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def sample_size_proportions(p1, p2, alpha=0.05, power=0.8):
    """Tamano muestral por grupo para comparar dos proporciones.

    Args:
        p1: Proporcion esperada en grupo 1 (0 < p1 < 1).
        p2: Proporcion esperada en grupo 2 (0 < p2 < 1).
        alpha: Nivel de significancia.
        power: Potencia deseada.

    Returns:
        dict con n_per_group, n_total, y parametros usados.
    """
    if not (0 < p1 < 1) or not (0 < p2 < 1):
        return {'success': False, 'error': 'Las proporciones deben estar entre 0 y 1 (excluidos)'}
    if p1 == p2:
        return {'success': False, 'error': 'Las proporciones deben ser distintas'}

    try:
        effect_size = 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))
        analysis = NormalIndPower()
        n = analysis.solve_power(
            effect_size=abs(effect_size),
            alpha=alpha,
            power=power,
            alternative='two-sided',
        )
        n_per_group = math.ceil(n)
        return {
            'success': True,
            'n_per_group': n_per_group,
            'n_total': n_per_group * 2,
            'p1': p1,
            'p2': p2,
            'effect_size_h': abs(effect_size),
            'alpha': alpha,
            'power': power,
            'test': 'Dos proporciones',
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def sample_size_survival(hazard_ratio, alpha=0.05, power=0.8, ratio=1.0):
    """Numero de eventos necesarios para log-rank test (Schoenfeld).

    Args:
        hazard_ratio: Hazard ratio esperado (HR != 1).
        alpha: Nivel de significancia.
        power: Potencia deseada.
        ratio: Ratio de asignacion (n2/n1).

    Returns:
        dict con n_events y parametros usados.
    """
    if hazard_ratio <= 0 or hazard_ratio == 1.0:
        return {'success': False, 'error': 'El hazard ratio debe ser > 0 y distinto de 1'}
    if ratio <= 0:
        return {'success': False, 'error': 'El ratio de asignacion debe ser > 0'}

    try:
        z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
        z_beta = sp_stats.norm.ppf(power)
        log_hr = math.log(hazard_ratio)
        d = math.ceil(((z_alpha + z_beta) ** 2 * (1 + ratio) ** 2) / (ratio * log_hr ** 2))
        n_group1 = math.ceil(d / (1 + ratio))
        n_group2 = math.ceil(d * ratio / (1 + ratio))
        return {
            'success': True,
            'n_events': d,
            'n_per_group': n_group1,
            'n_total': n_group1 + n_group2,
            'hazard_ratio': hazard_ratio,
            'ratio': ratio,
            'alpha': alpha,
            'power': power,
            'test': 'Log-rank (Schoenfeld)',
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def sensitivity_table(calc_func, effect_sizes, powers=(0.7, 0.8, 0.9), **fixed_params):
    """Genera tabla de sensibilidad: n para distintos effect sizes y potencias.

    Args:
        calc_func: Funcion de calculo (sample_size_ttest_ind, etc.).
        effect_sizes: Lista de tamanos del efecto a evaluar.
        powers: Tupla de potencias a evaluar.
        **fixed_params: Parametros fijos (alpha, k_groups, etc.).

    Returns:
        list[dict] con filas {effect_size, power_70, power_80, power_90, ...}.
    """
    rows = []
    for es in effect_sizes:
        row = {'effect_size': es}
        for pw in powers:
            params = {**fixed_params, 'power': pw}
            # Determinar el nombre del primer argumento segun la funcion
            if calc_func in (sample_size_ttest_ind, sample_size_ttest_paired, sample_size_anova):
                params['effect_size'] = es
            elif calc_func == sample_size_correlation:
                params['r'] = es
            elif calc_func == sample_size_survival:
                params['hazard_ratio'] = es
            result = calc_func(**params)
            key = f"power_{int(pw * 100)}"
            row[key] = result.get('n_per_group', '—') if result.get('success') else '—'
        rows.append(row)
    return rows
