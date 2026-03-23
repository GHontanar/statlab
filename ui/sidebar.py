"""Guía de tests estadísticos y personalización de figuras (sidebar)."""

import streamlit as st


def render_sidebar():
    """Renderiza sidebar con guía de tests y opciones de figuras.

    Returns:
        dict: Opciones de personalización de figuras (title, xlabel, ylabel, palette).
    """
    with st.sidebar:
        st.header("Guía de tests")
        st.caption("Referencia rápida para elegir el test adecuado")

        with st.expander("Comparar 2 grupos"):
            st.markdown("""
**Datos independientes** (sujetos distintos en cada grupo)

| Condición | Test |
|-----------|------|
| Datos normales, varianzas iguales | T-test independiente |
| Datos normales, varianzas distintas | T-test de Welch |
| Datos no normales u ordinales | Mann-Whitney U |

*Ejemplo: Comparar hemoglobina entre grupo tratamiento y control.*

**Datos pareados** (mismo sujeto medido 2 veces)

| Condición | Test |
|-----------|------|
| Datos normales | T-test pareado |
| Datos no normales | Wilcoxon signed-rank |

*Ejemplo: Presión arterial antes y después de un fármaco.*
""")

        with st.expander("Comparar 3+ grupos"):
            st.markdown("""
| Condición | Test | Post-hoc |
|-----------|------|----------|
| Datos normales | ANOVA one-way | Tukey HSD |
| Datos no normales | Kruskal-Wallis | Dunn (Bonferroni) |

*Ejemplo: Comparar eficacia de 3 fármacos distintos.*

El post-hoc solo se ejecuta si el test principal es significativo
(p < alpha). Indica **qué pares** de grupos difieren entre sí.
""")

        with st.expander("Correlación y regresión"):
            st.markdown("""
| Condición | Test |
|-----------|------|
| Relación lineal, datos normales | Pearson |
| Relación monótona, datos no normales | Spearman |
| Predecir Y a partir de X | Regresión lineal |

- **r** cercano a +1/-1 = correlación fuerte
- **R²** = proporción de varianza explicada
- **p-valor** = probabilidad de obtener esa r por azar

*Ejemplo: Correlación entre IMC y colesterol.*
""")

        with st.expander("Variables categóricas"):
            st.markdown("""
| Condición | Test |
|-----------|------|
| Tabla >2×2, o n grande | Chi-cuadrado |
| Tabla 2×2 con n pequeño (<20) | Fisher exacto |

Comparan si la distribución de una variable categórica
es independiente de otra.

*Ejemplo: Asociación entre tratamiento (A/B) y desenlace
(mejora/no mejora).*
""")

        with st.expander("Normalidad"):
            st.markdown("""
StatLab usa el **test de Shapiro-Wilk** automáticamente.

- **p > 0.05**: No se rechaza normalidad → test paramétrico
- **p < 0.05**: Se rechaza normalidad → test no paramétrico

Funciona bien con n entre 3 y 5000.
Con n muy grande, casi todo resulta "no normal" por exceso de poder.
""")

        with st.expander("Bland-Altman"):
            st.markdown("""
Evalúa **concordancia** entre dos métodos de medición.

- **Sesgo (bias)**: diferencia media entre métodos
- **Límites de acuerdo**: sesgo ± 1.96 DE
- Si el sesgo no es significativo y los límites son clínicamente aceptables,
  los métodos son intercambiables.

*Ejemplo: Comparar medición de glucosa con dos dispositivos distintos.*
""")

        with st.expander("Curva ROC"):
            st.markdown("""
Evalúa la **capacidad discriminativa** de un predictor continuo
para un desenlace binario.

| Métrica | Descripción |
|---------|------------|
| AUC | Área bajo la curva (0.5=azar, 1.0=perfecto) |
| Sensibilidad | Verdaderos positivos correctamente identificados |
| Especificidad | Verdaderos negativos correctamente identificados |
| Corte óptimo | Threshold que maximiza Youden's J |

*Ejemplo: Evaluar si un biomarcador predice enfermedad (sí/no).*
""")

        with st.expander("Kaplan-Meier"):
            st.markdown("""
**Análisis de supervivencia**: estima la probabilidad de que un
evento no haya ocurrido en función del tiempo.

- Necesitas: **tiempo** hasta evento y **estado** del evento (0=censurado, 1=evento)
- Opcional: variable de grupo para comparar curvas
- **Log-rank test**: compara curvas entre 2 grupos

*Ejemplo: Tiempo hasta recaída en pacientes con dos tratamientos.*
""")

        with st.expander("Regresión logística"):
            st.markdown("""
Evalúa si un **factor continuo** aumenta o disminuye el riesgo
de un **desenlace binario**.

| Métrica | Descripción |
|---------|------------|
| OR (Odds Ratio) | >1 = más riesgo, <1 = menos riesgo, =1 sin efecto |
| IC 95% del OR | Si incluye 1, el efecto no es significativo |
| Pseudo R² | Proporción de varianza explicada (McFadden) |

*Ejemplo: ¿La edad aumenta el riesgo de diabetes (sí/no)?*
""")

        with st.expander("ICC (fiabilidad)"):
            st.markdown("""
El **coeficiente de correlación intraclase** mide la concordancia
entre evaluadores o métodos de medición repetidos.

| ICC | Interpretación |
|-----|---------------|
| < 0.50 | Pobre |
| 0.50 - 0.75 | Moderada |
| 0.75 - 0.90 | Buena |
| > 0.90 | Excelente |

Datos deben estar **balanceados**: mismo n de sujetos por evaluador.

*Ejemplo: Dos radiólogos miden el tamaño de un tumor en 20 pacientes.*
""")

        with st.expander("Tamaño del efecto"):
            st.markdown("""
El p-valor indica si hay diferencia, pero no su magnitud.

| Métrica | Pequeño | Mediano | Grande |
|---------|---------|---------|--------|
| d de Cohen | <0.5 | 0.5-0.8 | >0.8 |
| η² (ANOVA) | <0.06 | 0.06-0.14 | >0.14 |
| R² | <0.09 | 0.09-0.25 | >0.25 |

*Un p-valor muy bajo con efecto pequeño puede no ser
clínicamente relevante.*
""")

        st.markdown("---")
        st.header("Personalizar figuras")

        custom_title = st.text_input("Título", value="", placeholder="Auto")
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

    return {
        'title': custom_title or None,
        'xlabel': custom_xlabel or None,
        'ylabel': custom_ylabel or None,
        'palette': custom_palette,
    }
