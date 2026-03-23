"""Página de bienvenida cuando no hay datos cargados."""

import streamlit as st


def render_landing():
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
    **Concordancia:** Bland-Altman
    **Discriminación:** Curva ROC (AUC, corte óptimo)
    **Supervivencia:** Kaplan-Meier, log-rank
    """)
