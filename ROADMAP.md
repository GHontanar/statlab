# StatLab — Roadmap

## Estado actual

App refactorizada en paquete `ui/` (11 modulos) con 230 tests.
14 tests estadisticos, figuras interactivas Plotly, informes PDF, calculadora de tamano muestral.
Despliegue: Streamlit Cloud / HuggingFace / LXC Proxmox.

---

## Completado

### Arquitectura y calidad (Roadmap 1)

- **A1.** Separar en modulos (`stats/`, `charts/`, `reports/`, `utils/`) ✔
- **A2.** Tests unitarios (230 tests, ~90% cobertura) ✔
- **A3.** Validacion de entrada ✔

### Bugs y robustez (Roadmaps 1-2)

- **B1.** `except:` desnudos silencian errores ✔
- **B2.** Datos pareados truncados sin aviso ✔
- **B3.** Cohen's d incorrecto para muestras desiguales ✔
- **B4.** `warnings.filterwarnings('ignore')` global ✔
- **B5.** Booleanos inferidos como Continua ✔
- **B6.** Normalidad "N/A" sin explicacion ✔
- **B7.** Evento KM no se pre-valida en UI ✔
- **B8.** Categorias case-sensitive ✔
- **B9.** Truncacion pareada sin alternativa ✔

### Supuestos y confianza (Roadmap 2)

- **S1.** Test de homogeneidad de varianzas (Levene) ✔
- **S2.** Intervalos de confianza (IC 95%) ✔
- **S3.** Seccion de supuestos antes de resultados ✔
- **S4.** Tabla resumen de todos los analisis ✔

### Funcionalidades estadisticas (Roadmaps 1-2)

- **F1.** Kaplan-Meier (survival) ✔
- **F2.** Bland-Altman ✔
- **F3.** Curvas ROC ✔
- **F4.** Informe PDF mejorado (ReportLab) ✔
- **F5.** Exportar figuras en SVG ✔
- **F6.** Multiples comparaciones avanzadas ✔
- **F7.** Regresion logistica ✔
- **F8.** Analisis de potencia / tamano muestral ✔
- **F9.** ICC (coeficiente de correlacion intraclase) ✔

### Figuras (Roadmaps 1-3)

- **G1-G7.** Resolucion, contexto, tamano adaptativo, brackets, puntos, contraste, fuente ✔
- **G8.** Personalizacion de figuras por el usuario ✔
- **PL1-PL7.** Migracion completa a Plotly (boxplot, violin, scatter, Bland-Altman, ROC, KM, barras/histograma/paired) ✔

### UX y paper-friendly (Roadmap 1)

- **P1.** Interpretacion en lenguaje natural ✔
- **P2.** Preguntas en vez de jerga ✔
- **P3.** Etiquetas sin jerga estadistica ✔
- **P4.** Preseleccion automatica del test ✔
- **P5.** Figura automatica con el resultado ✔
- **P6.** Tabla descriptiva visible ✔
- **P7.** Simplificar definicion de variables ✔
- **P8.** Advertencias contextuales ✔
- **U1.** Boton para limpiar historial ✔
- **U2.** Tabla de contingencia visual ✔
- **U3.** Selector de columna ID para datos pareados ✔
- **U4.** Indicador de progreso PDF ✔
- **U5.** Datos de ejemplo sin subir archivo ✔
- Guia de tests en sidebar ✔
- **U6.** Validacion de tamano de archivo ✔
- **U7.** Eliminar resultados individuales del historial ✔
- **U8.** Exportar tabla resumen a Excel ✔

### Calculadora de tamano muestral (Roadmap 3)

- **M1.** Dos grupos independientes (t-test) ✔
- **M2.** Muestras pareadas ✔
- **M3.** ANOVA (3+ grupos) ✔
- **M4.** Correlacion ✔
- **M5.** Comparar dos proporciones ✔
- **M6.** Supervivencia (log-rank) ✔

---

## Pendiente

### Fase 1: Internacionalizacion (i18n) — siguiente

Toggle espanol/ingles en sidebar. Archivo de traducciones centralizado.

- **I1.** Modulo `ui/i18n.py` con diccionario es/en
- **I2.** Traducir UI (labels, tooltips, mensajes)
- **I3.** Traducir interpretaciones (`reports/text.py`)
- **I4.** Traducir guia de tests (sidebar)

### Fase 2: Figuras y reportes

- **G9.** Q-Q plot
- **G10.** Heatmap de correlaciones
- **G11.** Forest plot
- **R1.** Leyendas de figuras en PDF
- **R2.** Metadatos en PDF (autor, titulo, institucion)

### Fase 3: Interpretacion con LLM

Interpretacion contextualizada via API de Claude.

- **L1.** Prompt engineering para interpretacion estadistica
- **L2.** Integracion con API (con API key del usuario)
- **L3.** Boton "Interpretar con IA" en resultados

---

## Orden de implementacion

1. **I1-I4** — Internacionalizacion (siguiente)
2. **G9-G11, R1-R2** — Figuras y reportes
3. **L1-L3** — LLM (cuando el resto este estable)
