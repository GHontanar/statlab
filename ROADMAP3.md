# StatLab — Roadmap 3

## Estado actual

App refactorizada en paquete `ui/` con 230 tests. Roadmaps 1 y 2 completados.
Decision: herramienta de analisis (no de aprendizaje). Despliegue en Streamlit Cloud/HuggingFace.
Fase 1 (calculadora) y Fase 2 (Plotly) completadas.

---

## Fase 1: Calculadora de tamano muestral ✅

Calcular n necesario ANTES de recolectar datos. Alto valor para comites de etica.

### M1. Dos grupos independientes (t-test) ✅
### M2. Muestras pareadas ✅
### M3. ANOVA (3+ grupos) ✅
### M4. Correlacion ✅
### M5. Comparar dos proporciones ✅
### M6. Supervivencia (log-rank) ✅

---

## Fase 2: Migracion a Plotly ✅

Figuras interactivas con Plotly (zoom, hover, pan). Export PNG/SVG via kaleido.

### PL1. Boxplot + puntos ✅
### PL2. Violin plot ✅
### PL3. Scatter + regresion ✅
### PL4. Bland-Altman ✅
### PL5. Curva ROC ✅
### PL6. Kaplan-Meier ✅
### PL7. Barras + error, histograma, paired ✅

---

## Fase 3: Internacionalizacion (i18n)

Toggle espanol/ingles en sidebar. Archivo de traducciones centralizado.

### I1. Modulo ui/i18n.py con diccionario es/en
### I2. Traducir UI (labels, tooltips, mensajes)
### I3. Traducir interpretaciones (reports/text.py)
### I4. Traducir guia de tests (sidebar)

---

## Fase 4: Interpretacion con LLM

Interpretacion contextualizada via API de Claude.

### L1. Prompt engineering para interpretacion estadistica
### L2. Integracion con API (con API key del usuario)
### L3. Boton "Interpretar con IA" en resultados

---

## Fase 5: Pendientes de Roadmap 2

### G9. Q-Q plot
### G10. Heatmap de correlaciones
### G11. Forest plot
### R1. Leyendas de figuras en PDF
### R2. Metadatos en PDF (autor, titulo, institucion)
### U6. Validacion de tamano de archivo
### U7. Eliminar resultados individuales del historial
### U8. Exportar tabla resumen a Excel

---

## Orden de implementacion

1. ~~M1-M6 — Calculadora de tamano muestral~~ ✅
2. ~~PL1-PL7 — Migracion a Plotly~~ ✅
3. **I1-I4 — Internacionalizacion** (siguiente)
4. G9-G11, R1-R2, U6-U8 — Pendientes
5. L1-L3 — LLM (cuando el resto este estable)
