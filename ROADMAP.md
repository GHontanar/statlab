# StatLab — Roadmap

## Estado actual

MVP funcional basado en Streamlit. Archivo monolitico (`app.py`, ~870 lineas) que cubre carga de datos, tests estadisticos, graficos y exportacion de informes.

---

## Bugs a corregir

### B1. `except:` desnudos silencian errores
- **Donde:** `app.py:190, 204` (post-hoc Tukey y Dunn)
- **Impacto:** Si el post-hoc falla, el usuario nunca lo sabe
- **Fix:** Capturar excepciones especificas, mostrar warning en UI

### B2. Datos pareados truncados sin aviso
- **Donde:** `app.py:146-147, 154` (t-test pareado y Wilcoxon)
- **Impacto:** Si los grupos tienen distinto n, se descartan datos silenciosamente
- **Fix:** Validar que n sea igual, advertir al usuario si no lo es

### B3. Cohen's d incorrecto para muestras desiguales
- **Donde:** `app.py:162`
- **Impacto:** Usa `(s1^2 + s2^2) / 2` en vez del pooled std ponderado por n
- **Fix:** Usar formula correcta: `sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2))`

### B4. `warnings.filterwarnings('ignore')` global
- **Donde:** `app.py:18`
- **Impacto:** Silencia advertencias legitimas de scipy/numpy
- **Fix:** Eliminar o limitar a warnings cosmeticos especificos

---

## Mejoras de arquitectura

### A1. Separar en modulos
El archivo unico viola el principio de responsabilidad unica y dificulta testing.

Estructura propuesta:
```
statlab/
  app.py              # Solo UI Streamlit (orquestacion)
  stats/
    tests.py           # run_test, check_normality, suggest_test
    effect_size.py     # Cohen's d, eta squared, R squared
  charts/
    figures.py         # generate_figure y tipos de grafico
  reports/
    pdf.py             # generate_pdf_report
    text.py            # format_result_text
  utils/
    data.py            # infer_variable_type, carga y validacion
  tests/
    test_stats.py
    test_charts.py
    test_utils.py
```

### A2. Tests unitarios
- Prioridad: `run_test`, `suggest_test`, `check_normality`, `infer_variable_type`
- Framework: pytest
- Cobertura objetivo: 70%

### A3. Validacion de entrada
- Verificar que columnas marcadas como "Continua" sean realmente numericas
- Validar n minimo por grupo antes de ejecutar tests
- Mostrar errores descriptivos en vez de tracebacks

---

## Mejoras de UX

### U1. Boton para limpiar historial de analisis
### U2. Mostrar tabla de contingencia visual para chi-cuadrado/Fisher
### U3. Selector de columna ID para datos pareados (no asumir orden por posicion)
### U4. Indicador de progreso al generar PDF con muchos analisis
### U5. Opcion de cargar datos de ejemplo sin subir archivo

---

## Calidad de figuras

### G1. Resolucion y formato de exportacion
- `st.pyplot()` renderiza a 72-100 dpi en pantalla (no se configura `fig.dpi` al crear)
- Solo se ofrece PNG para descarga; falta SVG y PDF individual
- **Fix:** Crear figuras con `dpi=150` minimo, ofrecer descarga en PNG (300 dpi), SVG y PDF

### G2. Contexto de seaborn demasiado pequeno
- `sns.set_context("paper")` es el mas pequeno; labels y ticks quedan diminutos
- **Fix:** Usar `"notebook"` o `"talk"` segun destino (pantalla vs publicacion)

### G3. Tamano de figura fijo
- `figsize=(8, 6)` para todo; no se adapta al tipo de grafico ni al numero de grupos
- **Fix:** Ajustar figsize segun tipo (scatter mas cuadrado, barras mas ancho con muchos grupos)

### G4. Sin brackets de significancia
- Solo un texto flotante con p-valor en esquina
- El estandar en publicaciones son barras horizontales entre grupos con asteriscos (`*`, `**`, `***`)
- **Fix:** Implementar brackets automaticos con statannotations o logica propia

### G5. Puntos individuales poco visibles
- Box plot: `stripplot` con `alpha=0.4` y `size=4` — quedan tenues
- Bar plot: no tiene puntos individuales superpuestos (estandar actual en publicaciones)
- **Fix:** Aumentar `alpha=0.6`, `size=5`; agregar stripplot a barras

### G6. Contraste y visibilidad
- Scatter: `edgecolors='white'` sobre fondo blanco pierde contraste
- Paired plot: lineas con `alpha=0.4` demasiado tenues, sin medias por grupo
- **Fix:** Bordes oscuros en scatter, alpha mayor en paired, agregar marcadores de media

### G7. Fuente generica
- Fuente por defecto de matplotlib (DejaVu Sans) — funcional pero no profesional
- **Fix:** Usar Arial/Helvetica (estandar en publicaciones biomedicas) con fallback

### G8. Sin personalizacion por el usuario
- No se pueden cambiar titulos, labels, colores ni tamano de fuente
- **Fix:** Panel lateral con opciones basicas de personalizacion (titulo, xlabel, ylabel, paleta)

---

## Features del README original pendientes

### F1. Kaplan-Meier (survival)
### F2. Bland-Altman
### F3. Curvas ROC
### F4. Informe PDF mejorado (reportlab en vez de matplotlib text)
### F5. Exportar figuras en SVG
### F6. Multiples comparaciones avanzadas

---

## Despliegue

- **Streamlit Community Cloud** — opcion mas simple para acceso web publico
- **LXC Proxmox + systemd** — ya documentado en README, ideal para self-hosted
- **No es viable como app estatica** — requiere backend Python (scipy, pandas, numpy)

---

## Prioridades sugeridas

1. Corregir bugs (B1-B4)
2. Calidad de figuras (G1-G8)
3. Separar en modulos (A1)
4. Agregar tests unitarios (A2)
5. Validacion de entrada (A3)
6. Mejoras de UX (U1-U5)
7. Features nuevas (F1-F6)
