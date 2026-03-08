# StatLab — Roadmap

## Estado actual

App modular basada en Streamlit para analisis estadistico y generacion de figuras.
Estructura en capas: `stats/`, `charts/`, `reports/`, `utils/`, con 153 tests y ~90% de cobertura.

---

## Completado

### B1. ~~`except:` desnudos silencian errores~~ ✔
### B2. ~~Datos pareados truncados sin aviso~~ ✔
### B3. ~~Cohen's d incorrecto para muestras desiguales~~ ✔
### B4. ~~`warnings.filterwarnings('ignore')` global~~ ✔
### A1. ~~Separar en modulos~~ ✔
### A2. ~~Tests unitarios~~ ✔
153 tests con pytest. Cobertura ~90%.
### A3. ~~Validacion de entrada~~ ✔
validate_continuous (tipo numerico, NaN), validate_group_sizes (n minimo por grupo).
### G1. ~~Resolucion y formato de exportacion~~ ✔
### G2. ~~Contexto de seaborn~~ ✔
### G3. ~~Tamano de figura adaptativo~~ ✔
### G4. ~~Brackets de significancia~~ ✔
### G5. ~~Puntos individuales visibles~~ ✔
### G6. ~~Contraste y visibilidad~~ ✔
### G7. ~~Fuente profesional~~ ✔
### F5. ~~Exportar figuras en SVG~~ ✔
### U1. ~~Boton para limpiar historial~~ ✔
### U2. ~~Tabla de contingencia visual~~ ✔
### U3. ~~Selector de columna ID para datos pareados~~ ✔
### U4. ~~Indicador de progreso PDF~~ ✔
### U5. ~~Datos de ejemplo sin subir archivo~~ ✔
### ~~Guia de tests en sidebar~~ ✔

---

### G8. ~~Personalizacion de figuras por el usuario~~ ✔
Panel con opciones: titulo, xlabel, ylabel, paleta de colores.
### F1. ~~Kaplan-Meier (survival)~~ ✔
Curvas de supervivencia, mediana, comparacion de grupos con log-rank test.
### F2. ~~Bland-Altman~~ ✔
Concordancia entre metodos: sesgo, limites de acuerdo, test de sesgo.
### F3. ~~Curvas ROC~~ ✔
AUC, corte optimo (Youden's J), sensibilidad/especificidad.

---

## Completado (reciente)

### P1. ~~Interpretacion en lenguaje natural~~ ✔
Parrafo auto-generado listo para copiar en Resultados de un paper. Cubre todos los tests.
### P2. ~~Preguntas en vez de jerga~~ ✔
Selector reformulado como preguntas de investigacion.
### P3. ~~Etiquetas sin jerga estadistica~~ ✔
"Que mediste?", "Como se dividen los sujetos?", tooltips en widgets clave.
### P4. ~~Preseleccion automatica del test~~ ✔
Test recomendado con explicacion; alternativas en expander "Cambiar test (avanzado)".
### P5. ~~Figura automatica con el resultado~~ ✔
Auto-genera la figura mas apropiada junto a los resultados.
### P6. ~~Tabla descriptiva visible~~ ✔
Tabla con n, media, DE, mediana por grupo antes del p-valor.
### P7. ~~Simplificar definicion de variables~~ ✔
Solo selectbox para ambiguas; claras en resumen; expander para corregir.
### P8. ~~Advertencias contextuales~~ ✔
Warnings para n < 10, normalidad con n < 20, grupos desbalanceados.

---

### F4. ~~Informe PDF mejorado~~ ✔
Reportlab en vez de matplotlib text. Portada, tablas descriptivas, tablas de resultados, interpretacion, figuras embebidas.
### F6. ~~Multiples comparaciones avanzadas~~ ✔
ANOVA: Tukey HSD, Scheffe, Bonferroni (t-test), Holm (t-test).
Kruskal: Dunn (Bonferroni/Holm/BH), Conover (Bonferroni).
Selector en UI con explicacion de cada metodo.

---

## Despliegue

- **Streamlit Community Cloud** — opcion mas simple para acceso web publico
- **LXC Proxmox + systemd** — ya documentado en README, ideal para self-hosted
- **No es viable como app estatica** — requiere backend Python (scipy, pandas, numpy)

---

## Prioridades sugeridas

1. Interpretacion en lenguaje natural + copiar para paper (P1)
2. Preguntas y etiquetas sin jerga (P2 + P3)
3. Preseleccion inteligente del test (P4)
4. Tabla descriptiva visible (P6)
5. Figura automatica (P5)
6. Simplificar definicion de variables (P7)
7. Advertencias contextuales (P8)
~~8. Informe PDF mejorado (F4)~~ ✔
~~9. Multiples comparaciones avanzadas (F6)~~ ✔

**Roadmap completo. Todos los items implementados.**
