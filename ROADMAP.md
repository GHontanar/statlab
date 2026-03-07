# StatLab — Roadmap

## Estado actual

App modular basada en Streamlit para analisis estadistico y generacion de figuras.
Estructura en capas: `stats/`, `charts/`, `reports/`, `utils/`, con 86 tests y 90% de cobertura.

---

## Completado

### B1. ~~`except:` desnudos silencian errores~~ ✔
### B2. ~~Datos pareados truncados sin aviso~~ ✔
### B3. ~~Cohen's d incorrecto para muestras desiguales~~ ✔
### B4. ~~`warnings.filterwarnings('ignore')` global~~ ✔
### A1. ~~Separar en modulos~~ ✔
Estructura actual:
```
app.py              (260 lineas — solo UI)
stats/tests.py      (255 lineas — logica estadistica)
charts/figures.py   (185 lineas — graficos)
reports/text.py     (84 lineas — formateo texto)
reports/pdf.py      (49 lineas — generacion PDF)
utils/data.py       (10 lineas — inferencia de tipos)
```
### A2. ~~Tests unitarios~~ ✔
86 tests con pytest. Cobertura: stats 96%, charts 100%, reports/text 94%, utils 100%. Total 90%.
### G1. ~~Resolucion y formato de exportacion~~ ✔
Figuras a 150 dpi en pantalla, descarga en PNG 300 dpi + SVG.
### G2. ~~Contexto de seaborn demasiado pequeno~~ ✔
Cambiado de `"paper"` a `"notebook"`.
### G3. ~~Tamano de figura fijo~~ ✔
figsize adaptativo por tipo de grafico y numero de grupos.
### G4. ~~Sin brackets de significancia~~ ✔
Brackets con asteriscos para 2 grupos, texto flotante para >2 grupos.
### G5. ~~Puntos individuales poco visibles~~ ✔
Mayor alpha/size en stripplot, puntos individuales en bar chart.
### G6. ~~Contraste y visibilidad~~ ✔
Bordes oscuros en scatter, lineas paired mas gruesas, marcador de media.
### G7. ~~Fuente generica~~ ✔
Arial/Helvetica con fallback a DejaVu Sans.
### F5. ~~Exportar figuras en SVG~~ ✔
Incluido en G1.

---

## Pendiente

### A3. Validacion de entrada
- Verificar que columnas marcadas como "Continua" sean realmente numericas
- Validar n minimo por grupo antes de ejecutar tests
- Mostrar errores descriptivos en vez de tracebacks

### G8. Personalizacion de figuras por el usuario
- Panel lateral con opciones: titulo, xlabel, ylabel, paleta de colores

---

### Mejoras de UX

#### U1. Boton para limpiar historial de analisis
#### U2. Mostrar tabla de contingencia visual para chi-cuadrado/Fisher
#### U3. Selector de columna ID para datos pareados (no asumir orden por posicion)
#### U4. Indicador de progreso al generar PDF con muchos analisis
#### U5. Opcion de cargar datos de ejemplo sin subir archivo

---

### Features nuevas

#### F1. Kaplan-Meier (survival)
#### F2. Bland-Altman
#### F3. Curvas ROC
#### F4. Informe PDF mejorado (reportlab en vez de matplotlib text)
#### F6. Multiples comparaciones avanzadas

---

## Despliegue

- **Streamlit Community Cloud** — opcion mas simple para acceso web publico
- **LXC Proxmox + systemd** — ya documentado en README, ideal para self-hosted
- **No es viable como app estatica** — requiere backend Python (scipy, pandas, numpy)

---

## Prioridades sugeridas

1. Validacion de entrada (A3)
2. Personalizacion de figuras (G8)
3. Mejoras de UX (U1-U5)
4. Features nuevas (F1-F4, F6)
