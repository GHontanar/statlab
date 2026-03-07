# StatLab — Roadmap

## Estado actual

App modular basada en Streamlit para analisis estadistico y generacion de figuras.
Estructura en capas: `stats/`, `charts/`, `reports/`, `utils/`, con 98 tests y ~90% de cobertura.

---

## Completado

### B1. ~~`except:` desnudos silencian errores~~ ✔
### B2. ~~Datos pareados truncados sin aviso~~ ✔
### B3. ~~Cohen's d incorrecto para muestras desiguales~~ ✔
### B4. ~~`warnings.filterwarnings('ignore')` global~~ ✔
### A1. ~~Separar en modulos~~ ✔
### A2. ~~Tests unitarios~~ ✔
98 tests con pytest. Cobertura ~90%.
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

## Pendiente

### G8. Personalizacion de figuras por el usuario
- Panel con opciones: titulo, xlabel, ylabel, paleta de colores

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

1. Personalizacion de figuras (G8)
2. Features nuevas (F1-F4, F6)
