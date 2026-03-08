# StatLab — Roadmap 2

## Estado actual

App completa con 11 tests estadisticos, 9 tipos de figura, informes PDF, 153 tests, ~90% cobertura.
Roadmap 1 completado al 100%.

---

## Prioridades

### Fase 1: Supuestos y confianza en resultados

#### S1. Test de homogeneidad de varianzas (Levene)
Mostrar junto a normalidad. Ajustar recomendacion automatica: varianzas iguales → t-test, distintas → Welch.

#### S2. Intervalos de confianza (IC 95%)
Incluir IC en resultados, interpretacion, tablas descriptivas y PDF. Obligatorio para publicacion.

#### S3. Seccion de supuestos antes de resultados
Tabla resumen: normalidad, homogeneidad, n minimo, balance de grupos. Visible antes del p-valor.

#### S4. Tabla resumen de todos los analisis
Dataframe comparativo con test, variables, p-valor, effect size, significancia de todos los analisis ejecutados.

---

### Fase 2: Fixes y robustez

#### B5. Booleanos inferidos como Continua
`utils/data.py` no detecta dtype `bool`. Agregar check.

#### B6. Normalidad "N/A" sin explicacion
Cuando Shapiro devuelve None (n < 3 o n > 5000), mostrar razon en la UI.

#### B7. Evento KM no se pre-valida en UI
Filtrar selectbox de evento a variables binarias (0/1). No esperar al error en ejecucion.

#### B8. Categorias case-sensitive
Detectar y advertir cuando existen categorias que solo difieren en mayusculas/minusculas.

#### B9. Truncacion pareada sin alternativa
Con columna ID seleccionada, emparejar por ID y excluir no emparejados en vez de truncar por posicion.

---

### Fase 3: Funcionalidades estadisticas

#### F7. Regresion logistica
Odds ratios, IC, p-valor por predictor. Comun en biomedicina para factores de riesgo.

#### F8. Analisis de potencia / tamano muestral
Potencia post-hoc del analisis ejecutado. Estimador de n necesario para detectar el efecto observado.

#### F9. ICC (coeficiente de correlacion intraclase)
Para estudios de fiabilidad inter/intra-observador. Complementa Bland-Altman.

---

### Fase 4: Figuras y reportes

#### G9. Q-Q plot
Verificacion visual de normalidad. Complemento del test de Shapiro-Wilk.

#### G10. Heatmap de correlaciones
Matriz de correlaciones para exploracion con multiples variables continuas.

#### G11. Forest plot
Visualizacion de effect sizes con IC de multiples comparaciones.

#### R1. Leyendas de figuras en PDF
Caption descriptivo debajo de cada figura embebida en el informe.

#### R2. Metadatos en PDF
Campos opcionales: autor, titulo del estudio, institucion. Aparecen en portada.

---

### Fase 5: UX y polish

#### U6. Validacion de tamano de archivo
Limite configurable (default 100 MB). Mensaje claro si se excede.

#### U7. Eliminar resultados individuales del historial
Boton por fila en vez de solo "limpiar todo".

#### U8. Exportar tabla resumen a Excel
Descargar la tabla resumen (S4) como .xlsx.

---

## Orden de implementacion sugerido

1. S1 — Levene (base para recomendacion automatica)
2. S2 — Intervalos de confianza
3. S3 — Seccion de supuestos
4. S4 — Tabla resumen
5. B5-B9 — Fixes
6. F7-F9 — Nuevas funcionalidades estadisticas
7. G9-G11, R1-R2 — Figuras y reportes
8. U6-U8 — UX y polish
