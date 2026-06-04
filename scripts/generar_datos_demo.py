"""Genera los datasets de demostración de StatLab (reproducible).

- datos_demo.csv: formato wide (una fila por paciente). Cubre los 15 tests que
  la app lee en formato wide.
- datos_demo_observadores.csv: formato long (medidas repetidas por observador).
  Cubre t-pareado, Wilcoxon e ICC.

Los datos están diseñados para que cada test dé un resultado claro en una demo
en directo. Ejecutar: python scripts/generar_datos_demo.py
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(20260604)

# ----------------------------------------------------------------------------
# DATASET 1: WIDE — ensayo clínico, una fila por paciente
# ----------------------------------------------------------------------------
N_POR_GRUPO = 25
grupos = ["Tratamiento_A", "Tratamiento_B", "Placebo"]
N = N_POR_GRUPO * len(grupos)

grupo = np.repeat(grupos, N_POR_GRUPO)
ids = [f"P{i:03d}" for i in range(1, N + 1)]
sexo = rng.choice(["M", "F"], size=N)
edad = np.round(rng.normal(55, 11, N)).clip(18, 85).astype(int)

# Biomarcador: NORMAL en cada grupo, medias separadas -> ANOVA / t-test claros
medias_bio = {"Tratamiento_A": 62, "Tratamiento_B": 55, "Placebo": 47}
biomarcador = np.array([rng.normal(medias_bio[g], 6) for g in grupo]).round(1)

# Citoquinas: SESGADA (lognormal) -> Shapiro falla -> no paramétrico (Kruskal/MW)
escala_cito = {"Tratamiento_A": 1.05, "Tratamiento_B": 0.85, "Placebo": 0.55}
citoquinas = np.array([rng.lognormal(escala_cito[g], 0.5) for g in grupo]).round(2)

# Correlación / regresión: dosis -> concentración plasmática (lineal fuerte)
dosis_mg = rng.uniform(10, 100, N).round(0)
concentracion = (0.8 * dosis_mg + rng.normal(0, 6, N) + 5).round(1)

# Bland-Altman: glucosa medida por laboratorio vs dispositivo (sesgo pequeño)
glucosa_lab = rng.normal(105, 18, N).round(0)
glucosa_disp = (glucosa_lab + rng.normal(-2.0, 4.0, N)).round(0)  # sesgo ~ -2

# Desenlace binario dependiente del biomarcador -> ROC/logística con señal
logit = 0.18 * (biomarcador - 54)
prob_resp = 1 / (1 + np.exp(-logit))
respuesta = np.where(rng.random(N) < prob_resp, "Respondedor", "No_respondedor")

# Severidad asociada al grupo -> chi² significativo
sev_probs = {
    "Tratamiento_A": [0.6, 0.3, 0.1],
    "Tratamiento_B": [0.4, 0.4, 0.2],
    "Placebo":       [0.15, 0.35, 0.5],
}
severidad = np.array([rng.choice(["Leve", "Moderada", "Grave"], p=sev_probs[g]) for g in grupo])

# Fumador asociado al desenlace -> Fisher 2x2 con señal
fumador = np.where(
    (respuesta == "No_respondedor") & (rng.random(N) < 0.55), "Si",
    np.where(rng.random(N) < 0.20, "Si", "No"))

# Supervivencia: hazard mayor en Placebo -> log-rank significativo
hazard = {"Tratamiento_A": 0.025, "Tratamiento_B": 0.04, "Placebo": 0.085}
t_evento = np.array([rng.exponential(1 / hazard[g]) for g in grupo])
TOPE = 36.0
evento = (t_evento <= TOPE).astype(int)
tiempo_meses = np.minimum(t_evento, TOPE).round(1)

wide = pd.DataFrame({
    "ID": ids,
    "Grupo": grupo,
    "Sexo": sexo,
    "Edad": edad,
    "Biomarcador": biomarcador,
    "Citoquinas": citoquinas,
    "Dosis_mg": dosis_mg,
    "Concentracion_plasma": concentracion,
    "Glucosa_lab": glucosa_lab,
    "Glucosa_dispositivo": glucosa_disp,
    "Respuesta": respuesta,
    "Severidad": severidad,
    "Fumador": fumador,
    "Tiempo_meses": tiempo_meses,
    "Evento": evento,
})
wide.to_csv("datos_demo.csv", index=False)

# ----------------------------------------------------------------------------
# DATASET 2: LONG — concordancia entre observadores (pareados + ICC)
# ----------------------------------------------------------------------------
N_PAC = 30
observadores = ["Obs_1", "Obs_2", "Obs_3"]
# Valor "verdadero" de cada lesión (variabilidad entre pacientes alta -> ICC alto)
valor_real = rng.normal(22, 6, N_PAC)
# Sesgo sistemático por observador (Obs_2 mide algo más alto -> pareado signif.)
sesgo_obs = {"Obs_1": 0.0, "Obs_2": 1.6, "Obs_3": 0.3}
ruido = 1.2  # ruido pequeño -> buena concordancia

filas = []
for j in range(N_PAC):
    pac = f"L{j+1:02d}"
    for obs in observadores:
        med = valor_real[j] + sesgo_obs[obs] + rng.normal(0, ruido)
        filas.append({"Paciente": pac, "Observador": obs, "Medicion_mm": round(med, 1)})
# Ordenado por paciente y observador: al filtrar por observador, los pacientes
# quedan en el mismo orden (requisito del cálculo de ICC por apilado).
long = pd.DataFrame(filas).sort_values(["Paciente", "Observador"]).reset_index(drop=True)
long.to_csv("datos_demo_observadores.csv", index=False)

print("Generado datos_demo.csv:", wide.shape)
print("Generado datos_demo_observadores.csv:", long.shape)
