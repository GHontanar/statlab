"""Constantes de dominio de la capa base.

Vive en utils/ para que pueda ser importada por utils, stats y ui sin violar
la direccion de dependencias (ui/constants.py no sirve: lo importarian capas
inferiores). Centraliza las etiquetas de tipo de variable y evita magic strings
y la inconsistencia de tildes que existia entre la capa logica y la UI.
"""

# Etiquetas canonicas del tipo de variable (con tilde, ortografia correcta).
TIPO_NUMERICA = 'Numérica'
TIPO_CATEGORICA = 'Categórica'

# Umbral de cardinalidad: una variable numerica con <= este numero de valores
# unicos se infiere como categorica (p. ej. codigos 1/2/3, escalas).
UMBRAL_CARDINALIDAD_CATEGORICA = 10
