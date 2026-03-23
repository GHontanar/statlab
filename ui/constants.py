"""Constantes y dataclass AnalysisConfig para la capa UI."""

from dataclasses import dataclass, field
from typing import Optional

Q_DIFF_GROUPS = "¿Hay diferencia entre grupos?"
Q_CORRELATION = "¿Están relacionadas dos mediciones?"
Q_ASSOCIATION = "¿Se asocian dos categorías?"
Q_AGREEMENT = "¿Dos métodos miden lo mismo?"
Q_PREDICTION = "¿Un valor predice un desenlace?"
Q_SURVIVAL = "¿Cuánto tiempo hasta un evento?"
Q_RISK = "¿Un factor aumenta el riesgo?"
Q_RELIABILITY = "¿Las mediciones son reproducibles?"

ANALYSIS_TYPES = [
    Q_DIFF_GROUPS, Q_CORRELATION, Q_ASSOCIATION,
    Q_AGREEMENT, Q_PREDICTION, Q_SURVIVAL,
    Q_RISK, Q_RELIABILITY,
]

AUTO_FIGURE_MAP = {
    Q_DIFF_GROUPS: "boxplot",
    Q_CORRELATION: "scatter",
    Q_ASSOCIATION: None,
    Q_AGREEMENT: "bland_altman",
    Q_PREDICTION: "roc",
    Q_SURVIVAL: "kaplan_meier",
    Q_RISK: None,
    Q_RELIABILITY: None,
}

AVAILABLE_FIGURES = {
    Q_DIFF_GROUPS: {
        "Box plot + puntos": "boxplot",
        "Violin plot": "violin",
        "Barras + error (SEM)": "bar_error",
        "Datos pareados": "paired",
        "Histograma por grupo": "histogram",
    },
    Q_CORRELATION: {
        "Scatter + regresión": "scatter",
        "Histograma": "histogram",
    },
    Q_AGREEMENT: {
        "Bland-Altman": "bland_altman",
    },
    Q_PREDICTION: {
        "Curva ROC": "roc",
    },
    Q_SURVIVAL: {
        "Kaplan-Meier": "kaplan_meier",
    },
}

DEFAULT_FIGURES = {"Histograma": "histogram"}


@dataclass
class AnalysisConfig:
    analysis_type: str
    var_dep: str
    var_group: str
    selected_test_id: str
    selected_groups: Optional[list] = None
    alpha: float = 0.05
    paired_id_col: Optional[str] = None
    extra: Optional[dict] = field(default_factory=dict)
