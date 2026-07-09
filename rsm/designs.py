"""
designs.py
==========

Generacion de disenos experimentales para RSM:

  - Diseno Central Compuesto (CCD)  -> central_composite_design()
  - Diseno Box-Behnken (BBD)        -> box_behnken_design()

Todos los disenos se devuelven en un DataFrame de pandas con las variables
codificadas (rango tipico [-alpha, +alpha]) y, si se entregan los rangos
naturales, tambien en unidades reales de ingenieria.

Nomenclatura de codificacion
----------------------------
Para un factor natural  X  con centro  X0  y semi-rango  dX  (= (Xmax-Xmin)/2):

        x = (X - X0) / dX          (codificado)
        X = X0 + x * dX            (natural)

Referencias:
  Montgomery, D. C. (2017). Design and Analysis of Experiments, 8a ed. Wiley.
  Myers, Montgomery & Anderson-Cook (2016). Response Surface Methodology, 4a ed.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Contenedor de la definicion de factores
# ---------------------------------------------------------------------------
@dataclass
class Factor:
    """Definicion de un factor experimental."""
    name: str
    low: float          # nivel natural en x = -1
    high: float         # nivel natural en x = +1
    units: str = ""

    @property
    def center(self) -> float:
        return (self.low + self.high) / 2.0

    @property
    def half_range(self) -> float:
        return (self.high - self.low) / 2.0

    def to_natural(self, coded: np.ndarray) -> np.ndarray:
        return self.center + np.asarray(coded, dtype=float) * self.half_range

    def to_coded(self, natural: np.ndarray) -> np.ndarray:
        return (np.asarray(natural, dtype=float) - self.center) / self.half_range


@dataclass
class DesignResult:
    """Resultado de la generacion de un diseno."""
    coded: pd.DataFrame                 # matriz codificada
    natural: pd.DataFrame               # matriz en unidades naturales
    factors: list[Factor]
    kind: str                           # 'CCD' o 'BBD'
    meta: dict = field(default_factory=dict)

    @property
    def n_runs(self) -> int:
        return len(self.coded)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _full_factorial_2level(k: int) -> np.ndarray:
    """Matriz factorial completa 2^k con niveles {-1, +1}."""
    return np.array(list(itertools.product([-1.0, 1.0], repeat=k)))


def ccd_alpha(k: int, mode: str = "rotatable", n_factorial: int | None = None,
              n_center: int = 4) -> float:
    """
    Valor de alpha (distancia axial) para un CCD.

      - 'rotatable' : alpha = (n_f)^(1/4)  -> varianza de prediccion constante
                      a distancia fija del centro.
      - 'face'      : alpha = 1            -> diseno de caras centradas (CCF).
      - 'orthogonal': alpha que ortogonaliza los terminos cuadraticos.
    """
    nf = n_factorial if n_factorial is not None else 2 ** k
    if mode == "face":
        return 1.0
    if mode == "rotatable":
        return nf ** 0.25
    if mode == "orthogonal":
        na = 2 * k
        n = nf + na + n_center
        return ((nf * (np.sqrt(n) - np.sqrt(nf)) ** 2) / 4.0) ** 0.25
    raise ValueError(f"Modo de alpha desconocido: {mode!r}")


# ---------------------------------------------------------------------------
# Diseno Central Compuesto (CCD)
# ---------------------------------------------------------------------------
def central_composite_design(factors: Sequence[Factor],
                             n_center: int = 4,
                             alpha: str | float = "rotatable") -> DesignResult:
    """
    Genera un Diseno Central Compuesto (CCD).

    Componentes:
      - Porcion factorial  : 2^k puntos en {-1, +1}
      - Porcion axial      : 2k puntos a distancia +-alpha sobre cada eje
      - Puntos centrales   : n_center replicas en el origen

    Parametros
    ----------
    factors   : lista de Factor (k factores).
    n_center  : numero de replicas al centro (estima error puro).
    alpha     : 'rotatable' | 'face' | 'orthogonal' | valor numerico.
    """
    k = len(factors)
    if k < 2:
        raise ValueError("El CCD requiere al menos 2 factores.")

    factorial = _full_factorial_2level(k)

    if isinstance(alpha, str):
        a = ccd_alpha(k, mode=alpha, n_factorial=len(factorial), n_center=n_center)
        alpha_mode = alpha
    else:
        a = float(alpha)
        alpha_mode = "custom"

    # Puntos axiales
    axial = []
    for i in range(k):
        for sign in (-1.0, 1.0):
            row = np.zeros(k)
            row[i] = sign * a
            axial.append(row)
    axial = np.array(axial)

    center = np.zeros((n_center, k))

    coded = np.vstack([factorial, axial, center])
    names = [f.name for f in factors]

    coded_df = pd.DataFrame(coded, columns=names)
    block = (["Factorial"] * len(factorial)
             + ["Axial"] * len(axial)
             + ["Central"] * len(center))
    coded_df.insert(0, "Punto", block)
    coded_df.insert(0, "Corrida", np.arange(1, len(coded_df) + 1))

    natural_df = coded_df.copy()
    for f in factors:
        natural_df[f.name] = f.to_natural(coded_df[f.name].values)

    meta = dict(k=k, alpha=a, alpha_mode=alpha_mode, n_center=n_center,
                n_factorial=len(factorial), n_axial=len(axial))
    return DesignResult(coded_df, natural_df, list(factors), "CCD", meta)


# ---------------------------------------------------------------------------
# Diseno Box-Behnken (BBD)
# ---------------------------------------------------------------------------
# Bloques base de Box-Behnken para k = 3..7 (indices de los pares de factores
# que se combinan en {-1,+1}; el resto de factores del bloque quedan en 0).
_BBD_BLOCKS = {
    3: [(0, 1), (0, 2), (1, 2)],
    4: [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)],
    5: [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3),
        (1, 4), (2, 3), (2, 4), (3, 4)],
}


def box_behnken_design(factors: Sequence[Factor],
                       n_center: int = 3) -> DesignResult:
    """
    Genera un Diseno Box-Behnken (BBD) para k = 3, 4 o 5 factores.

    En cada bloque se combina un par de factores en el factorial 2^2 (+-1)
    mientras los demas quedan en su nivel central (0). No incluye vertices,
    por lo que evita combinaciones extremas simultaneas (util cuando esos
    puntos son inviables o costosos en la practica).
    """
    k = len(factors)
    if k not in _BBD_BLOCKS:
        raise ValueError("BBD implementado para k = 3, 4 o 5 factores.")

    ff2 = _full_factorial_2level(2)     # 4 combinaciones (+-1, +-1)
    rows = []
    for (i, j) in _BBD_BLOCKS[k]:
        for combo in ff2:
            row = np.zeros(k)
            row[i] = combo[0]
            row[j] = combo[1]
            rows.append(row)
    edge = np.array(rows)
    center = np.zeros((n_center, k))

    coded = np.vstack([edge, center])
    names = [f.name for f in factors]

    coded_df = pd.DataFrame(coded, columns=names)
    block = ["Arista"] * len(edge) + ["Central"] * len(center)
    coded_df.insert(0, "Punto", block)
    coded_df.insert(0, "Corrida", np.arange(1, len(coded_df) + 1))

    natural_df = coded_df.copy()
    for f in factors:
        natural_df[f.name] = f.to_natural(coded_df[f.name].values)

    meta = dict(k=k, n_center=n_center, n_edge=len(edge))
    return DesignResult(coded_df, natural_df, list(factors), "BBD", meta)


# ---------------------------------------------------------------------------
# Prueba rapida
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    facs = [
        Factor("Temperatura", 110, 150, "C"),
        Factor("Tiempo", 15, 35, "min"),
        Factor("Velocidad", 1.0, 2.0, "m/s"),
    ]
    ccd = central_composite_design(facs, n_center=5)
    print("CCD:", ccd.n_runs, "corridas, alpha =", round(ccd.meta["alpha"], 3))
    bbd = box_behnken_design(facs, n_center=5)
    print("BBD:", bbd.n_runs, "corridas")
    print(bbd.natural.head())
