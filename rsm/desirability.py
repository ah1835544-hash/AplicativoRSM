"""
desirability.py
===============

Optimizacion de multiples respuestas mediante la funcion de deseabilidad
de Derringer & Suich (1980).

Para cada respuesta i se define una deseabilidad individual d_i in [0, 1]:

  - Maximizar (larger-the-better):
        d = 0                              si y <= L
        d = ((y - L)/(T - L))^s            si L < y < T
        d = 1                              si y >= T

  - Minimizar (smaller-the-better):
        d = 1                              si y <= T
        d = ((U - y)/(U - T))^s            si T < y < U
        d = 0                              si y >= U

  - Objetivo (target-is-best): dos ramas hacia el valor objetivo T.

La deseabilidad global es la media geometrica ponderada:

        D = ( prod_i d_i^{w_i} )^{1 / sum(w_i)}

Se maximiza D dentro de la region experimental codificada.

Referencia:
  Derringer, G., & Suich, R. (1980). Simultaneous optimization of several
  response variables. Journal of Quality Technology, 12(4), 214-219.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .models import RSMFit


@dataclass
class ResponseGoal:
    """Objetivo de una respuesta para la deseabilidad."""
    fit: RSMFit
    name: str
    goal: str                   # 'max' | 'min' | 'target'
    low: float                  # L
    high: float                 # U
    target: float | None = None
    weight: float = 1.0         # importancia relativa (w)
    s_low: float = 1.0          # exponente rama inferior
    s_high: float = 1.0         # exponente rama superior

    def desirability(self, y: float) -> float:
        g = self.goal
        if g == "max":
            if y <= self.low:
                return 0.0
            if y >= self.high:
                return 1.0
            return ((y - self.low) / (self.high - self.low)) ** self.s_low
        if g == "min":
            if y <= self.low:
                return 1.0
            if y >= self.high:
                return 0.0
            return ((self.high - y) / (self.high - self.low)) ** self.s_high
        if g == "target":
            T = self.target if self.target is not None else \
                0.5 * (self.low + self.high)
            if y < self.low or y > self.high:
                return 0.0
            if y <= T:
                return ((y - self.low) / (T - self.low)) ** self.s_low
            return ((self.high - y) / (self.high - T)) ** self.s_high
        raise ValueError(f"Objetivo desconocido: {self.goal!r}")


@dataclass
class DesirabilityResult:
    x_coded: np.ndarray
    x_natural: np.ndarray
    D: float
    individual: dict            # nombre respuesta -> (y_pred, d_i)
    factors: list
    goals: list
    success: bool = True
    grid: pd.DataFrame = field(default=None)


def overall_desirability(x: np.ndarray, goals: list[ResponseGoal],
                         factor_names) -> tuple[float, dict]:
    """Evalua D global y las deseabilidades individuales en un punto x."""
    Xdf = pd.DataFrame([{nm: x[i] for i, nm in enumerate(factor_names)}])
    ds = []
    ws = []
    detail = {}
    for goal in goals:
        y = float(goal.fit.predict(Xdf)[0])
        d = goal.desirability(y)
        detail[goal.name] = (y, d)
        ds.append(max(d, 0.0))
        ws.append(goal.weight)
    ds = np.array(ds)
    ws = np.array(ws)
    if np.any(ds <= 0):
        D = 0.0
    else:
        D = float(np.exp(np.sum(ws * np.log(ds)) / np.sum(ws)))
    return D, detail


def optimize_desirability(goals: list[ResponseGoal], factors,
                          bound: float = 1.0,
                          n_starts: int = 40) -> DesirabilityResult:
    """
    Maximiza la deseabilidad global D dentro de [-bound, bound]^k.

    Usa SLSQP con multiples reinicios aleatorios para evitar optimos locales.
    """
    factor_names = [f.name for f in factors]
    k = len(factor_names)

    def neg_D(x):
        D, _ = overall_desirability(x, goals, factor_names)
        return -D

    bounds = [(-bound, bound)] * k
    starts = [np.zeros(k)]
    rng = np.random.default_rng(2024)
    for _ in range(n_starts):
        starts.append(rng.uniform(-bound, bound, k))

    best = None
    for x0 in starts:
        r = minimize(neg_D, x0, method="SLSQP", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r

    x_opt = best.x
    D, detail = overall_desirability(x_opt, goals, factor_names)
    x_nat = np.array([f.to_natural(x_opt[i]) for i, f in enumerate(factors)])

    return DesirabilityResult(
        x_coded=x_opt, x_natural=x_nat, D=D, individual=detail,
        factors=list(factors), goals=list(goals),
        success=bool(best.success),
    )
