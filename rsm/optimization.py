"""
optimization.py
===============

Metodos de optimizacion sobre la superficie de respuesta ajustada:

  - Ascenso (o descenso) mas pronunciado      -> steepest_path()
  - Analisis canonico                          -> canonical_analysis()
       (punto estacionario y eigenvalores -> clasificacion max/min/silla)
  - Analisis de cresta (ridge analysis)        -> ridge_analysis()
  - Optimizacion numerica con restricciones    -> numeric_optimum()

Se trabaja con el modelo cuadratico en forma matricial:

        y_hat(x) = b0 + x' b + x' B x

donde  b  es el vector de coeficientes lineales y  B  es la matriz simetrica
de coeficientes cuadraticos (diagonal = bii, fuera de diagonal = bij/2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .models import RSMFit, build_model_matrix


# ---------------------------------------------------------------------------
# Extraccion de b (lineal) y B (cuadratica) desde el ajuste
# ---------------------------------------------------------------------------
def extract_b_B(fit: RSMFit) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Devuelve (b0, b, B) del modelo cuadratico en variables codificadas.

    B es simetrica: B[i,i] = coef(xi^2); B[i,j] = coef(xi:xj)/2 (i != j).
    """
    names = fit.factor_names
    k = len(names)
    coef = fit.coef

    b0 = float(coef.get("Intercept", 0.0))
    b = np.array([float(coef.get(nm, 0.0)) for nm in names])

    B = np.zeros((k, k))
    for i, nm in enumerate(names):
        B[i, i] = float(coef.get(f"{nm}^2", 0.0))
    for i in range(k):
        for j in range(i + 1, k):
            cij = float(coef.get(f"{names[i]}:{names[j]}", 0.0))
            B[i, j] = cij / 2.0
            B[j, i] = cij / 2.0
    return b0, b, B


def predict_quadratic(b0: float, b: np.ndarray, B: np.ndarray,
                      x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(b0 + x @ b + x @ B @ x)


# ---------------------------------------------------------------------------
# 1) Ascenso mas pronunciado
# ---------------------------------------------------------------------------
@dataclass
class SteepestPath:
    path_coded: pd.DataFrame
    path_natural: pd.DataFrame
    direction: np.ndarray
    maximize: bool


def steepest_path(fit: RSMFit, factors, base_response: str = "y",
                  step: float = 0.5, n_steps: int = 8,
                  maximize: bool = True,
                  use_first_order: bool = True) -> SteepestPath:
    """
    Trayectoria de ascenso (o descenso) mas pronunciado.

    La direccion se toma del gradiente del modelo en el origen codificado.
    Con use_first_order=True se usa solo la parte lineal b (recomendado
    cuando la region actual esta lejos del optimo). En caso contrario se
    usa el gradiente completo  grad = b + 2 B x  evaluado en el centro.

    step    : incremento de la variable con mayor coeficiente por paso.
    n_steps : numero de pasos a lo largo de la trayectoria.
    """
    _, b, B = extract_b_B(fit)
    grad = b if use_first_order else b  # en el centro (x=0) grad = b
    if not maximize:
        grad = -grad

    imax = int(np.argmax(np.abs(grad)))
    if grad[imax] == 0:
        grad = grad + 1e-12
    # Escalado: la variable dominante avanza 'step' por paso.
    unit = grad / abs(grad[imax])
    names = fit.factor_names

    rows = []
    for s in range(n_steps + 1):
        x = unit * step * s
        row = {nm: x[i] for i, nm in enumerate(names)}
        row["Paso"] = s
        row["y_pred"] = fit.predict(pd.DataFrame([{nm: x[i] for i, nm in enumerate(names)}]))[0]
        rows.append(row)

    coded = pd.DataFrame(rows)[["Paso"] + names + ["y_pred"]]
    natural = coded.copy()
    for f in factors:
        natural[f.name] = f.to_natural(coded[f.name].values)

    return SteepestPath(coded, natural, unit, maximize)


# ---------------------------------------------------------------------------
# 2) Analisis canonico
# ---------------------------------------------------------------------------
@dataclass
class CanonicalResult:
    x_stationary_coded: np.ndarray
    x_stationary_natural: np.ndarray
    y_stationary: float
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    nature: str                         # 'maximo' | 'minimo' | 'silla' | ...
    B: np.ndarray


def canonical_analysis(fit: RSMFit, factors) -> CanonicalResult:
    """
    Analisis canonico del modelo de segundo orden.

    Punto estacionario:   x_s = -0.5 * B^{-1} * b
    Respuesta en el:      y_s = b0 + 0.5 * x_s' b
    Los eigenvalores de B clasifican la naturaleza del punto:
        todos < 0  -> maximo
        todos > 0  -> minimo
        signos mixtos -> punto silla (minimax)
    """
    b0, b, B = extract_b_B(fit)
    names = fit.factor_names

    try:
        x_s = -0.5 * np.linalg.solve(B, b)
    except np.linalg.LinAlgError:
        x_s = -0.5 * np.linalg.pinv(B) @ b

    y_s = predict_quadratic(b0, b, B, x_s)
    eigvals, eigvecs = np.linalg.eigh(B)

    tol = 1e-8
    if np.all(eigvals < -tol):
        nature = "Maximo"
    elif np.all(eigvals > tol):
        nature = "Minimo"
    elif np.any(eigvals > tol) and np.any(eigvals < -tol):
        nature = "Punto silla (minimax)"
    else:
        nature = "Cresta estacionaria / indefinido"

    x_nat = np.array([f.to_natural(x_s[i]) for i, f in enumerate(factors)])

    return CanonicalResult(
        x_stationary_coded=x_s, x_stationary_natural=x_nat,
        y_stationary=y_s, eigenvalues=eigvals, eigenvectors=eigvecs,
        nature=nature, B=B,
    )


# ---------------------------------------------------------------------------
# 3) Analisis de cresta (ridge analysis)
# ---------------------------------------------------------------------------
@dataclass
class RidgeResult:
    table_coded: pd.DataFrame
    table_natural: pd.DataFrame
    maximize: bool


def ridge_analysis(fit: RSMFit, factors, radii=None,
                   maximize: bool = True) -> RidgeResult:
    """
    Analisis de cresta: optimo restringido a esferas de radio R crecientes
    centradas en el origen codificado.

    Para cada R se resuelve:
        opt   x' b + x' B x
        s.a.  ||x|| = R

    Util cuando el punto estacionario cae fuera de la region experimental
    o es un punto silla: entrega la mejor direccion de mejora a cada radio.
    """
    b0, b, B = extract_b_B(fit)
    names = fit.factor_names
    k = len(names)
    if radii is None:
        radii = np.linspace(0.0, 1.7, 12)

    rows = []
    for R in radii:
        if R == 0:
            x_opt = np.zeros(k)
        else:
            x_opt = _ridge_point(b, B, R, maximize)
        y = predict_quadratic(b0, b, B, x_opt)
        row = {"Radio": R}
        for i, nm in enumerate(names):
            row[nm] = x_opt[i]
        row["y_pred"] = y
        rows.append(row)

    coded = pd.DataFrame(rows)
    natural = coded.copy()
    for f in factors:
        natural[f.name] = f.to_natural(coded[f.name].values)

    return RidgeResult(coded, natural, maximize)


def _ridge_point(b, B, R, maximize):
    """Optimo sobre la esfera ||x||=R via multiplicador de Lagrange (grid)."""
    k = len(b)
    eig = np.linalg.eigvalsh(B)
    sign = 1.0 if maximize else -1.0

    # Rango de mu que garantiza (mu I - B) definida positiva para el problema.
    lo = sign * eig.max() + 1e-6
    hi = lo + 200.0
    mus = np.linspace(lo, hi, 4000)

    best_x, best_val, best_gap = None, -np.inf, np.inf
    for mu in mus:
        A = (mu * np.eye(k) - sign * B)
        try:
            x = 0.5 * sign * np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        gap = abs(np.linalg.norm(x) - R)
        if gap < best_gap:
            best_gap = gap
            best_x = x
    if best_x is None:
        best_x = np.zeros(k)
    # Reescalar exactamente al radio pedido
    nrm = np.linalg.norm(best_x)
    if nrm > 0:
        best_x = best_x * (R / nrm)
    return best_x


# ---------------------------------------------------------------------------
# 4) Optimizacion numerica (una respuesta)
# ---------------------------------------------------------------------------
@dataclass
class NumericOptimum:
    x_coded: np.ndarray
    x_natural: np.ndarray
    y: float
    success: bool
    message: str


def numeric_optimum(fit: RSMFit, factors, maximize: bool = True,
                    bound: float = 1.0) -> NumericOptimum:
    """
    Optimizacion numerica del modelo dentro de la region codificada
    [-bound, bound]^k mediante SLSQP con multiples reinicios.
    """
    b0, b, B = extract_b_B(fit)
    k = len(fit.factor_names)
    sign = -1.0 if maximize else 1.0    # minimize(-y) para maximizar

    def obj(x):
        return sign * predict_quadratic(b0, b, B, x)

    bounds = [(-bound, bound)] * k
    starts = [np.zeros(k)]
    rng = np.random.default_rng(12345)
    for _ in range(20):
        starts.append(rng.uniform(-bound, bound, k))

    best = None
    for x0 in starts:
        r = minimize(obj, x0, method="SLSQP", bounds=bounds)
        if r.success and (best is None or r.fun < best.fun):
            best = r
    if best is None:
        best = minimize(obj, np.zeros(k), method="L-BFGS-B", bounds=bounds)

    x_opt = best.x
    y_opt = sign * best.fun
    x_nat = np.array([f.to_natural(x_opt[i]) for i, f in enumerate(factors)])
    return NumericOptimum(x_opt, x_nat, float(y_opt), bool(best.success),
                          str(best.message))
