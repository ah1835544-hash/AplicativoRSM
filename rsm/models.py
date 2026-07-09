"""
models.py
=========

Ajuste de modelos de superficie de respuesta y diagnostico:

  - Construccion de la matriz de diseno (primer o segundo orden)
  - Ajuste por minimos cuadrados ordinarios (OLS, statsmodels)
  - ANOVA de la regresion
  - Descomposicion del error residual en Falta de Ajuste (Lack of Fit)
    y Error Puro usando corridas replicadas
  - R2, R2 ajustado, R2 predictivo (Q2 via PRESS)
  - Analisis de residuos

El modelo cuadratico completo para k factores es:

    y = b0 + sum_i bi*xi + sum_i bii*xi^2 + sum_{i<j} bij*xi*xj + e
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm
    _HAS_SM = True
except Exception:                       # pragma: no cover
    _HAS_SM = False


# ---------------------------------------------------------------------------
# Construccion de terminos del modelo
# ---------------------------------------------------------------------------
def build_model_matrix(X: pd.DataFrame, factor_names: Sequence[str],
                       order: int = 2) -> pd.DataFrame:
    """
    Construye la matriz de terminos del modelo (sin la constante).

    order = 1 -> solo terminos lineales.
    order = 2 -> lineales + interacciones dobles + cuadraticos.
    """
    terms = {}
    # Lineales
    for name in factor_names:
        terms[name] = X[name].values.astype(float)

    if order >= 2:
        # Interacciones dobles
        for a, b in itertools.combinations(factor_names, 2):
            terms[f"{a}:{b}"] = X[a].values.astype(float) * X[b].values.astype(float)
        # Cuadraticos
        for name in factor_names:
            terms[f"{name}^2"] = X[name].values.astype(float) ** 2

    return pd.DataFrame(terms, index=X.index)


# ---------------------------------------------------------------------------
# Resultado del ajuste
# ---------------------------------------------------------------------------
@dataclass
class RSMFit:
    response: str
    factor_names: list[str]
    order: int
    coef: pd.Series                 # coeficientes (incluye 'Intercept')
    stderr: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    r2: float
    r2_adj: float
    r2_pred: float                  # Q2 via PRESS
    press: float
    anova: pd.DataFrame             # tabla ANOVA (regresion / residual / total)
    lof: pd.DataFrame | None        # falta de ajuste vs error puro (o None)
    residuals: np.ndarray
    fitted: np.ndarray
    y: np.ndarray
    Xmat: pd.DataFrame              # matriz de diseno usada (con Intercept)
    sigma2: float                   # cuadrado medio del error (MSE)
    dof_resid: int
    n: int

    # -- Prediccion en el espacio codificado --------------------------------
    def predict(self, Xnew: pd.DataFrame) -> np.ndarray:
        M = build_model_matrix(Xnew, self.factor_names, self.order)
        M = M[[c for c in self.coef.index if c != "Intercept"]]
        M.insert(0, "Intercept", 1.0)
        return M.values @ self.coef.values

    def summary_coef(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "Coeficiente": self.coef,
            "Error estandar": self.stderr,
            "t": self.tvalues,
            "p-valor": self.pvalues,
        })
        df["Signif."] = df["p-valor"].apply(_stars)
        return df


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "."
    return ""


# ---------------------------------------------------------------------------
# Falta de ajuste / error puro
# ---------------------------------------------------------------------------
def _lack_of_fit(Xcoded: pd.DataFrame, factor_names: Sequence[str],
                 y: np.ndarray, ss_resid: float, df_resid: int,
                 p_params: int) -> pd.DataFrame | None:
    """
    Descompone SS_residual = SS_falta_de_ajuste + SS_error_puro.

    El error puro se estima a partir de grupos de corridas con niveles de
    factores identicos (replicas). Requiere al menos una replica.
    """
    key = [tuple(np.round(r, 8)) for r in Xcoded[list(factor_names)].values]
    df = pd.DataFrame({"key": key, "y": y})
    groups = df.groupby("key")["y"]

    ss_pe = 0.0
    df_pe = 0
    for _, vals in groups:
        m = len(vals)
        if m > 1:
            ss_pe += np.sum((vals.values - vals.values.mean()) ** 2)
            df_pe += (m - 1)

    if df_pe == 0:
        return None                     # sin replicas -> no separable

    ss_lof = ss_resid - ss_pe
    df_lof = df_resid - df_pe
    if df_lof <= 0:
        return None

    ms_lof = ss_lof / df_lof
    ms_pe = ss_pe / df_pe
    f_lof = ms_lof / ms_pe if ms_pe > 0 else np.nan
    p_lof = 1.0 - stats.f.cdf(f_lof, df_lof, df_pe) if ms_pe > 0 else np.nan

    return pd.DataFrame({
        "Fuente": ["Falta de ajuste", "Error puro", "Residual total"],
        "gl": [df_lof, df_pe, df_resid],
        "SS": [ss_lof, ss_pe, ss_resid],
        "MS": [ms_lof, ms_pe, ss_resid / df_resid if df_resid else np.nan],
        "F": [f_lof, np.nan, np.nan],
        "p-valor": [p_lof, np.nan, np.nan],
    })


# ---------------------------------------------------------------------------
# PRESS y R2 predictivo
# ---------------------------------------------------------------------------
def _press(Xmat: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    """
    PRESS = sum( (e_i / (1 - h_ii))^2 ),  con h_ii los apalancamientos (hat).
    Equivale a validacion cruzada leave-one-out sin re-ajustar.
    """
    resid = y - Xmat @ beta
    XtX_inv = np.linalg.pinv(Xmat.T @ Xmat)
    H = Xmat @ XtX_inv @ Xmat.T
    h = np.clip(np.diag(H), 0, 1 - 1e-10)
    return float(np.sum((resid / (1.0 - h)) ** 2))


# ---------------------------------------------------------------------------
# Ajuste principal
# ---------------------------------------------------------------------------
def fit_rsm(Xcoded: pd.DataFrame, y: np.ndarray, factor_names: Sequence[str],
            response: str = "y", order: int = 2) -> RSMFit:
    """
    Ajusta un modelo RSM (OLS) sobre variables CODIFICADAS.

    Xcoded : DataFrame con las columnas de los factores (codificados).
    y      : vector respuesta.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    M = build_model_matrix(Xcoded, factor_names, order)
    Xmat = M.copy()
    Xmat.insert(0, "Intercept", 1.0)
    Xarr = Xmat.values
    p = Xarr.shape[1]                   # numero de parametros (incl. intercepto)

    # --- Ajuste OLS ---
    beta, *_ = np.linalg.lstsq(Xarr, y, rcond=None)
    fitted = Xarr @ beta
    resid = y - fitted

    df_resid = n - p
    ss_resid = float(resid @ resid)
    ss_total = float(np.sum((y - y.mean()) ** 2))
    ss_reg = ss_total - ss_resid
    df_reg = p - 1
    df_total = n - 1

    sigma2 = ss_resid / df_resid if df_resid > 0 else np.nan

    # Errores estandar de los coeficientes
    XtX_inv = np.linalg.pinv(Xarr.T @ Xarr)
    var_beta = sigma2 * np.diag(XtX_inv)
    stderr = np.sqrt(np.abs(var_beta))
    with np.errstate(divide="ignore", invalid="ignore"):
        tvals = beta / stderr
        pvals = 2.0 * (1.0 - stats.t.cdf(np.abs(tvals), df_resid)) if df_resid > 0 \
            else np.full_like(beta, np.nan)

    # Si statsmodels esta disponible, usarlo para robustez (mismos resultados)
    if _HAS_SM and df_resid > 0:
        res = sm.OLS(y, Xarr).fit()
        beta = res.params
        stderr = res.bse
        tvals = res.tvalues
        pvals = res.pvalues

    names = list(Xmat.columns)
    coef = pd.Series(beta, index=names)
    stderr = pd.Series(np.asarray(stderr), index=names)
    tvals = pd.Series(np.asarray(tvals), index=names)
    pvals = pd.Series(np.asarray(pvals), index=names)

    # --- R2 ---
    r2 = 1.0 - ss_resid / ss_total if ss_total > 0 else np.nan
    r2_adj = 1.0 - (ss_resid / df_resid) / (ss_total / df_total) \
        if df_resid > 0 and ss_total > 0 else np.nan
    press = _press(Xarr, y, np.asarray(beta))
    r2_pred = 1.0 - press / ss_total if ss_total > 0 else np.nan

    # --- ANOVA global ---
    ms_reg = ss_reg / df_reg if df_reg > 0 else np.nan
    ms_resid = ss_resid / df_resid if df_resid > 0 else np.nan
    f_model = ms_reg / ms_resid if ms_resid and ms_resid > 0 else np.nan
    p_model = 1.0 - stats.f.cdf(f_model, df_reg, df_resid) \
        if df_resid > 0 else np.nan
    anova = pd.DataFrame({
        "Fuente": ["Regresion (modelo)", "Residual", "Total"],
        "gl": [df_reg, df_resid, df_total],
        "SS": [ss_reg, ss_resid, ss_total],
        "MS": [ms_reg, ms_resid, np.nan],
        "F": [f_model, np.nan, np.nan],
        "p-valor": [p_model, np.nan, np.nan],
    })

    lof = _lack_of_fit(Xcoded, factor_names, y, ss_resid, df_resid, p)

    return RSMFit(
        response=response, factor_names=list(factor_names), order=order,
        coef=coef, stderr=stderr, tvalues=tvals, pvalues=pvals,
        r2=r2, r2_adj=r2_adj, r2_pred=r2_pred, press=press,
        anova=anova, lof=lof, residuals=resid, fitted=fitted, y=y,
        Xmat=Xmat, sigma2=sigma2, dof_resid=df_resid, n=n,
    )


# ---------------------------------------------------------------------------
# Efectos estandarizados (para diagrama de Pareto)
# ---------------------------------------------------------------------------
def standardized_effects(fit: RSMFit) -> pd.Series:
    """Valores |t| de cada termino (excluyendo el intercepto)."""
    t = fit.tvalues.drop(labels=["Intercept"], errors="ignore")
    return t.abs().sort_values(ascending=True)


if __name__ == "__main__":
    # Prueba minima con datos simulados
    rng = np.random.default_rng(0)
    n = 30
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    y = 10 + 2 * x1 - 3 * x2 - 1.5 * x1 ** 2 - 2 * x2 ** 2 + rng.normal(0, 0.2, n)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    fit = fit_rsm(X, y, ["x1", "x2"], "y", order=2)
    print(fit.summary_coef().round(3))
    print("R2 =", round(fit.r2, 3), "R2adj =", round(fit.r2_adj, 3))
