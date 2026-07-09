"""
plots.py
========

Visualizaciones para RSM (Plotly + Matplotlib):

  - contour_plot()        : grafico de contorno de la respuesta (2 factores)
  - surface_plot()        : superficie de respuesta 3D
  - pareto_plot()         : diagrama de Pareto de efectos estandarizados
  - perturbation_plot()   : grafico de perturbacion
  - residual_plots()      : diagnostico de residuos (4 paneles)

Los factores no graficados se fijan en un nivel de referencia (por defecto 0,
el centro del diseno codificado, o el valor optimo si se entrega).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from .models import RSMFit, standardized_effects


# ---------------------------------------------------------------------------
# Malla de prediccion para 2 factores
# ---------------------------------------------------------------------------
def _grid_prediction(fit: RSMFit, i: int, j: int, fixed: np.ndarray,
                     lo: float = -1.7, hi: float = 1.7, n: int = 60):
    names = fit.factor_names
    xi = np.linspace(lo, hi, n)
    xj = np.linspace(lo, hi, n)
    Xi, Xj = np.meshgrid(xi, xj)
    pts = []
    for a in range(n):
        for b in range(n):
            x = fixed.copy()
            x[i] = Xi[a, b]
            x[j] = Xj[a, b]
            pts.append({nm: x[t] for t, nm in enumerate(names)})
    Z = fit.predict(pd.DataFrame(pts)).reshape(n, n)
    return Xi, Xj, Z, xi, xj


def contour_plot(fit: RSMFit, factors, i: int = 0, j: int = 1,
                 fixed_coded: np.ndarray | None = None,
                 natural_axes: bool = True):
    """Grafico de contorno de y_pred respecto a los factores i, j."""
    k = len(fit.factor_names)
    fixed = np.zeros(k) if fixed_coded is None else np.array(fixed_coded, float)
    Xi, Xj, Z, xi, xj = _grid_prediction(fit, i, j, fixed)

    if natural_axes:
        ax_x = factors[i].to_natural(xi)
        ax_y = factors[j].to_natural(xj)
        xlab = f"{factors[i].name} ({factors[i].units})"
        ylab = f"{factors[j].name} ({factors[j].units})"
    else:
        ax_x, ax_y = xi, xj
        xlab, ylab = fit.factor_names[i], fit.factor_names[j]

    fig = go.Figure(data=go.Contour(
        x=ax_x, y=ax_y, z=Z, colorscale="Viridis",
        contours=dict(showlabels=True),
        colorbar=dict(title=fit.response),
    ))
    fig.update_layout(
        title=f"Contorno de {fit.response}",
        xaxis_title=xlab, yaxis_title=ylab,
        template="plotly_white", height=480,
    )
    return fig


def surface_plot(fit: RSMFit, factors, i: int = 0, j: int = 1,
                 fixed_coded: np.ndarray | None = None,
                 natural_axes: bool = True):
    """Superficie de respuesta 3D respecto a los factores i, j."""
    k = len(fit.factor_names)
    fixed = np.zeros(k) if fixed_coded is None else np.array(fixed_coded, float)
    Xi, Xj, Z, xi, xj = _grid_prediction(fit, i, j, fixed)

    if natural_axes:
        ax_x = factors[i].to_natural(xi)
        ax_y = factors[j].to_natural(xj)
        xlab = f"{factors[i].name} ({factors[i].units})"
        ylab = f"{factors[j].name} ({factors[j].units})"
    else:
        ax_x, ax_y = xi, xj
        xlab, ylab = fit.factor_names[i], fit.factor_names[j]

    fig = go.Figure(data=go.Surface(
        x=ax_x, y=ax_y, z=Z, colorscale="Viridis",
        colorbar=dict(title=fit.response),
    ))
    fig.update_layout(
        title=f"Superficie de respuesta de {fit.response}",
        scene=dict(xaxis_title=xlab, yaxis_title=ylab,
                   zaxis_title=fit.response),
        template="plotly_white", height=560,
    )
    return fig


def pareto_plot(fit: RSMFit, alpha: float = 0.05):
    """
    Diagrama de Pareto de efectos estandarizados (|t| por termino).
    La linea vertical es el valor critico t para el nivel alpha dado.
    """
    eff = standardized_effects(fit)
    tcrit = stats.t.ppf(1 - alpha / 2, max(fit.dof_resid, 1))

    fig, ax = plt.subplots(figsize=(7, max(3, 0.45 * len(eff))))
    colors = ["#2e7d32" if v >= tcrit else "#9e9e9e" for v in eff.values]
    ax.barh(eff.index, eff.values, color=colors)
    ax.axvline(tcrit, color="#c62828", linestyle="--",
               label=f"t critico = {tcrit:.2f} (alpha={alpha})")
    ax.set_xlabel("|Efecto estandarizado|  (|t|)")
    ax.set_title(f"Diagrama de Pareto - {fit.response}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def perturbation_plot(fit: RSMFit, factors,
                      center_coded: np.ndarray | None = None,
                      lo: float = -1.0, hi: float = 1.0, n: int = 50):
    """
    Grafico de perturbacion: variacion de y_pred al mover UN factor a la vez
    desde un punto de referencia (por defecto el centro), manteniendo el resto
    constante. Curvas empinadas = factores mas influyentes; curvatura = efecto
    cuadratico.
    """
    names = fit.factor_names
    k = len(names)
    center = np.zeros(k) if center_coded is None else np.array(center_coded, float)
    dev = np.linspace(lo, hi, n)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    letters = "ABCDEFGH"
    for idx, nm in enumerate(names):
        ys = []
        for d in dev:
            x = center.copy()
            x[idx] = center[idx] + d
            ys.append(fit.predict(pd.DataFrame([{n2: x[t] for t, n2 in enumerate(names)}]))[0])
        ax.plot(dev, ys, label=f"{letters[idx]}: {nm}", linewidth=2)
    ax.set_xlabel("Desviacion desde el punto de referencia (codificada)")
    ax.set_ylabel(f"{fit.response} (predicho)")
    ax.set_title(f"Grafico de perturbacion - {fit.response}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def residual_plots(fit: RSMFit):
    """Cuatro paneles de diagnostico de residuos."""
    resid = fit.residuals
    fitted = fit.fitted
    std_resid = resid / (np.sqrt(fit.sigma2) if fit.sigma2 > 0 else 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    # (1) Normal Q-Q
    (osm, osr), (slope, inter, r) = stats.probplot(std_resid, dist="norm")
    axes[0, 0].scatter(osm, osr, s=18, color="#1565c0")
    axes[0, 0].plot(osm, slope * osm + inter, color="#c62828")
    axes[0, 0].set_title("Q-Q normal de residuos")
    axes[0, 0].set_xlabel("Cuantiles teoricos")
    axes[0, 0].set_ylabel("Residuos estandarizados")

    # (2) Residuos vs ajustados
    axes[0, 1].scatter(fitted, std_resid, s=18, color="#1565c0")
    axes[0, 1].axhline(0, color="#c62828", linestyle="--")
    axes[0, 1].set_title("Residuos vs. valores ajustados")
    axes[0, 1].set_xlabel("Valor ajustado")
    axes[0, 1].set_ylabel("Residuo estandarizado")

    # (3) Observado vs predicho
    axes[1, 0].scatter(fit.y, fitted, s=18, color="#2e7d32")
    lims = [min(fit.y.min(), fitted.min()), max(fit.y.max(), fitted.max())]
    axes[1, 0].plot(lims, lims, color="#c62828", linestyle="--")
    axes[1, 0].set_title("Observado vs. predicho")
    axes[1, 0].set_xlabel("Observado")
    axes[1, 0].set_ylabel("Predicho")

    # (4) Residuos vs orden de corrida
    axes[1, 1].plot(np.arange(1, len(resid) + 1), std_resid,
                    marker="o", color="#6a1b9a")
    axes[1, 1].axhline(0, color="#c62828", linestyle="--")
    axes[1, 1].set_title("Residuos vs. orden de corrida")
    axes[1, 1].set_xlabel("Orden de corrida")
    axes[1, 1].set_ylabel("Residuo estandarizado")

    fig.suptitle(f"Diagnostico de residuos - {fit.response}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig
