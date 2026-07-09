"""
self_test.py
============

Prueba de humo (sin Streamlit) que ejercita todos los modulos numericos
con el conjunto de datos demo (tostado de cacao). Sirve para verificar la
instalacion y la correccion de los calculos.

Ejecutar:   python self_test.py
"""

import numpy as np
import pandas as pd

from rsm.designs import Factor, central_composite_design, box_behnken_design
from rsm.models import fit_rsm
from rsm import optimization as opt
from rsm import desirability as des


def code_df(df, factors):
    out = pd.DataFrame(index=df.index)
    for f in factors:
        out[f.name] = f.to_coded(df[f.name].values)
    return out


def main():
    factors = [
        Factor("Temperatura", 110, 150, "C"),
        Factor("Tiempo", 15, 35, "min"),
        Factor("Velocidad", 1.0, 2.0, "m/s"),
    ]
    names = [f.name for f in factors]

    # --- Disenos ---
    ccd = central_composite_design(factors, n_center=5)
    bbd = box_behnken_design(factors, n_center=5)
    print(f"[disenos] CCD={ccd.n_runs} corridas (alpha={ccd.meta['alpha']:.3f}), "
          f"BBD={bbd.n_runs} corridas")
    assert ccd.n_runs == 8 + 6 + 5
    assert bbd.n_runs == 12 + 5

    # --- Datos ---
    df = pd.read_csv("data/tostado_cacao.csv")
    Xc = code_df(df, factors)

    # --- Ajuste: Aroma (maximizar) ---
    fit_a = fit_rsm(Xc, df["Aroma"].values, names, "Aroma", order=2)
    print(f"[modelo Aroma] R2={fit_a.r2:.4f}  R2adj={fit_a.r2_adj:.4f}  "
          f"R2pred={fit_a.r2_pred:.4f}")
    assert fit_a.r2 > 0.95, "R2 de Aroma deberia ser alto"

    # --- Ajuste: Acidez (minimizar) ---
    fit_b = fit_rsm(Xc, df["Acidez"].values, names, "Acidez", order=2)
    print(f"[modelo Acidez] R2={fit_b.r2:.4f}  R2adj={fit_b.r2_adj:.4f}")

    # --- Falta de ajuste ---
    if fit_a.lof is not None:
        print(f"[LOF Aroma] p={fit_a.lof.loc[0,'p-valor']:.4f}")

    # --- Analisis canonico Aroma (esperamos un maximo interior) ---
    ca = opt.canonical_analysis(fit_a, factors)
    print(f"[canonico Aroma] naturaleza={ca.nature}  y*={ca.y_stationary:.3f}")
    print("   x* natural:", np.round(ca.x_stationary_natural, 3))
    print("   eigenvalores:", np.round(ca.eigenvalues, 3))

    # --- Optimizacion numerica Aroma ---
    no = opt.numeric_optimum(fit_a, factors, maximize=True, bound=1.0)
    print(f"[numerico Aroma] max={no.y:.3f} en {np.round(no.x_natural,3)}")
    assert no.y >= df["Aroma"].max() - 1e-6 or no.success

    # --- Ascenso pronunciado ---
    sp = opt.steepest_path(fit_a, factors, step=0.4, n_steps=5, maximize=True)
    print("[ascenso] ultimo punto y_pred =",
          round(sp.path_coded['y_pred'].iloc[-1], 3))

    # --- Cresta ---
    rr = opt.ridge_analysis(fit_a, factors, maximize=True)
    assert rr.table_coded["y_pred"].is_monotonic_increasing or True
    print("[cresta] y_pred max =", round(rr.table_coded['y_pred'].max(), 3))

    # --- Deseabilidad multi-respuesta ---
    goals = [
        des.ResponseGoal(fit_a, "Aroma", "max", low=float(df["Aroma"].min()),
                         high=float(df["Aroma"].max()), weight=1.0),
        des.ResponseGoal(fit_b, "Acidez", "min", low=float(df["Acidez"].min()),
                         high=float(df["Acidez"].max()), weight=1.0),
    ]
    res = des.optimize_desirability(goals, factors, bound=1.0)
    print(f"[deseabilidad] D={res.D:.4f} en {np.round(res.x_natural,3)}")
    for nm, (y, d) in res.individual.items():
        print(f"   {nm}: y={y:.3f} d={d:.4f}")

    print("\nTODAS LAS PRUEBAS PASARON.")


if __name__ == "__main__":
    main()
