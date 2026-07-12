"""
app.py  -  Aplicativo RSM  (Metodologia de Superficie de Respuesta)
===================================================================

Aplicativo interactivo (Streamlit) que integra los metodos RSM para un caso
del sector agroindustrial ecuatoriano: optimizacion del tostado de cacao
Nacional. Permite a un usuario no especialista:

  1. Definir factores y generar diseños (CCD / Box-Behnken).
  2. Cargar datos experimentales.
  3. Ajustar modelos de 1er/2do orden con ANOVA, falta de ajuste y R2.
  4. Diagnosticar (residuos, Pareto, perturbacion).
  5. Visualizar contornos y superficies 3D.
  6. Optimizar: ascenso pronunciado, analisis canonico, cresta y numerico.
  7. Optimizar multiples respuestas (deseabilidad de Derringer-Suich).
  8. Exportar un reporte de resultados.

Ejecutar con:   streamlit run app.py
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from rsm.designs import Factor, central_composite_design, box_behnken_design
from rsm.models import fit_rsm
from rsm import optimization as opt
from rsm import desirability as des
from rsm import plots


# ---------------------------------------------------------------------------
# Configuracion general
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Aplicativo RSM - Agroindustria EC",
                   page_icon="📈", layout="wide")

DATA_DEMO = "data/tostado_cacao.csv"


def _default_factors():
    return [
        Factor("Temperatura", 110.0, 150.0, "C"),
        Factor("Tiempo", 20.0, 50.0, "min"),
        Factor("Velocidad", 1.0, 2.0, "m/s"),
    ]


if "factors" not in st.session_state:
    st.session_state.factors = _default_factors()
if "data" not in st.session_state:
    st.session_state.data = None


def code_dataframe(df: pd.DataFrame, factors) -> pd.DataFrame:
    """Devuelve un DataFrame con los factores codificados."""
    out = pd.DataFrame(index=df.index)
    for f in factors:
        out[f.name] = f.to_coded(df[f.name].values)
    return out


# ===========================================================================
# Barra lateral: definicion de factores
# ===========================================================================
st.sidebar.title("Configuracion")
st.sidebar.caption("Metodologia de Superficie de Respuesta (RSM)")

with st.sidebar.expander("1. Definir factores", expanded=True):
    k = st.number_input("Numero de factores (k)", 2, 5,
                        len(st.session_state.factors), 1)
    facs = []
    for idx in range(int(k)):
        base = st.session_state.factors[idx] if idx < len(st.session_state.factors) \
            else Factor(f"X{idx+1}", -1.0, 1.0, "")
        c1, c2, c3 = st.columns(3)
        name = c1.text_input(f"Nombre {idx+1}", base.name, key=f"nm{idx}")
        low = c2.number_input(f"Bajo (-1) {idx+1}", value=float(base.low),
                              key=f"lo{idx}", format="%.4f")
        high = c3.number_input(f"Alto (+1) {idx+1}", value=float(base.high),
                               key=f"hi{idx}", format="%.4f")
        units = st.text_input(f"Unidades {idx+1}", base.units, key=f"un{idx}")
        facs.append(Factor(name, low, high, units))
    st.session_state.factors = facs

factors = st.session_state.factors
factor_names = [f.name for f in factors]

st.sidebar.markdown("---")
st.sidebar.info("Caso demo: **Tostado de cacao Nacional**. "
                "Respuestas: *Aroma* (maximizar) y *Acidez* (minimizar).")


# ===========================================================================
# Encabezado
# ===========================================================================
st.title("Aplicativo de resolucion RSM")
st.markdown(
    "Superficie de Respuesta aplicada al **sector agroindustrial ecuatoriano**. "
    "Flujo completo: diseno experimental -> ajuste y diagnostico -> "
    "optimizacion mono y multi-respuesta."
)

tabs = st.tabs([
    "1. Diseño",
    "2. Datos",
    "3. Modelo y ANOVA",
    "4. Diagnostico",
    "5. Visualizacion",
    "6. Optimizacion",
    "7. Multi-respuesta",
    "8. Reporte",
])


# ===========================================================================
# TAB 1 - Diseno experimental
# ===========================================================================
with tabs[0]:
    st.header("Generacion de diseno experimental")
    c1, c2, c3 = st.columns(3)
    kind = c1.selectbox("Tipo de diseno",
                        ["Central Compuesto (CCD)", "Box-Behnken (BBD)"])
    n_center = c2.number_input("Puntos centrales", 1, 10, 5, 1)
    if kind.startswith("Central"):
        alpha_mode = c3.selectbox("Alpha (axial)",
                                  ["rotatable", "face", "orthogonal"])
    else:
        alpha_mode = None

    if st.button("Generar diseno", type="primary"):
        try:
            if kind.startswith("Central"):
                dr = central_composite_design(factors, int(n_center), alpha_mode)
            else:
                dr = box_behnken_design(factors, int(n_center))
            st.session_state.design = dr
        except Exception as e:
            st.error(f"No se pudo generar el diseno: {e}")

    if "design" in st.session_state:
        dr = st.session_state.design
        st.success(f"Diseno {dr.kind} generado: **{dr.n_runs} corridas**. "
                   + (f"alpha = {dr.meta.get('alpha'):.3f}"
                      if dr.kind == "CCD" else ""))
        cc1, cc2 = st.columns(2)
        with cc1:
            st.caption("Matriz codificada")
            st.dataframe(dr.coded.round(3), height=340, use_container_width=True)
        with cc2:
            st.caption("Matriz en unidades naturales")
            st.dataframe(dr.natural.round(3), height=340, use_container_width=True)

        csv = dr.natural.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar plan experimental (CSV)", csv,
                           file_name=f"diseno_{dr.kind}.csv", mime="text/csv")
        st.info("Realice los experimentos segun este plan, registre las "
                "respuestas y cargue el archivo en la pestana **2. Datos**.")


# ===========================================================================
# TAB 2 - Datos
# ===========================================================================
with tabs[1]:
    st.header("Carga de datos experimentales")
    c1, c2 = st.columns([2, 1])
    up = c1.file_uploader("Suba un CSV con columnas de factores y respuestas",
                          type=["csv"])
    if c2.button("Usar datos demo (cacao)"):
        try:
            st.session_state.data = pd.read_csv(DATA_DEMO)
            st.success("Datos demo cargados.")
        except Exception as e:
            st.error(f"No se encontro {DATA_DEMO}: {e}")

    if up is not None:
        try:
            st.session_state.data = pd.read_csv(up)
            st.success("Archivo cargado.")
        except Exception as e:
            st.error(f"Error al leer el CSV: {e}")

    if st.session_state.data is not None:
        df = st.session_state.data
        st.dataframe(df, use_container_width=True, height=320)
        cols = list(df.columns)
        st.markdown("**Mapeo de columnas**")
        cc = st.columns(len(factors) + 1)
        fmap = {}
        for i, f in enumerate(factors):
            default = f.name if f.name in cols else cols[min(i, len(cols)-1)]
            fmap[f.name] = cc[i].selectbox(f"Factor: {f.name}", cols,
                                           index=cols.index(default),
                                           key=f"map{i}")
        resp_default = [c for c in cols if c not in fmap.values()]
        responses = st.multiselect("Columnas de respuesta",
                                   cols, default=resp_default)
        st.session_state.fmap = fmap
        st.session_state.responses = responses
        st.session_state.df_factor_natural = df.rename(
            columns={v: k2 for k2, v in fmap.items()})
        if responses:
            st.success(f"Respuestas seleccionadas: {', '.join(responses)}")
    else:
        st.warning("Cargue datos o use el conjunto demo para continuar.")


# ---------------------------------------------------------------------------
# Ajuste de modelos (compartido por varias pestanas)
# ---------------------------------------------------------------------------
def _fit_all(order: int):
    df = st.session_state.data
    fmap = st.session_state.get("fmap", {f.name: f.name for f in factors})
    responses = st.session_state.get("responses", [])
    nat = df.rename(columns={v: k2 for k2, v in fmap.items()})
    Xc = code_dataframe(nat, factors)
    fits = {}
    for r in responses:
        fits[r] = fit_rsm(Xc, df[r].values, factor_names, response=r, order=order)
    return fits


# ===========================================================================
# TAB 3 - Modelo y ANOVA
# ===========================================================================
with tabs[2]:
    st.header("Ajuste del modelo y ANOVA")
    if st.session_state.data is None or not st.session_state.get("responses"):
        st.warning("Primero cargue datos y seleccione respuestas (pestana 2).")
    else:
        order = st.radio("Orden del modelo", [1, 2], index=1, horizontal=True,
                         format_func=lambda o: f"{o}er orden" if o == 1 else "2do orden")
        st.session_state.order = order
        try:
            fits = _fit_all(order)
            st.session_state.fits = fits
        except Exception as e:
            st.error(f"Error al ajustar: {e}")
            fits = {}

        for r, fit in fits.items():
            st.subheader(f"Respuesta: {r}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("R2", f"{fit.r2:.4f}")
            m2.metric("R2 ajustado", f"{fit.r2_adj:.4f}")
            m3.metric("R2 predictivo", f"{fit.r2_pred:.4f}")
            m4.metric("PRESS", f"{fit.press:.3f}")

            st.caption("Coeficientes (variables codificadas)")
            st.dataframe(fit.summary_coef().round(4), use_container_width=True)

            cc1, cc2 = st.columns(2)
            with cc1:
                st.caption("ANOVA del modelo")
                st.dataframe(fit.anova.round(4), use_container_width=True)
            with cc2:
                st.caption("Falta de ajuste vs. error puro")
                if fit.lof is not None:
                    st.dataframe(fit.lof.round(4), use_container_width=True)
                    p_lof = fit.lof.loc[0, "p-valor"]
                    if p_lof > 0.05:
                        st.success(f"Falta de ajuste NO significativa "
                                   f"(p = {p_lof:.3f}). El modelo es adecuado.")
                    else:
                        st.warning(f"Falta de ajuste significativa "
                                   f"(p = {p_lof:.3f}). Revise el modelo.")
                else:
                    st.info("Sin replicas suficientes para separar el error puro.")

            # Ecuacion del modelo
            eq = f"{r} = {fit.coef['Intercept']:.3f}"
            for nm in fit.coef.index:
                if nm == "Intercept":
                    continue
                c = fit.coef[nm]
                eq += f" {'+' if c >= 0 else '-'} {abs(c):.3f}*{nm}"
            st.caption("Modelo ajustado (codificado)")
            st.code(eq, language="text")
            st.markdown("---")


# ===========================================================================
# TAB 4 - Diagnostico
# ===========================================================================
with tabs[3]:
    st.header("Diagnostico del modelo")
    fits = st.session_state.get("fits")
    if not fits:
        st.warning("Ajuste un modelo en la pestana 3.")
    else:
        r = st.selectbox("Respuesta", list(fits.keys()), key="diag_r")
        fit = fits[r]
        alpha = st.slider("Nivel de significancia (Pareto)", 0.01, 0.20, 0.05, 0.01)
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plots.pareto_plot(fit, alpha))
        with c2:
            st.pyplot(plots.perturbation_plot(fit, factors))
        st.pyplot(plots.residual_plots(fit))


# ===========================================================================
# TAB 5 - Visualizacion
# ===========================================================================
with tabs[4]:
    st.header("Visualizacion de la superficie")
    fits = st.session_state.get("fits")
    if not fits:
        st.warning("Ajuste un modelo en la pestana 3.")
    else:
        r = st.selectbox("Respuesta", list(fits.keys()), key="viz_r")
        fit = fits[r]
        c1, c2, c3 = st.columns(3)
        i = c1.selectbox("Factor eje X", range(len(factors)),
                         format_func=lambda t: factor_names[t], key="viz_i")
        j = c2.selectbox("Factor eje Y", range(len(factors)),
                         index=min(1, len(factors)-1),
                         format_func=lambda t: factor_names[t], key="viz_j")
        natural_axes = c3.checkbox("Ejes en unidades naturales", True)

        fixed = np.zeros(len(factors))
        if len(factors) > 2:
            st.caption("Nivel (codificado) de los factores no graficados")
            others = [t for t in range(len(factors)) if t not in (i, j)]
            oc = st.columns(len(others))
            for c, t in zip(oc, others):
                fixed[t] = c.slider(factor_names[t], -1.7, 1.7, 0.0, 0.1,
                                    key=f"fix{t}")

        if i == j:
            st.error("Seleccione dos factores distintos.")
        else:
            cc1, cc2 = st.columns(2)
            with cc1:
                st.plotly_chart(
                    plots.contour_plot(fit, factors, i, j, fixed, natural_axes),
                    use_container_width=True)
            with cc2:
                st.plotly_chart(
                    plots.surface_plot(fit, factors, i, j, fixed, natural_axes),
                    use_container_width=True)


# ===========================================================================
# TAB 6 - Optimizacion (mono-respuesta)
# ===========================================================================
with tabs[5]:
    st.header("Optimizacion de una respuesta")
    fits = st.session_state.get("fits")
    if not fits:
        st.warning("Ajuste un modelo en la pestana 3.")
    else:
        r = st.selectbox("Respuesta a optimizar", list(fits.keys()), key="opt_r")
        fit = fits[r]
        maximize = st.radio("Objetivo", ["Maximizar", "Minimizar"],
                            horizontal=True) == "Maximizar"

        method = st.selectbox("Metodo", [
            "Ascenso/descenso mas pronunciado",
            "Analisis canonico (punto estacionario)",
            "Analisis de cresta (ridge)",
            "Optimizacion numerica",
        ])

        if method.startswith("Ascenso"):
            c1, c2 = st.columns(2)
            step = c1.slider("Tamano de paso (codificado)", 0.1, 1.0, 0.5, 0.1)
            nsteps = c2.slider("Numero de pasos", 3, 20, 8, 1)
            sp = opt.steepest_path(fit, factors, step=step, n_steps=nsteps,
                                   maximize=maximize)
            st.caption("Trayectoria (unidades naturales)")
            st.dataframe(sp.path_natural.round(3), use_container_width=True)
            st.info("Direccion codificada: "
                    + ", ".join(f"{n}={v:+.3f}"
                                for n, v in zip(factor_names, sp.direction)))

        elif method.startswith("Analisis canonico"):
            ca = opt.canonical_analysis(fit, factors)
            st.metric("Naturaleza del punto estacionario", ca.nature)
            tbl = pd.DataFrame({
                "Factor": factor_names,
                "x* (codificado)": np.round(ca.x_stationary_coded, 4),
                "X* (natural)": np.round(ca.x_stationary_natural, 4),
            })
            st.dataframe(tbl, use_container_width=True)
            st.metric(f"{r} en el punto estacionario",
                      f"{ca.y_stationary:.3f}")
            st.caption("Eigenvalores de B (signo -> naturaleza)")
            st.dataframe(pd.DataFrame({
                "Eigenvalor": np.round(ca.eigenvalues, 4)}),
                use_container_width=True)

        elif method.startswith("Analisis de cresta"):
            rr = opt.ridge_analysis(fit, factors, maximize=maximize)
            st.caption("Optimo restringido por radio (unidades naturales)")
            st.dataframe(rr.table_natural.round(3), use_container_width=True)
            fig = plots.go.Figure()
            fig.add_scatter(x=rr.table_coded["Radio"], y=rr.table_coded["y_pred"],
                            mode="lines+markers", name=r)
            fig.update_layout(title="Respuesta optima a lo largo de la cresta",
                              xaxis_title="Radio (codificado)",
                              yaxis_title=r, template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)

        else:  # numerica
            no = opt.numeric_optimum(fit, factors, maximize=maximize)
            st.success(f"Optimo numerico: {r} = {no.y:.3f}  "
                       f"({'ok' if no.success else no.message})")
            st.dataframe(pd.DataFrame({
                "Factor": factor_names,
                "x* (codificado)": np.round(no.x_coded, 4),
                "X* (natural)": np.round(no.x_natural, 4),
            }), use_container_width=True)


# ===========================================================================
# TAB 7 - Multi-respuesta (deseabilidad)
# ===========================================================================
with tabs[6]:
    st.header("Optimizacion multi-respuesta (Derringer-Suich)")
    fits = st.session_state.get("fits")
    if not fits or len(fits) < 1:
        st.warning("Ajuste al menos una respuesta (pestana 3).")
    else:
        st.caption("Defina el objetivo, los limites y el peso de cada respuesta.")
        goals = []
        for r, fit in fits.items():
            with st.expander(f"Respuesta: {r}", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                goal = c1.selectbox("Objetivo", ["max", "min", "target"],
                                    key=f"g{r}")
                ymin = float(np.min(fit.y)); ymax = float(np.max(fit.y))
                low = c2.number_input("Limite inferior (L)", value=round(ymin, 3),
                                      key=f"lo_{r}", format="%.3f")
                high = c3.number_input("Limite superior (U)", value=round(ymax, 3),
                                       key=f"hi_{r}", format="%.3f")
                weight = c4.number_input("Peso (w)", 0.1, 5.0, 1.0, 0.1,
                                         key=f"w_{r}")
                target = None
                if goal == "target":
                    target = st.number_input("Valor objetivo (T)",
                                             value=round((ymin+ymax)/2, 3),
                                             key=f"t_{r}", format="%.3f")
                goals.append(des.ResponseGoal(fit, r, goal, low, high,
                                              target, weight))

        bound = st.slider("Region de busqueda (|x| max, codificado)",
                          1.0, 1.7, 1.0, 0.1)
        if st.button("Optimizar deseabilidad", type="primary"):
            res = des.optimize_desirability(goals, factors, bound=bound)
            st.session_state.des_result = res

        if "des_result" in st.session_state:
            res = st.session_state.des_result
            st.metric("Deseabilidad global D", f"{res.D:.4f}")
            st.caption("Condiciones optimas (unidades naturales)")
            st.dataframe(pd.DataFrame({
                "Factor": factor_names,
                "x* (codificado)": np.round(res.x_coded, 4),
                "X* (natural)": np.round(res.x_natural, 4),
            }), use_container_width=True)
            st.caption("Respuestas y deseabilidad individual en el optimo")
            det = pd.DataFrame([
                {"Respuesta": nm, "y predicho": round(v[0], 3),
                 "d_i": round(v[1], 4)}
                for nm, v in res.individual.items()])
            st.dataframe(det, use_container_width=True)
            if res.D == 0:
                st.warning("D = 0: alguna respuesta quedo fuera de sus limites. "
                           "Ajuste L/U o amplie la region de busqueda.")


# ===========================================================================
# TAB 8 - Reporte
# ===========================================================================
with tabs[7]:
    st.header("Reporte de resultados")
    fits = st.session_state.get("fits")
    if not fits:
        st.warning("Genere resultados en las pestanas anteriores.")
    else:
        buf = io.StringIO()
        buf.write("REPORTE DE ANALISIS RSM\n")
        buf.write("=" * 60 + "\n\n")
        buf.write("FACTORES\n")
        for f in factors:
            buf.write(f"  - {f.name}: [{f.low}, {f.high}] {f.units} "
                      f"(centro {f.center}, semirango {f.half_range})\n")
        buf.write("\n")
        for r, fit in fits.items():
            buf.write(f"RESPUESTA: {r}\n")
            buf.write(f"  R2={fit.r2:.4f}  R2adj={fit.r2_adj:.4f}  "
                      f"R2pred={fit.r2_pred:.4f}\n")
            buf.write("  Coeficientes (codificados):\n")
            for nm in fit.coef.index:
                buf.write(f"    {nm:>15s} = {fit.coef[nm]:+.4f} "
                          f"(p={fit.pvalues[nm]:.4f})\n")
            if fit.lof is not None:
                buf.write(f"  Falta de ajuste p = {fit.lof.loc[0,'p-valor']:.4f}\n")
            buf.write("\n")
        if "des_result" in st.session_state:
            res = st.session_state.des_result
            buf.write("OPTIMIZACION MULTI-RESPUESTA (Derringer-Suich)\n")
            buf.write(f"  Deseabilidad global D = {res.D:.4f}\n")
            for nm, v in zip(factor_names, res.x_natural):
                buf.write(f"    {nm} = {v:.4f}\n")
            for nm, v in res.individual.items():
                buf.write(f"    {nm}: y={v[0]:.3f}  d={v[1]:.4f}\n")

        report = buf.getvalue()
        st.text_area("Vista previa", report, height=380)
        st.download_button("Descargar reporte (TXT)",
                           report.encode("utf-8"),
                           file_name="reporte_rsm.txt", mime="text/plain")

st.markdown("---")
st.caption("Aplicativo RSM - Optimizacion agroindustrial. "
           "Universidad Central del Ecuador. Uso academico.")
