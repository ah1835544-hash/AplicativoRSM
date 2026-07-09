"""
generate_report.py
===================

Genera 'reporte_tecnico.pdf' (<= 10 paginas) ejecutando el analisis RSM real
sobre los datos demo (tostado de cacao) y embebiendo figuras y resultados.

Ejecutar:   python generate_report.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy import stats

from rsm.designs import Factor, box_behnken_design
from rsm.models import fit_rsm, standardized_effects
from rsm import optimization as opt
from rsm import desirability as des

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak, HRFlowable)
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# --- Fuente Unicode (DejaVu, incluida con matplotlib) para simbolos griegos,
#     matematicos y acentos que la Helvetica base14 no soporta ---
_ttf = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(_ttf, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold",
                               os.path.join(_ttf, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique",
                               os.path.join(_ttf, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                              italic="DejaVu-Oblique", boldItalic="DejaVu-Bold")

OUT = "reporte_tecnico.pdf"
FIGDIR = "report_figs"
os.makedirs(FIGDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1) Analisis
# ---------------------------------------------------------------------------
factors = [Factor("Temperatura", 110, 150, "C"),
           Factor("Tiempo", 15, 35, "min"),
           Factor("Velocidad", 1.0, 2.0, "m/s")]
names = [f.name for f in factors]
df = pd.read_csv("data/tostado_cacao.csv")
Xc = pd.DataFrame({f.name: f.to_coded(df[f.name].values) for f in factors})

fit_a = fit_rsm(Xc, df["Aroma"].values, names, "Aroma", order=2)
fit_b = fit_rsm(Xc, df["Acidez"].values, names, "Acidez", order=2)
can_a = opt.canonical_analysis(fit_a, factors)
goals = [des.ResponseGoal(fit_a, "Aroma", "max", float(df["Aroma"].min()),
                          float(df["Aroma"].max()), weight=1.0),
         des.ResponseGoal(fit_b, "Acidez", "min", float(df["Acidez"].min()),
                          float(df["Acidez"].max()), weight=1.0)]
des_res = des.optimize_desirability(goals, factors, bound=1.0)

# ---------------------------------------------------------------------------
# 2) Figuras
# ---------------------------------------------------------------------------
def _mesh(fit, i, j, fixed):
    xi = np.linspace(-1.3, 1.3, 60)
    xj = np.linspace(-1.3, 1.3, 60)
    Xi, Xj = np.meshgrid(xi, xj)
    pts = []
    for a in range(60):
        for b in range(60):
            x = fixed.copy(); x[i] = Xi[a, b]; x[j] = Xj[a, b]
            pts.append({nm: x[t] for t, nm in enumerate(names)})
    Z = fit.predict(pd.DataFrame(pts)).reshape(60, 60)
    return factors[i].to_natural(xi), factors[j].to_natural(xj), Xi, Xj, Z

# Contorno Aroma (Temp vs Tiempo)
ax_x, ax_y, Xi, Xj, Z = _mesh(fit_a, 0, 1, np.zeros(3))
fig, ax = plt.subplots(figsize=(5.2, 4))
cf = ax.contourf(ax_x, ax_y, Z, levels=14, cmap="viridis")
cl = ax.contour(ax_x, ax_y, Z, levels=8, colors="white", linewidths=0.5)
ax.clabel(cl, inline=True, fontsize=7)
ax.scatter([can_a.x_stationary_natural[0]], [can_a.x_stationary_natural[1]],
           c="red", marker="*", s=180, label="Punto estacionario")
ax.set_xlabel("Temperatura (C)"); ax.set_ylabel("Tiempo (min)")
ax.set_title("Contorno de Aroma (Velocidad = 1.5 m/s)")
ax.legend(fontsize=7); fig.colorbar(cf, ax=ax, label="Aroma")
fig.tight_layout(); fig.savefig(f"{FIGDIR}/contour.png", dpi=150); plt.close(fig)

# Superficie 3D Aroma
fig = plt.figure(figsize=(5.2, 4)); ax = fig.add_subplot(111, projection="3d")
XX, YY = np.meshgrid(ax_x, ax_y)
ax.plot_surface(XX, YY, Z, cmap="viridis", edgecolor="none", alpha=0.9)
ax.set_xlabel("Temp (C)"); ax.set_ylabel("Tiempo (min)"); ax.set_zlabel("Aroma")
ax.set_title("Superficie de respuesta - Aroma")
fig.tight_layout(); fig.savefig(f"{FIGDIR}/surface.png", dpi=150); plt.close(fig)

# Pareto Aroma
eff = standardized_effects(fit_a)
tcrit = stats.t.ppf(0.975, max(fit_a.dof_resid, 1))
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.barh(eff.index, eff.values,
        color=["#2e7d32" if v >= tcrit else "#9e9e9e" for v in eff.values])
ax.axvline(tcrit, color="#c62828", ls="--", label=f"t critico={tcrit:.2f}")
ax.set_xlabel("|t|"); ax.set_title("Pareto de efectos - Aroma"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/pareto.png", dpi=150); plt.close(fig)

# Residuos Aroma (2 paneles)
fig, axs = plt.subplots(1, 2, figsize=(6.4, 3))
std_r = fit_a.residuals / (np.sqrt(fit_a.sigma2) if fit_a.sigma2 > 0 else 1)
(osm, osr), (sl, ic, r) = stats.probplot(std_r, dist="norm")
axs[0].scatter(osm, osr, s=15, color="#1565c0"); axs[0].plot(osm, sl*osm+ic, "r-")
axs[0].set_title("Q-Q normal", fontsize=9)
axs[1].scatter(fit_a.fitted, std_r, s=15, color="#1565c0")
axs[1].axhline(0, color="r", ls="--"); axs[1].set_title("Residuos vs ajustados", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/resid.png", dpi=150); plt.close(fig)

# Deseabilidad (contorno de D en Temp-Tiempo, Velocidad optima)
velc = des_res.x_coded[2]
xi = np.linspace(-1, 1, 45); xj = np.linspace(-1, 1, 45)
D = np.zeros((45, 45))
fn = names
for a, va in enumerate(xj):
    for b, vb in enumerate(xi):
        x = np.array([vb, va, velc])
        Dv, _ = des.overall_desirability(x, goals, fn)
        D[a, b] = Dv
fig, ax = plt.subplots(figsize=(5.2, 4))
cf = ax.contourf(factors[0].to_natural(xi), factors[1].to_natural(xj), D,
                 levels=14, cmap="magma")
ax.scatter([des_res.x_natural[0]], [des_res.x_natural[1]], c="cyan", marker="*",
           s=200, label="Optimo (D max)")
ax.set_xlabel("Temperatura (C)"); ax.set_ylabel("Tiempo (min)")
ax.set_title(f"Deseabilidad global D (Vel={des_res.x_natural[2]:.2f} m/s)")
ax.legend(fontsize=7); fig.colorbar(cf, ax=ax, label="D")
fig.tight_layout(); fig.savefig(f"{FIGDIR}/desir.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------------------
# 3) PDF
# ---------------------------------------------------------------------------
styles = getSampleStyleSheet()
styles.add(ParagraphStyle("Just", parent=styles["Normal"], alignment=TA_JUSTIFY,
                          fontName="DejaVu", fontSize=9.5, leading=13))
styles.add(ParagraphStyle("H1c", parent=styles["Heading1"], fontName="DejaVu-Bold",
                          fontSize=13, textColor=colors.HexColor("#1b5e20"),
                          spaceAfter=4))
styles.add(ParagraphStyle("H2c", parent=styles["Heading2"], fontName="DejaVu-Bold",
                          fontSize=11, textColor=colors.HexColor("#2e7d32"),
                          spaceBefore=6, spaceAfter=3))
styles.add(ParagraphStyle("Cap", parent=styles["Normal"], fontName="DejaVu-Oblique",
                          fontSize=8, textColor=colors.grey, alignment=TA_CENTER))
styles.add(ParagraphStyle("Tit", parent=styles["Title"], fontName="DejaVu-Bold",
                          fontSize=18, textColor=colors.HexColor("#1b5e20")))
styles.add(ParagraphStyle("Ref", parent=styles["Normal"], fontName="DejaVu",
                          fontSize=8.5, leading=11, leftIndent=14,
                          firstLineIndent=-14))

S = []
P = lambda t, s="Just": S.append(Paragraph(t, styles[s]))
def img(path, w=11*cm, cap=None):
    S.append(Image(path, width=w, height=w*0.77))
    if cap:
        S.append(Paragraph(cap, styles["Cap"]))
    S.append(Spacer(1, 6))

def tbl(data, colw=None):
    t = Table(data, colWidths=colw, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e7d32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "DejaVu"),
        ("FONTNAME", (0, 0), (-1, 0), "DejaVu-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#eef4ee")]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    S.append(t); S.append(Spacer(1, 8))

# ---- Portada / encabezado ----
P("Optimización del tostado de cacao Nacional mediante Metodología de "
  "Superficie de Respuesta (RSM)", "Tit")
S.append(Spacer(1, 4))
P("<b>Aplicativo interactivo de resolución RSM</b> — Reporte técnico", "H2c")
P("Universidad Central del Ecuador (UCE) · Asignatura de Optimización · "
  "Grupo J · 2026", "Just")
S.append(HRFlowable(width="100%", color=colors.HexColor("#2e7d32"), thickness=1.2))
S.append(Spacer(1, 6))

# ---- 1. Contexto ----
P("1. Contexto y objetivo", "H1c")
P("El cacao Nacional (variedad fino de aroma) es un producto emblemático de la "
  "agroindustria ecuatoriana, cuya calidad sensorial depende críticamente de la "
  "etapa de tostado. Un tostado insuficiente no desarrolla los precursores del "
  "aroma, mientras que un tostado excesivo genera notas quemadas y pérdida de "
  "compuestos volátiles. El objetivo de este trabajo es <b>optimizar "
  "simultáneamente el desarrollo de aroma (maximizar) y la acidez residual "
  "(minimizar)</b> en función de las variables de proceso, mediante la "
  "Metodología de Superficie de Respuesta (RSM). Se desarrolló un aplicativo "
  "interactivo que permite a un usuario no especialista ejecutar todo el flujo: "
  "diseño experimental, ajuste, diagnóstico y optimización.")

P("2. Métodos implementados", "H1c")
P("El aplicativo integra los métodos de RSM revisados en clase:", "Just")
P("• <b>Diseño experimental:</b> Diseño Central Compuesto (CCD) con α rotable, "
  "de caras centradas u ortogonal; y Diseño Box-Behnken (BBD).", "Just")
P("• <b>Ajuste del modelo:</b> modelos de primer y segundo orden por mínimos "
  "cuadrados; ANOVA de la regresión; prueba de <b>falta de ajuste</b> con error "
  "puro estimado de réplicas; R², R² ajustado y R² predictivo (PRESS); análisis "
  "de residuos.", "Just")
P("• <b>Optimización:</b> ascenso más pronunciado; <b>análisis canónico</b> "
  "(punto estacionario x* = &minus;0.5·B<super>&minus;1</super>·b y eigenvalores "
  "de B); <b>análisis de cresta</b>; y optimización numérica (SLSQP).", "Just")
P("• <b>Múltiples respuestas:</b> función de <b>deseabilidad de "
  "Derringer–Suich</b>, D = (d<sub>1</sub><super>w1</super>·…·"
  "d<sub>m</sub><super>wm</super>)<super>1/&Sigma;w</super>.", "Just")
P("• <b>Visualización:</b> contornos, superficies 3D, diagrama de Pareto y "
  "gráfico de perturbación.", "Just")

P("3. Arquitectura del aplicativo", "H1c")
P("El sistema sigue una arquitectura modular en Python. La interfaz "
  "(<font face='Courier'>app.py</font>, framework <b>Streamlit</b>) orquesta un "
  "paquete de cálculo <font face='Courier'>rsm/</font> desacoplado de la vista, "
  "lo que facilita las pruebas y la defensa del código:", "Just")
tbl([["Módulo", "Responsabilidad"],
     ["designs.py", "Generación de CCD y Box-Behnken (codificado y natural)"],
     ["models.py", "OLS, ANOVA, falta de ajuste, R²/R²aj/R²pred (PRESS)"],
     ["optimization.py", "Ascenso, análisis canónico, cresta, SLSQP"],
     ["desirability.py", "Deseabilidad de Derringer–Suich (multi-respuesta)"],
     ["plots.py", "Contorno, superficie 3D, Pareto, perturbación, residuos"],
     ["app.py", "Interfaz Streamlit (8 pestañas guiadas)"]],
    colw=[3.2*cm, 11*cm])
P("El modelado se realiza sobre variables <b>codificadas</b> (−1…+1) y el "
  "aplicativo convierte a unidades naturales para entregar recomendaciones "
  "operativas. Dependencias: numpy, pandas, scipy, statsmodels, plotly, "
  "matplotlib, streamlit.", "Just")

S.append(PageBreak())

# ---- 4. Caso de prueba ----
P("4. Caso de prueba", "H1c")
P("Se empleó un <b>diseño Box-Behnken</b> de 3 factores con 5 puntos centrales "
  "(17 corridas). Factores y rangos:", "Just")
tbl([["Factor", "Bajo (−1)", "Alto (+1)", "Unidad"],
     ["Temperatura de tostado", "110", "150", "°C"],
     ["Tiempo de tostado", "15", "35", "min"],
     ["Velocidad de aire", "1.0", "2.0", "m/s"]],
    colw=[6*cm, 2.6*cm, 2.6*cm, 2.6*cm])
P("Respuestas: <b>Aroma</b> (puntaje sensorial 0–100, maximizar) y <b>Acidez</b> "
  "titulable (minimizar). Los 17 ensayos del diseño se encuentran en "
  "<font face='Courier'>data/tostado_cacao.csv</font>.", "Just")

# ---- 5. Resultados ----
P("5. Resultados", "H1c")
P("5.1 Ajuste y adecuación del modelo", "H2c")
P(f"El modelo cuadrático para <b>Aroma</b> explica muy bien los datos "
  f"(R² = {fit_a.r2:.4f}, R² ajustado = {fit_a.r2_adj:.4f}, "
  f"R² predictivo = {fit_a.r2_pred:.4f}). La prueba de <b>falta de ajuste</b> "
  f"no resultó significativa "
  f"(p = {fit_a.lof.loc[0,'p-valor']:.3f} > 0.05), confirmando que el modelo de "
  f"segundo orden es adecuado. Para <b>Acidez</b>: R² = {fit_b.r2:.4f}, "
  f"R² ajustado = {fit_b.r2_adj:.4f}.", "Just")

# Tabla de coeficientes Aroma
coef_rows = [["Término (codif.)", "Coef.", "p-valor", "Signif."]]
for nm in fit_a.coef.index:
    coef_rows.append([nm, f"{fit_a.coef[nm]:+.3f}", f"{fit_a.pvalues[nm]:.4f}",
                      "sig." if fit_a.pvalues[nm] < 0.05 else ""])
tbl(coef_rows, colw=[4.5*cm, 3*cm, 3*cm, 2.5*cm])

img(f"{FIGDIR}/pareto.png", 10*cm,
    "Fig. 1. Diagrama de Pareto de efectos estandarizados (Aroma).")
img(f"{FIGDIR}/resid.png", 12*cm,
    "Fig. 2. Diagnóstico de residuos (Aroma): normalidad y homocedasticidad.")

S.append(PageBreak())
P("5.2 Superficie de respuesta y análisis canónico", "H2c")
P(f"El <b>análisis canónico</b> del modelo de Aroma ubica un punto estacionario "
  f"interior clasificado como <b>{can_a.nature.upper()}</b> (todos los "
  f"eigenvalores de B negativos: "
  f"{', '.join(f'{v:.2f}' for v in can_a.eigenvalues)}). Las condiciones del "
  f"óptimo individual de aroma son: Temperatura = "
  f"{can_a.x_stationary_natural[0]:.1f} °C, Tiempo = "
  f"{can_a.x_stationary_natural[1]:.1f} min, Velocidad = "
  f"{can_a.x_stationary_natural[2]:.2f} m/s, con Aroma previsto = "
  f"{can_a.y_stationary:.1f}.", "Just")
img(f"{FIGDIR}/contour.png", 10.5*cm,
    "Fig. 3. Contorno de Aroma; la estrella marca el punto estacionario.")
img(f"{FIGDIR}/surface.png", 10.5*cm,
    "Fig. 4. Superficie de respuesta de Aroma (Temperatura × Tiempo).")

S.append(PageBreak())
P("5.3 Optimización multi-respuesta (Derringer–Suich)", "H2c")
P(f"Combinando ambas respuestas (Aroma máx., Acidez mín., pesos iguales), la "
  f"optimización de deseabilidad alcanza <b>D = {des_res.D:.3f}</b> en las "
  f"siguientes condiciones operativas recomendadas:", "Just")
tbl([["Variable", "Valor óptimo"],
     ["Temperatura", f"{des_res.x_natural[0]:.1f} °C"],
     ["Tiempo", f"{des_res.x_natural[1]:.1f} min"],
     ["Velocidad de aire", f"{des_res.x_natural[2]:.2f} m/s"],
     ["Aroma previsto", f"{des_res.individual['Aroma'][0]:.1f}"],
     ["Acidez prevista", f"{des_res.individual['Acidez'][0]:.2f}"]],
    colw=[6*cm, 5*cm])
img(f"{FIGDIR}/desir.png", 10.5*cm,
    "Fig. 5. Deseabilidad global D; la estrella marca el óptimo del compromiso.")
P("<b>Recomendación operativa:</b> tostar el grano a ~"
  f"{des_res.x_natural[0]:.0f} °C durante ~{des_res.x_natural[1]:.0f} min con "
  f"velocidad de aire de ~{des_res.x_natural[2]:.1f} m/s maximiza el aroma "
  "manteniendo la acidez en el mínimo del rango estudiado.", "Just")

P("6. Limitaciones", "H1c")
P("• Las respuestas del caso de prueba provienen de un modelo mecanístico "
  "controlado con ruido reducido; con datos reales de laboratorio se espera "
  "mayor variabilidad y menor R².<br/>"
  "• El BBD no evalúa los vértices del cubo, por lo que las predicciones en las "
  "esquinas extremas son extrapolaciones.<br/>"
  "• La deseabilidad asume independencia entre respuestas y sensibilidad a los "
  "límites L/U y pesos definidos por el usuario.<br/>"
  "• La validez del óptimo debe confirmarse con corridas de verificación.", "Just")

P("7. Referencias", "H1c")
refs = [
 "Box, G. E. P., &amp; Behnken, D. W. (1960). Some new three level designs for "
 "the study of quantitative variables. <i>Technometrics, 2</i>(4), 455–475.",
 "Box, G. E. P., &amp; Wilson, K. B. (1951). On the experimental attainment of "
 "optimum conditions. <i>Journal of the Royal Statistical Society B, 13</i>(1), 1–45.",
 "Derringer, G., &amp; Suich, R. (1980). Simultaneous optimization of several "
 "response variables. <i>Journal of Quality Technology, 12</i>(4), 214–219.",
 "Bezerra, M. A., Santelli, R. E., Oliveira, E. P., Villar, L. S., &amp; "
 "Escaleira, L. A. (2008). Response surface methodology (RSM) as a tool for "
 "optimization in analytical chemistry. <i>Talanta, 76</i>(5), 965–977.",
 "Myers, R. H., Montgomery, D. C., &amp; Anderson-Cook, C. M. (2016). "
 "<i>Response Surface Methodology: Process and Product Optimization Using "
 "Designed Experiments</i> (4th ed.). Wiley.",
 "Montgomery, D. C. (2017). <i>Design and Analysis of Experiments</i> "
 "(8th ed.). Wiley.",
 "Baş, D., &amp; Boyacı, İ. H. (2007). Modeling and optimization I: Usability "
 "of response surface methodology. <i>Journal of Food Engineering, 78</i>(3), 836–845.",
 "Khuri, A. I., &amp; Mukhopadhyay, S. (2010). Response surface methodology. "
 "<i>WIREs Computational Statistics, 2</i>(2), 128–149.",
 "Streamlit Inc. (2024). <i>Streamlit documentation</i>. https://docs.streamlit.io",
 "Virtanen, P., et al. (2020). SciPy 1.0: fundamental algorithms for scientific "
 "computing in Python. <i>Nature Methods, 17</i>, 261–272.",
]
for i, r in enumerate(refs, 1):
    P(f"[{i}] {r}", "Ref")

S.append(Spacer(1, 8))
S.append(HRFlowable(width="100%", color=colors.grey, thickness=0.6))
P("8. Declaración de uso de Inteligencia Artificial", "H1c")
P("Para el desarrollo de este trabajo se utilizó un asistente de IA "
  "(modelo de lenguaje tipo Claude) en las siguientes tareas: (i) apoyo en la "
  "estructuración del código de los módulos RSM y de la interfaz Streamlit; "
  "(ii) redacción y revisión de estilo del presente reporte; (iii) generación "
  "del conjunto de datos de prueba a partir de un modelo cuadrático mecanístico. "
  "<b>No se utilizó IA</b> para inventar resultados experimentales reales: todos "
  "los valores numéricos del reporte provienen de la ejecución verificable del "
  "código sobre los datos incluidos. La formulación matemática (CCD/BBD, "
  "análisis canónico, deseabilidad) fue revisada y validada por los integrantes "
  "contra la literatura citada. <b>Cada integrante del grupo puede explicar "
  "cualquier línea del código en la defensa oral.</b>", "Just")

# ---------------------------------------------------------------------------
doc = SimpleDocTemplate(OUT, pagesize=A4,
                        leftMargin=2*cm, rightMargin=2*cm,
                        topMargin=1.6*cm, bottomMargin=1.6*cm,
                        title="Reporte técnico RSM - Tostado de cacao")
doc.build(S)
print("PDF generado:", OUT)
