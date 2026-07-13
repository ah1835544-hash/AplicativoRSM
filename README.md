# Aplicativo de resolución RSM — Optimización agroindustrial (Ecuador)

Aplicativo funcional e interactivo (Python + Streamlit) que integra los métodos
de **Metodología de Superficie de Respuesta (RSM)** revisados en clase, aplicado
a un caso real del sector agroindustrial ecuatoriano: **optimización del tostado
de cacao Nacional**.

Permite a un usuario **no especialista**: cargar datos, generar el diseño
experimental, ajustar y diagnosticar el modelo, y obtener **recomendaciones
operativas concretas** (condiciones óptimas de proceso).

> Asignatura: Optimización — Universidad Central del Ecuador (UCE).
> Aplicación: resolución RSM. Uso académico.

---

## 1. Métodos integrados

| Bloque | Métodos | Módulo |
|---|---|---|
| **Diseño experimental** | Central Compuesto (CCD, α rotable/caras/ortogonal) y Box-Behnken (BBD) | `rsm/designs.py` |
| **Ajuste del modelo** | 1er y 2do orden, ANOVA, **prueba de falta de ajuste** (error puro con réplicas), R², R² ajustado, R² predictivo (PRESS), análisis de residuos | `rsm/models.py` |
| **Optimización** | Ascenso/descenso más pronunciado, **análisis canónico** (punto estacionario y eigenvalores), **análisis de cresta** (ridge), optimización numérica (SLSQP) | `rsm/optimization.py` |
| **Múltiples respuestas** | Función de **deseabilidad de Derringer–Suich** | `rsm/desirability.py` |
| **Visualización** | Gráficos de **contorno**, **superficies 3D**, **diagrama de Pareto** y **gráfico de perturbación**, diagnóstico de residuos | `rsm/plots.py` |

---

## 2. Instalación y ejecución (local)

Requiere **Python 3.10+**.

```bash
# 1) (opcional) crear entorno virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 2) instalar dependencias
pip install -r requirements.txt

# 3) ejecutar el aplicativo
streamlit run app.py
```

Se abre en el navegador (por defecto `http://localhost:8501`).

### Windows sin `python` en el PATH
Si `python` no se reconoce, use la ruta completa del intérprete, por ejemplo:

```powershell
$py = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
& $py -m pip install -r requirements.txt
& $py -m streamlit run app.py
```

### Prueba rápida (sin interfaz)
```bash
python self_test.py
```
Debe imprimir los resultados del análisis y `TODAS LAS PRUEBAS PASARON`.

---

## 3. Uso del aplicativo (flujo)

1. **Barra lateral → Definir factores**: nombre, nivel bajo (−1), alto (+1) y unidades.
2. **Pestaña 1 · Diseño**: elija CCD o BBD, puntos centrales y α; genere y **descargue el plan** experimental.
3. **Pestaña 2 · Datos**: suba el CSV con sus resultados o pulse **“Usar datos demo (cacao)”**. Mapee columnas de factores y respuestas.
4. **Pestaña 3 · Modelo y ANOVA**: elija el orden (1/2); revise coeficientes, ANOVA, **falta de ajuste** y R².
5. **Pestaña 4 · Diagnóstico**: Pareto, perturbación y residuos.
6. **Pestaña 5 · Visualización**: contorno y superficie 3D interactivos.
7. **Pestaña 6 · Optimización**: ascenso pronunciado / canónico / cresta / numérico.
8. **Pestaña 7 · Multi-respuesta**: defina objetivos (max/min/target), límites y pesos; obtenga las **condiciones óptimas** por deseabilidad.
9. **Pestaña 8 · Reporte**: exporte un resumen `.txt`.

---

## 4. Datos de prueba

`data/tostado_cacao.csv` — Diseño **Box-Behnken (3 factores, 5 centros = 17 corridas)**:

| Factor | Bajo (−1) | Alto (+1) | Unidad |
|---|---|---|---|
| Temperatura de tostado | 110 | 150 | °C |
| Tiempo de tostado | 15 | 35 | min |
| Velocidad de aire | 1.0 | 2.0 | m/s |

Respuestas: **Aroma** (puntaje sensorial 0–100, *maximizar*) y **Acidez**
titulable (*minimizar*).

---

## 5. Estructura del proyecto

```
AplicativoRSM/
├── app.py                  # Aplicativo Streamlit (interfaz)
├── self_test.py            # Prueba de humo de los módulos numéricos
├── requirements.txt
├── README.md
├── reporte_tecnico.pdf     # Reporte técnico (≤10 págs) + declaración de IA
├── .streamlit/config.toml
├── data/
│   └── tostado_cacao.csv   # Datos de prueba (BBD)
└── rsm/
    ├── __init__.py
    ├── designs.py          # CCD, Box-Behnken
    ├── models.py           # OLS, ANOVA, falta de ajuste, R²/PRESS
    ├── optimization.py     # ascenso, canónico, cresta, numérico
    ├── desirability.py     # Derringer–Suich
    └── plots.py            # contorno, superficie, Pareto, perturbación, residuos
```

---

## 6. Notas de reproducibilidad

- Modelado sobre **variables codificadas** (−1…+1); el aplicativo convierte
  automáticamente a unidades naturales para las recomendaciones.
- La **falta de ajuste** se calcula sólo si el diseño tiene réplicas
  (p. ej. puntos centrales), para estimar el error puro.
- El **R² predictivo** se obtiene por PRESS (validación *leave-one-out*).
- Optimización numérica y de deseabilidad con **múltiples reinicios** para evitar
  óptimos locales.

---

## 7. Declaración de uso de IA
Este proyecto utilizó herramientas de IA generativa exclusivamente como asistentes de coprogramación para optimizar arreglos vectoriales en NumPy y estructurar la lógica UX/UI en Streamlit. Las interpretaciones estadísticas, la verificación de los signos de los eigenvalores frente a los criterios físicos del proceso y la redacción del informe técnico final fueron auditadas, corregidas y validadas en su totalidad por los autores del grupo.

## 8. Licencia
Uso académico. © Grupo J — UCE.
