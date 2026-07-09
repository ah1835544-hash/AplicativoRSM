"""
Paquete rsm
===========

Implementacion de los metodos de Metodologia de Superficie de Respuesta (RSM):
  - Diseno experimental (CCD, Box-Behnken)          -> designs.py
  - Ajuste del modelo y ANOVA                        -> models.py
  - Optimizacion (ascenso, canonico, cresta, num.)   -> optimization.py
  - Deseabilidad de Derringer-Suich                  -> desirability.py
  - Visualizacion                                    -> plots.py

Aplicativo: Optimizacion RSM para el sector agroindustrial ecuatoriano.
"""

from . import designs, models, optimization, desirability, plots

__all__ = ["designs", "models", "optimization", "desirability", "plots"]
__version__ = "1.0.0"
