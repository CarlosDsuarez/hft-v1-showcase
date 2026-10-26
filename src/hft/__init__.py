"""Núcleo intradía HFT-lite (barras de 1–5 min).

Alcance: trading sistemático intradía sobre Alpaca paper. Las decisiones
usan barras de 1 a 5 minutos y el libro queda plano al cierre de la sesión.
Backtest, simulación y paper comparten la misma ruta de decisión.

No-objetivos: market making real (rebates, prioridad de cola), cualquier
edge que dependa de latencia sub-segundo, y modelos de ML o LLM.

Reglas:

1. Sin IA. Prohibidos los LLM y las APIs de IA, los modelos de ML
   entrenados (sklearn, xgboost, torch y equivalentes), los optimizadores
   de caja negra y la búsqueda automática de parámetros. Permitida la
   estadística clásica (medias, varianzas, OLS, AR(1), z-scores, ADF) con
   parámetros fijados de antemano y justificados en la TheoryCard.

2. Ligero. Este paquete solo usa la biblioteca estándar, numpy y pandas.
   No importa, ni de forma transitiva a través de ``src.*``, cvxpy, scipy,
   sklearn, yfinance, plotly, matplotlib ni pyarrow. Una dependencia nueva
   exige aprobación explícita.

3. Causalidad. Una barra marcada T (inicio del intervalo) solo es
   utilizable desde T + timeframe + gracia. La decisión en t usa barras
   finales ``<= t``. La ejecución ocurre como pronto en t+1, más la
   latencia configurable. Cada feature nueva trae un test de truncamiento.

4. Una sola ruta. Backtest, simulación en vivo y paper llaman a
   ``Engine.step`` con la misma estrategia. Solo cambian el proveedor de
   datos y el adaptador de ejecución.

5. Pesimismo. Un límite pasivo se llena solo si el precio cruza el límite.
   Si el stop y el take caben en la misma barra, se asume el stop primero.
   Las tarifas regulatorias van incluidas. El backtest nunca es más
   optimista que el paper.
"""
