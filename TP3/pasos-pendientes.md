# TP3: secuencia de pasos pendientes

_Consigna: `TP3/consigna/Consigna TP3.pdf`. Entrega 18/11/2026 (campus hasta el 19/11 23:59). Actualizado
2026-11-16 tras la sesión de Etapa C (dashboard); antes, 2026-11-09 por la de Etapa A/B._

_Para el detalle hallazgo por hallazgo de la revisión del pipeline (bugs encontrados, decisiones de feature
engineering, hipótesis conversadas) ver `revision-pipeline-error.md` -este documento es el resumen de estado
general, no reemplaza ese detalle._

## Qué ya está hecho

| Notebook | Qué produce | Archivo exportado |
|---|---|---|
| `1_analisis_estaciones.ipynb` | Selección y filtrado de las estaciones de la región central | `data/processed/estaciones_central.geojson`, `estados_central.geojson` |
| `2_relaciones-bivariadas.ipynb` | Limpieza de `pronosticos-observaciones.csv`, variable de error con signo y absoluta para temp/HR/dewpoint/viento (**pronosticado − observado**), EDA bivariado | `data/processed/mediciones_limpias.csv` |
| `3_analisis_temporal.ipynb` | Panel regularizado estación×hora de `obs_*` (imputación acotada, feature engineering: rolling 3h, encoding cíclico, vecinas); cálculo de `error_temp_c`, y sobre esa serie: ADF/KPSS, ACF/PACF, CCF contra variables de sensores, y su propio feature engineering (lags 6/24, rolling, viento, humedad/dewpoint/windspeed, vecinas, todo anclado en `shift(6)`) | `panel_temporal_limpio.csv`, `panel_features.csv`, `panel_features_error_temp_c.csv` |
| `4_atributos_geograficos.ipynb` | Distancia a agua/aeropuerto/área urbana, bioma, tests de hipótesis (replican/amplían TP2) | `data/processed/estaciones_atributos.csv` |
| `5_separacion_train_test.ipynb` | Dataset único de modelado: une `panel_features.csv` + `mediciones_limpias.csv` + `estaciones_atributos.csv`, define `target_valido_<var>` por variable, separa train/test por fecha | `data/processed/dataset_modelado.csv` |
| `6_baselines_globales.ipynb` | Baselines de `temp_c`: cero, naive, naive estacional, media móvil, drift | `data/out/modelado/` |
| `7_modelado_global.ipynb` ✅ **re-ejecutada 2026-11-15** | Etapa A: entrena y compara 5 familias de modelos sobre `error_temp_c` (Ridge/Lasso/ElasticNet, KNN, árbol, Random Forest, Gradient Boosting), con `TimeSeriesSplit` + `HalvingRandomSearchCV` para RF/GB, y elige los 2 modelos finales -ahora con dataset extendido, `CORTE` corregido, `fcst_rh_pct`/`fcst_windspeed_kmh` sumadas (hallazgo 19) y el gate de hora categórica (hallazgo 18, descartado) | `data/processed/modelos/modelo_interpretable_temp_c.joblib`, `modelo_min_error_temp_c.joblib`, `data/out/modelado_global/` |
| `8_interpretabilidad.ipynb` ✅ **re-ejecutada 2026-11-15** | Etapa B: carga los 2 modelos reales (no placeholders), feature importance, SHAP, sesgo pre/post corrección por hora y por bioma, diagnóstico de la ventana 10h-14h -ahora sin "empeora", ver sección 3 abajo | `data/out/interpretabilidad/` |

## Secuencia de pasos

### 1. Dataset de modelado — ✅ hecho
Ver `5_separacion_train_test.ipynb`: unión de los 3 archivos, target por variable con `target_valido_<var>`,
separación train/test temporal (corte fijo, sin gap porque todas las features miran sólo hacia atrás).

### 2. Etapa A — Modelado de Machine Learning (entregables 1 y 2) — ✅ hecho (re-ejecutada 2026-11-15)
- **Baselines** (`6_baselines_globales.ipynb`): ninguno le gana a "cero" (no corregir) -MAE test 1,436°C (antes
  1,241°C; el `CORTE` se corrigió, hallazgo 20), el mejor baseline (naive estacional) da 1,591°C (antes 1,499°C).
  Hallazgo esperado, no una falla del pipeline (ver hallazgo 8 de `revision-pipeline-error.md`). El train/test MAE
  de los baselines converge mucho más que antes -confirma que el `MAE_test` actual es representativo.
- **`7_modelado_global.ipynb`**: los 5 modelos entrenados sí le ganan a "cero", con una mejora relativa *mayor*
  que antes de extender el dataset. Resultado final:

  | Modelo | MAE test | Mejora vs. "cero" | MASE |
  |---|---|---|---|
  | **Random Forest** (mínimo error, entregable 2) | 1,141 | **+20,5%** | 0,638 |
  | Gradient Boosting | 1,159 | +19,3% | 0,648 |
  | KNN | 1,203 | +16,2% | 0,672 |
  | **Árbol de decisión** (interpretable, entregable 1) | 1,215 | +15,4% | 0,679 |
  | Ridge | 1,245 | +13,3% | 0,695 |
  | ElasticNet | 1,268 | +11,7% | 0,708 |
  | Lasso | 1,270 | +11,6% | 0,709 |

  (antes: RF 1,067/+14,0%, GB 1,069/+13,8%, árbol 1,135/+8,5%, KNN 1,148/+7,5%, Lasso/ElasticNet 1,154/+7,0%,
  Ridge 1,173/+5,5% -el MAE absoluto subió porque el nuevo test, más largo y representativo, cae en un tramo más
  frío/difícil del año, pero la mejora *relativa* contra no corregir aumentó en todos los modelos; también se
  reordenó el medio de la tabla: Ridge ahora le gana a Lasso/ElasticNet, y KNN a el árbol). Hiperparámetros
  ganadores documentados en `data/out/modelado_global/tabla_comparacion_modelos_temp_c.csv` (columna
  `mejores_parametros`); `cv_MAE_std` (hallazgo 20) nueva, dispersión ~7-9% entre folds para todos los modelos.
- **Evaluación** (MAE/RMSE pronóstico original vs. corregido): cubierta -el MAE del residuo del modelo equivale al
  MAE del pronóstico corregido contra la observación.
- Se auditó la notebook (`auditoria.md`) y se aplicaron 2 de los 5 puntos propuestos: reintegrar
  `obs_dewpoint_c`/`obs_windspeed_kmh` para los modelos de árbol, y ensanchar los grids de hiperparámetros
  (`HalvingRandomSearchCV` para RF/GB). También se sumaron `fcst_temp_c`/`fcst_pop_pct` como features -no estaban
  incluidas originalmente y resultaron de las más importantes para los modelos de árbol (`fcst_pop_pct` quedó 3º en
  SHAP de Random Forest).
- **Sesión 2026-11-15**: se sumaron `fcst_rh_pct`/`fcst_windspeed_kmh` (hallazgo 19) -Ridge les da coeficiente alto
  (6º y 7º en magnitud de 25 features)-, y se probó (y descartó) `hora_local` categórica contra `hora_sin`/`hora_cos`
  para los 3 modelos de árbol (hallazgo 18): la cíclica sigue ganando.

### 3. Etapa B — Interpretabilidad — ✅ hecho (re-ejecutada 2026-11-15)
`8_interpretabilidad.ipynb`, con los modelos reales de la corrida nueva (dataset extendido, `fcst_rh_pct`/
`fcst_windspeed_kmh` incluidas):
- Feature importance del árbol (interpretable) y SHAP de Random Forest (mínimo error): siguen coincidiendo en que
  `error_temp_c_lag24`, `error_temp_c_lag6` y `hora_sin`/`hora_cos` son lo más importante. `fcst_rh_pct` (nueva)
  entra fuerte -en SHAP le gana a `fcst_pop_pct` (0,116 vs. 0,102), que hasta la corrida anterior era "el
  pronóstico que más aportaba" (3º en SHAP)-; `fcst_windspeed_kmh` (nueva) entra justo detrás.
- Sesgo capturado: **sí por bioma** (los 5 biomas mejoran con ambos modelos, incluido el más difícil,
  *Northwestern Forested Mountains*), **y ahora también sí por hora, sin excepción** -**cambio real respecto a la
  corrida anterior**: antes el modelo empeoraba un poco en la ventana 10h-14h/20h; con el dataset extendido, el
  residuo queda por debajo de no corregir en las 24 horas del día. Lo que persiste es que mejora *menos* en esa
  ventana y en la noche (19h-23h, +10,5% a +18,3%) que en la mañana (+26% a +39%) -la causa (colapso de la
  correlación de `error_temp_c_lag6` en el cruce por cero del ciclo diario) es la misma que antes, sólo que ya no
  alcanza para que el modelo quede peor que el pronóstico sin corregir. La mejora global subió de 14,0% a 21,0%
  (Random Forest) -detalle actualizado en el hallazgo 9 de `revision-pipeline-error.md`.
- El caso de SHAP local (mayor residuo en la ventana 10h-13h) cambió de estación/fecha/signo respecto a la corrida
  anterior, pero ilustra la misma limitación de fondo desde otro ángulo: antes ninguna feature de historia
  reciente anticipaba la anomalía; ahora `error_temp_c_lag6` sí tenía señal, pero apuntaba en la dirección
  equivocada. Ambos casos van en el informe como la misma limitación (anomalías puntuales sin aviso en el feature
  set actual), no como hallazgos separados.

### 4. Etapa C — Dashboard (entregable 4) — ✅ hecho (2026-11-16)

**El diseño, las decisiones de visualización y la justificación de cada panel están en
`plan-dashboard.md`** -este resumen no lo reemplaza. Lo hecho:

1. **Herramienta**: Streamlit + Altair (ambas instaladas en esta sesión, junto con `pyarrow`).
2. **`dashboard/precomputo.py`**: carga los 2 `.joblib`, predice una vez y deja 3,5 MB en
   `data/processed/dashboard/` (parquet de las 47.151 filas de test con las dos correcciones, atributos de
   estación, `metricas.json`, SHAP de la muestra, contorno de estados simplificado). Existe porque el modelo de
   mínimo error pesa 1,1 GB: el dashboard no puede cargarlo en vivo. Validación: los MAE que imprime coinciden al
   cuarto decimal con `data/out/interpretabilidad/tabla_resultados.csv`.
3. **`dashboard/app.py`**: tres niveles (Resumen guiado / Explicabilidad / Detalle técnico) según la Clase 13,
   con la vista de monitoreo (observado, pronóstico y corregido, más el error con signo, por estación) y los
   filtros de la consigna -franja horaria, lluvia vía `fcst_pop_pct`, y ubicación por estado/bioma- dentro de un
   formulario con botón "Aplicar". Verificado con `streamlit.testing.v1.AppTest`: 10 caminos sin excepción,
   incluido el de selección vacía.
4. **Incertidumbre** (no estaba en el pipeline y la consigna la pide): banda del 80% split-conformal, calibrada
   en los primeros 2 días del test y medida en los 3 restantes -cobertura observada 81,3% contra 80% nominal.
   Calibrarla con residuos de train daba 55,4%: ver `plan-dashboard.md`, sección 5.

Para correrlo: `python TP3/dashboard/precomputo.py` y después
`python -m streamlit run TP3/dashboard/app.py`.

### 5. Informe (entregable 3) — 🔴 Pendiente, insumos ya listos
Redactar comparando los dos modelos: performance (tabla de la Etapa A) e insight de interpretabilidad de cada uno
(Etapa B -feature importance, SHAP, sesgo por hora/bioma, el hallazgo del mediodía). Todos los números y gráficos
ya están generados en `data/out/modelado_global/` y `data/out/interpretabilidad/`, sólo falta la redacción.

## Ideas identificadas pero no implementadas (quedan para si hay tiempo)

De `auditoria.md`, puntos 1, 3 y 4 -ver análisis de viabilidad completo en la conversación de esta sesión:
- **Punto 1** (recomputar el target sólo contra `obs_temp_c` real, sin interpolación): válido metodológicamente,
  impacto empírico chico (13,3% vs. 13,5% de mejora si se excluyen las filas con `obs` imputado) -bajo costo si se
  quiere por rigor del informe.
- **Punto 3** (features espaciales direccionales, gradiente de error entre estaciones): viable pero no trivial -K=2
  vecinas no alcanza para un gradiente confiable, densidad de estaciones muy dispareja por bioma.
- **Punto 4** (ingesta de datos externos vía S3/Glue/PySpark): **descartado** -describe infraestructura que no
  existe en este proyecto (100% pandas local) y está fuera de alcance dado el tiempo restante hasta la entrega.

## Archivos clave para retomar

- `TP3/revision-pipeline-error.md`: historial detallado de hallazgos, bugs resueltos y decisiones de esta sesión
  (9 hallazgos documentados al 2026-11-09).
- `TP3/auditoria.md`: los 5 puntos de la auditoría externa, con veredicto de viabilidad para cada uno.
- `TP3/data/processed/modelos/modelo_interpretable_temp_c.joblib`, `modelo_min_error_temp_c.joblib`: los 2 modelos
  finales entrenados, listos para el dashboard o para re-evaluar sin reentrenar.
- `TP3/data/out/modelado_global/`, `TP3/data/out/interpretabilidad/`: tablas y gráficos ya generados, insumo
  directo para el informe (entregable 3).
- `TP3/notebooks/utils.py`: paleta y helpers de gráficos compartidos por todas las notebooks.
- `TP3/plan-dashboard.md`: el diseño del dashboard (Etapa C) derivado de la Clase 13, con la justificación de cada
  panel, los filtros que se descartaron y por qué, y las decisiones de visualización y de incertidumbre.
- `TP3/dashboard/`: `precomputo.py` (genera la fuente de datos), `app.py` (el tablero) y `paleta.py` (colores
  compartidos con los notebooks). Los parquet/json que consume no se versionan: los regenera `precomputo.py`.
