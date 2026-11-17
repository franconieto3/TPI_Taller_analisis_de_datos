# TP3 · Auditoría integral de entrega (2026-11-17)

_Alcance: lectura estática del repo en claude/tender-turing-e949p2 (commit 9e102cf). TP3/data/, consigna/ y
apuntes/ no están versionados (.gitignore), así que no se re-ejecutó nada: las cifras salen de los outputs
guardados en las notebooks y de los .md de seguimiento. Entrega: 18/11 (campus hasta el 19/11 23:59)._

## 1. Resumen ejecutivo

*Estado original (2026-11-17): 🔴 Falta un entregable crítico (informe técnico). El resto del trabajo está 🟡 con
ajustes menores.*

*Actualización 2026-11-16: los tres bloqueantes están cerrados -informe redactado (v1), dashboard ejecutable por
un tercero y cifras unificadas-, más E4, E8 y E9. Ver las filas marcadas ✅ abajo.*

- *Etapa A (modelado)*: sólida. Target bien definido, 5 baselines explícitos, TimeSeriesSplit en el tuning,
  holdout cronológico 81/19, 7 modelos comparados con MAE/RMSE/MASE. Random Forest (MAE 1,141 °C, −20,96 % vs.
  no corregir) y árbol de decisión (1,215 °C, −15,85 %; cifras unificadas el 2026-11-16, ver E3). Las 4 notebooks clave (5-8) no tienen celdas con error ni
  celdas sin ejecutar.
- *Etapa B (interpretabilidad)*: buena cobertura (impureza, permutación, SHAP global + local, sesgo pre/post por
  hora y bioma; el árbol explicativo y los coeficientes de Ridge se agregaron en la §8 de la nb 8 el 2026-11-16).
  La §9 (2026-11-16) cerró las tres brechas que quedaban: SHAP del modelo interpretable sobre el test completo,
  3 dependence plots y el análisis pre/post de lluvia y viento. Único resto: el beeswarm del RF sigue con n=100.
- *Etapa C (dashboard)*: completo y bien fundamentado (vista tripartita, banda conformal 80 % con cobertura
  81,3 %, filtros en st.form). Brechas: el umbral de lluvia es PoP ≥ 50 % (la consigna pide ≥ 30 % de ejemplo),
  y la hora sólo se filtra por franjas de 6 h. *Ejecutable por un tercero desde el 2026-11-16*: datos versionados
  + requirements.txt, verificado con un clon limpio.
- *Informe (entregable 3)*: 🟡 primera versión en `TP3/informe.md` (2026-11-16), pendiente de revisión del grupo.

## 2. Matriz de control de entregables oficiales

| Entregable / Requisito | Estado | Hallazgo / Brecha | Acción correctiva recomendada |
|---|---|---|---|
| 1. Modelo interpretable (pipeline + CV + métricas) | 🟢 *(resuelto 2026-11-16)* | 7_modelado_global.ipynb: árbol max_depth=8, min_samples_leaf=50 elegido entre Ridge/Lasso/ElasticNet/árbol; son 232 hojas (no 256). La §8 nueva de 8_interpretabilidad.ipynb agrega las dos lecturas que faltaban: **árbol explicativo** (plot_tree/export_text con max_depth=3 → `08_arbol_explicativo_d3.png`) y **Ridge** alpha=1000 re-ajustado, MAE 1,245 (→ `09_coeficientes_ridge.png`, `tabla_coeficientes_ridge.csv`) | Nada pendiente en código. Para el informe: (a) `max_depth=3` trunca *la visualización, no el modelo*, así que no hay que declarar ningún modelo alternativo -sólo que las ramas grises están colapsadas-; (b) los cortes de hora están sobre `hora_sin`/`hora_cos` y hay que traducirlos (la §8 incluye la tabla de equivalencias); (c) la cifra de mejora del árbol es 15,85 % con `cero = 1,4435`, ya unificada en la nb 7 (E3 resuelto) |
| 2. Modelo de precisión (pipeline + CV + métricas) | 🟢 | Random Forest con HalvingRandomSearchCV + TimeSeriesSplit(5), MAE test 1,141, MASE 0,638, cv_MAE_std reportado | Mencionar sobreajuste (MAE train 0,661 vs. test 1,141) y que GB queda a 0,018 °C con un modelo ~100× más liviano |
| 3. Informe técnico (precisión vs. interpretabilidad) | 🟡 *(v1 escrita 2026-11-16)* | `TP3/informe.md`: 12 secciones + 2 anexos, con convención de signo, datos y split, baselines (incluido el MOS nuevo), los 7 modelos, interpretabilidad de cada uno, sesgos pre/post (hora, bioma, lluvia, viento), banda conformal, tradeoff y limitaciones | Falta la revisión del grupo. Pendiente de decisión: formato de entrega (el .md o un PDF) |
| 4. Dashboard funcional | 🟢 *(resuelto 2026-11-16)* | Los 3,5 MB de data/processed/dashboard/ quedaron versionados con una excepción acotada en .gitignore, y se agregaron dashboard/requirements.txt (sólo lo que app.py necesita) y requirements-precomputo.txt. Verificado con un clon limpio: los 6 archivos bajan intactos y app.py los encuentra | Nada bloqueante. Opcional: desplegar en Streamlit Community Cloud y entregar también el link |
| Target = error del pronóstico | 🟢 | error_temp_c = fcst − obs (convención inversa a la de la consigna, obs − fcst), pero consistente: fcst_corregido = fcst − ê en precomputo.py | Declarar la convención en la primera página del informe y en el dashboard (ya está en el caption) para evitar que el jurado lea el signo al revés |
| Baselines explícitos | 🟢 | 6_baselines_globales.ipynb: cero, naive, naive estacional, media móvil, drift + MASE; más el baseline MOS (sesgo medio estación × hora) agregado en nb 8 §9.1 | E3 y E4 resueltos: los baselines se recalculan sobre la población de los modelos (nb 7 §12) y el MOS da 1,330, que sí le gana a no corregir |
| Validación temporal | 🟢 | Split por fecha (CORTE = 2026-11-10), CV con TimeSeriesSplit, assert de orden cronológico; features ancladas en shift(6) | Ver Error E5 (folds partidos por filas, no por timestamp) |
| Comparación Error NWS vs. Refinado | 🟢 | Barras MAE (nb 7), MAE por hora/bioma pre-post (nb 8), KPIs + series en dashboard | — |
| SHAP / Feature Importance en ambos modelos | 🟢 *(resuelto 2026-11-16)* | nb 8 §9.3: TreeExplainer sobre el árbol con las **47.151 filas de test** (2,1 s) + 3 dependence plots. El RF conserva impureza, permutación y SHAP con n=100 | Nada bloqueante. Para subir las 100 filas del RF haría falta un modelo más liviano (GB); declarado como limitación en el informe (§11) |
| Mitigación de sesgos del TP2 | 🟢 *(resuelto 2026-11-16)* | Diurno ✅ (24/24 horas mejoran). Lluvia y viento ✅ en nb 8 §9.2: los modelos mejoran en los 5 cortes, y el pronóstico base resulta **más preciso** con lluvia y con viento alto, no menos | El viento *como variable objetivo* queda fuera de alcance (el TP3 modela temp_c), declarado en el informe §8.4 junto con la cautela del n = 2.025 de lluvia |

## 3. Análisis detallado por etapa

### Etapa A — ML pipeline y corrección de sesgo

*Target y baselines.* El target se calcula en la nb 3 y se consume desde dataset_modelado.csv; el hallazgo 1
(ambigüedad del merge) está corregido con drop(columns=COL_ERROR) antes del merge en nb 7, nb 8 y
precomputo.py. La corrección queda

$$\hat{T}{\text{corr}} = T{\text{fcst}} - \hat{e}, \qquad e = T_{\text{fcst}} - T_{\text{obs}}$$

equivalente a la fórmula de la consigna con $\hat{E} = -\hat{e}$. Resultado clave para el informe: **ningún
baseline de persistencia le gana a no corregir** (cero 1,436 vs. naive estacional 1,591), y los 7 modelos sí.
_Actualización 2026-11-16 (E4): el baseline MOS -sesgo medio por estación × hora- **sí le gana** (1,330 vs.
1,4435), y los dos entregables le ganan a él por 8,7 % y 14,2 %._ El MASE usa
como escala la persistencia $e_{t-6}$ (1,790 °C), así que

$$\text{MASE}_{RF} = \frac{1{,}141}{1{,}790} = 0{,}638$$

lo que significa que el modelo reduce un 36 % el error de "repetir el último error conocido".

*Feature engineering.*

| Requisito | Presente | Comentario |
|---|---|---|
| Cíclicas de hora | ✅ hora_sin/cos | Ablation contra one-hot (8bis): la cíclica gana en los 3 modelos de árbol |
| Cíclica de mes | ❌ justificado | 27 días de datos: un mes_sin/cos sería casi constante. Decirlo explícitamente en el informe |
| Lags | ✅ lag6, lag24 | Elegidos con PACF; lag48 descartado por redundante (hallazgo 5) |
| Medias móviles | ✅ roll3_mean | Anclado en shift(6) |
| Humedad | ✅ obs_rh_pct_lag6, fcst_rh_pct | Respaldado por CCF diferenciada (hallazgo 6) |
| Presión | ❌ | No existe en las fuentes (nb 7 lo declara). Correcto |
| PoP | ✅ fcst_pop_pct | |
| Elevación / geografía | ✅ | elevación, distancias, bioma, urbano/agua |
| Espaciales | ✅ vecinas_lag6_mean | Gradiente direccional evaluado y descartado con evidencia (hallazgo 16) |

*Validación temporal.* No hay fuga: el tuning sólo ve X_train, el test se evalúa una vez, y toda feature
dinámica mira al menos 6 h hacia atrás (el horizonte del pronóstico). Dos matices para declarar, no para corregir:
(a) el test también decide qué familia gana (hallazgo 10); (b) un único holdout de 5 días, mitigado con
cv_MAE_std (7-9 % entre folds).

### Etapa B — Interpretabilidad y sesgos

- *Factores de falla, bien identificados*: historia del error (lag24, lag6) y hora del día dominan en los
  tres métodos; fcst_rh_pct y fcst_pop_pct son las covariables de pronóstico más relevantes; la geografía es
  marginal. La agrupación por bloques del dashboard (árbol → historia del error; RF → hora) es un hallazgo bueno para
  el informe.
- *Mecanismo del mediodía*: diagnosticado con evidencia (colapso de $\rho(e_{t-6}, e_t)$ en 10-13 h). La
  mejora es menor ahí y de noche (+10,5 a +18,3 %) que a la mañana (+26 a +39 %).
- *SHAP local*: caso E7246 (residuo 8,7 °C) bien interpretado como anomalía no anticipable.
- *Brechas, estado al 2026-11-16*: (1) SHAP del árbol ✅ resuelto (nb 8 §9.3, test completo); (2) dependence
  plots ✅ resueltos (los 3 pedidos, sobre el SHAP del árbol); (3) el beeswarm del **Random Forest** sigue con 100
  filas sobre 47.151 y sigue siendo estadísticamente pobre -subirlo exige un modelo más liviano, queda declarado
  como limitación en el informe §11-; (4) *lluvia* ✅ resuelta (nb 8 §9.2, informe §8.4): el modelo mejora un
  18,5 % en horas con pop ≥ 30 %, aunque son sólo 2.025 filas; (5) *viento*: el TP3 sólo modela temp_c, así que
  el sesgo de sobreestimación del viento del TP2 *no se corrige* -queda declarado como fuera de alcance, y sí se
  midió el error de temperatura condicionado a viento alto (+17,6 % de mejora, contra +23,0 % con viento bajo).

### Etapa C — Dashboard

| Requisito | Estado | Detalle |
|---|---|---|
| Vista tripartita obs/fcst/corregido | ✅ | Resumen, panel por estación con error con signo apilado |
| Rango horario | 🟡 | Sólo 4 franjas de 6 h (multiselect). No se puede pedir "7-9 h", justo la ventana de peor pronóstico |
| Días de lluvia | 🟡 | PoP ≥ 50 %, no el 30 % del ejemplo de la consigna, y el umbral está hardcodeado en 3 lugares (aplicar_filtros, etiquetas) |
| Ubicación | ✅ | Estado y bioma; estación en el selector del Resumen |
| Incertidumbre | ✅ | Split-conformal 80 % por (bioma, franja), cobertura 81,3 % medida fuera de calibración. Muy buen punto para el informe |
| Robustez | ✅ (declarado) | AppTest con 10 caminos según pasos-pendientes.md; no pude re-verificar (sin datos ni streamlit en este entorno) |
| Reproducibilidad | 🔴 | Ver entregable 4 |

## 4. Matriz de errores y outliers

| # | Módulo | Descripción del problema | Gravedad | Solución recomendada |
|---|---|---|---|---|
| E1 | Informe | Entregable obligatorio inexistente a 1 día de la entrega | *Alta* | Redactarlo primero, antes de cualquier mejora técnica |
| E2 | Dashboard / git | Parquet/json del tablero y modelos fuera de git; precomputo.py depende de un .joblib de 1,1 GB. El jurado no puede ejecutarlo | *Alta* | Subir data/processed/dashboard/ (3,4 MB) con una excepción en .gitignore, requirements.txt con versiones fijas, o deploy en Streamlit Cloud |
| E3 | nb 6 → nb 7 | ✅ **RESUELTO (2026-11-16)** en la §12 nueva de nb 7, que recalcula los baselines sobre X_test y reescribe el CSV y la figura **sin reentrenar nada** (~8 s; re-ejecutar la nb 7 entera habría costado 2-4 h). Corrección del enunciado original: los baselines de la nb 6 se midieron sobre **32.214** filas de test, no 54.243 -su máscara exige los 4 lags-, contra las 47.151 de los modelos | *Media* | Hecho. Las mejoras subieron +0,4 pp de forma uniforme (RF 20,5 → 20,96 %, árbol 15,4 → 15,85 %) y ningún ranking cambió. El MASE ya estaba bien: la nb 7 calculaba la persistencia sobre X_test |
| E4 | nb 6 | ✅ **RESUELTO (2026-11-16)**, en nb 8 §9.1 en vez de nb 6/7, para no re-ejecutar el tuning del RF. Resultado: el MOS da **MAE 1,330**, así que **sí le gana a no corregir** (−7,9 %) -la afirmación "ningún baseline supera al pronóstico crudo" valía sólo para los 4 de persistencia-. El árbol le gana por 8,7 % y el RF por 14,2 % | *Media* | Hecho. El informe (§4) ya está reescrito con esta tabla: el margen honesto del modelo es el 14,2 % sobre MOS, no el 21,0 % sobre no corregir |
| E5 | nb 7, CV | TimeSeriesSplit parte por posición de fila sobre el panel apilado: un mismo target_hour_utc puede quedar a ambos lados de un corte. Fuga mínima (misma hora, otras estaciones correlacionadas espacialmente) | Baja | Definir folds por timestamp (GroupKFold cronológico o cortes por fecha); al menos mencionarlo |
| E6 | nb 7, selección | El modelo final se elige por MAE_test, no por cv_MAE: sesgo optimista de selección | Media | Elegir por cv_MAE (o mostrar que el ranking coincide) y declararlo en el informe |
| E7 | Modelo de precisión | RF sobreajustado (train 0,661 vs. test 1,141) y de 1,1 GB; fuerza SHAP con n=100 | Media | No reentrenar ahora. En el informe: GB (1,159) como alternativa casi equivalente y desplegable; tradeoff explícito |
| E8 | nb 8, SHAP | ✅ **RESUELTO (2026-11-16)** en nb 8 §9.3: SHAP del árbol sobre el **test completo** (47.151 filas, 2,1 s -no hizo falta limitarse a 2.000: TreeSHAP sobre 1 árbol de 232 hojas es ~5×10⁵ veces más barato por fila que sobre el bosque) + los 3 dependence plots | Media | Hecho. Queda abierto sólo el beeswarm del **Random Forest**, que sigue con n=100: para subirlo haría falta un modelo más liviano (GB) |
| E9 | nb 8, sesgos TP2 | ✅ **RESUELTO (2026-11-16)** en nb 8 §9.2, con umbral pop ≥ 30 % y terciles de viento. Hallazgo contraintuitivo: con lluvia pronosticada el pronóstico base **falla menos** (1,297 vs. 1,450 °C), igual que con viento alto (1,341 vs. 1,512). Los modelos mejoran en los 5 cortes, pero menos donde el pronóstico ya venía bien | Media | Hecho (informe §8.4), con el viento como target declarado fuera de alcance. Cautela al citar: pop ≥ 30 % son sólo 2.025 filas (4,3 % del test) |
| E10 | Target | Target calculado también sobre obs_temp_c imputado (interpolación < 12 h) (auditoría, punto 1) | Baja | Ya cuantificado (13,3 vs. 13,5 %) y el dashboard tiene filtro para excluirlas. Reportarlo como robustez |
| E11 | Outliers | Residuos extremos (E7246: e = 7,2 °C, residuo 8,7 °C; KELD −6,1 °C en la corrida previa). MAE y SHAP medio no los muestran | Baja | Reportar P90/P99 del residuo (metricas.json ya tiene P90) y la tasa de |residuo| > 3 °C antes/después |
| E12 | Dashboard | Umbral de PoP fijo en 50 % y franjas de 6 h | Baja | st.slider de PoP (default 30) y st.slider de rango de hora local (0-23) |
| E13 | Docs | plan-dashboard.md y pasos-pendientes.md mezclan números de corridas (14 %/20,5 %/21 %) | Baja | Unificar con una tabla "cifras oficiales" de la corrida 2026-11-15 |

## 5. Plan de acción (priorizado para entregar el 18/11)

*Bloqueantes (hoy)*
- [x] *Redactar el informe técnico* (entregable 3): **primera versión en `TP3/informe.md`** (2026-11-16), falta
  la revisión del grupo. Estructura: problema y convención de signo → datos y split → baselines
  (ninguno gana a no corregir) → 7 modelos (tabla con MAE/RMSE/MASE/cv_std) → interpretabilidad de cada modelo
  (bloques historia vs. hora) → sesgos pre/post (hora, bioma, lluvia) → incertidumbre conformal → tradeoff
  interpretabilidad vs. precisión (árbol −15,85 % vs. RF −20,96 %, 38 KB vs. 1,1 GB) → limitaciones (mediodía,
  anomalías puntuales, único holdout, test elige familia, viento fuera de alcance).
- [x] *Hacer el dashboard ejecutable* (2026-11-16): versionados los 3,5 MB de data/processed/dashboard/ y
  agregados dashboard/requirements.txt y requirements-precomputo.txt; verificado con un clon limpio. Era:
  versionar data/processed/dashboard/ + requirements.txt (streamlit,
  altair, pandas, pyarrow con versiones), o desplegar y entregar el link. Probar con un clon limpio.
- [x] *Unificar cifras* (E3), hecho el 2026-11-16 en nb 7 §12: informe, tabla de comparación, nb 8 y dashboard
  usan hoy el mismo `cero = 1,4435` y los mismos porcentajes de mejora.

*Alta relación valor/esfuerzo (si queda tiempo hoy)*
- [x] nb 8 §9.2: tabla pre/post por fcst_pop_pct ≥ 30 (lluvia) y por terciles de viento pronosticado (E9).
  Hecho el 2026-11-16.
- [x] nb 8 §8: export_text/plot_tree del árbol a profundidad 3 + Ridge como segunda lectura (entregable 1, hecho
  el 2026-11-16). Falta la otra mitad de E8: SHAP del árbol con n ≥ 2.000.
- [x] nb 8 §9.3: SHAP del árbol (test completo) + los 3 shap.plots.scatter para hora_sin, error_temp_c_lag24 y
  fcst_pop_pct (E8). Hecho el 2026-11-16.
- [x] nb 8 §9.1: baseline de sesgo medio estación × hora (E4), hecho el 2026-11-16 -se implementó en la nb 8 y
  no en la 6/7 para no re-ejecutar el tuning del Random Forest.

*Mejoras opcionales (no bloquean la entrega)*
- [ ] Dashboard: sliders de PoP y de hora local (E12).
- [ ] Seleccionar el modelo final por cv_MAE o mostrar que coincide (E6).
- [ ] CV con cortes por timestamp (E5).
- [ ] Shrinkage en 10-14 h (hallazgo 17), walk-forward (hallazgo 15): dejarlos como trabajo futuro en el informe.