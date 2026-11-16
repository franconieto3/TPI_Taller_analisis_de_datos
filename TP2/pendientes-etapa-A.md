# Etapa A (TP2): qué está hecho y qué falta

_Última actualización: 13/10. Avance global estimado: **≈ 45–50 %**._

## Contexto

La Etapa A de la consigna (`consigna/Consigna TP2.pdf`) pide tres análisis exploratorios del **error de pronóstico** (observado − pronosticado):

1. **Distribución de los errores**: asimetría, normalidad y valores atípicos.
2. **Patrones temporales y cíclicos**: autocorrelación y estacionalidad del error.
3. **Relaciones bivariadas**: dispersión y matrices de correlación de las variables **contra la magnitud (|error|) y la dirección (signo) del error**.

Además, los objetivos generales piden usar atributos geográficos, climáticos y urbanos para encontrar zonas con mejores o peores pronósticos.

`notebooks/relaciones-bivariadas.ipynb` tiene la limpieza (incluida la exclusión de las 129 estaciones `sin_meteo`), la matriz de correlación, observado vs. pronosticado, las relaciones físicas, el ciclo diario, el error de temperatura por hora y por PoP, la rosa de vientos, el mapa del viento por estación y, desde el 12/10, la **sección 8 "Error del pronóstico por variable"**: distribución del error de las 4 variables (fig. 08), error según la hora del día (fig. 09) y sesgo según el valor observado (fig. 10). Desde el 13/10 la sección 1 exporta el dataset limpio (`data/processed/mediciones_limpias.csv`) y la sección 8 incluye el error según la elevación de la estación (fig. 11).

> **Convención de signo.** La consigna y las secciones 3–7 del notebook usan **observado − pronosticado**. La sección 8 usa
> **pronosticado − observado** (positivo = el pronóstico sobreestima). Hay que aclararlo en el informe y en la síntesis, o unificar el signo.

## Estado por ítem

| Ítem | Estado | Lo que ya hay | Lo que falta |
|---|---|---|---|
| 0. Preparación | ✅ 100 % | Dataset limpio en `data/processed/mediciones_limpias.csv` con errores con signo y absolutos de las 4 variables; estado, zona horaria, lat/lon, elevación y red de cada estación; exclusión de las estaciones `sin_meteo`; `cargar_mediciones_limpias()` en `utils.py` | — |

| 1. Distribución de errores | 🟡 ~80 % | Histogramas con sesgo/MAE/RMSE/n (fig. 08); cuantización y calmas del viento explicadas; mapa de sesgo del viento por estación (fig. 07); QQ-plots con asimetría y curtosis (`distribucion-errores.ipynb`, fig. 01); errores extremos clasificados en sensor o modelo, con estaciones sospechosas y eventos (figs. 02–03); mapas de sesgo y MAE por estación de las 4 variables (figs. 04–05) | normal/KDE superpuesta, percentiles, % de atípicos 1,5·RIC, test de normalidad |

| 2. Patrones temporales 
| 🟡 ~85 % 
| Sesgo, RIC y MAE por hora local para las 4 variables, y tabla de extremos (fig. 09); ACF hasta 72 h por estación y regional (`patrones-temporales.ipynb`, fig. 01); sesgo y MAE diarios con días atípicos y heatmap estado × día (figs. 02–03); heatmap día × hora local de temperatura y viento con descomposición por región (fig. 04) 
| PACF, mínimo de las 9 h a revisar, descomposición (opcional) |

| 3. Relaciones bivariadas 
| 🟡 ~65 % 
| Sesgo según el valor observado para las 4 variables (fig. 10); error de temp. vs. PoP (fig. 05); sesgo del viento por red y por franja de latitud; elevación y latitud en la matriz (fig. 01); sesgo y MAE por estación vs. elevación (fig. 11) 
| Errores en la matriz, error de temp. vs. HR, error de viento vs. viento pronosticado y por sector, boxplots por estado y por red |

| 4. Cierre 
| ❌ ~10 % 
| Los sesgos están en las conclusiones (secc. 9), mezclados con el resto 
| Síntesis de sesgos direccionales, informe en PDF |

## Pasos

Leyenda: ✅ hecho · 🟡 parcial · ⬜ pendiente.

### 0. Preparación (reutilizar la limpieza)
- ✅ Dataset limpio en `data/processed/mediciones_limpias.csv` (217.297 filas, 345 estaciones, 34 columnas), exportado al final de la sección 1 de `relaciones-bivariadas.ipynb`. Incluye la limpieza (apóstrofos, umbrales de 15 °C y 120 km/h, hora local y exclusión de `sin_meteo`), los atributos de estación y los errores. Los notebooks nuevos lo leen con `from utils import cargar_mediciones_limpias` (convierte los timestamps a UTC).
- ✅ Columnas de error para las 4 variables, con signo pronosticado − observado. En el notebook (secc. 8): `error_Temperatura`, `error_Humedad relativa`, `error_Punto de rocío`, `error_Velocidad del viento`. En el dataset limpio: `error_temp_c`, `error_rh_pct`, `error_dewpoint_c`, `error_windspeed_kmh` y su valor absoluto (`abs_error_<variable>`).
- ✅ Atributos de estación: `estado`, `timeZone`, `latitud`/`longitud`, `elevation` y `red` (`subProvider`, o `provider` si no tiene), todos desde `data/processed/estaciones.csv` (`ETL/armar_estaciones.py`). Las estaciones de ARDOT y RDOT no tienen elevación (22.794 filas) y las boyas de los Grandes Lagos no tienen red. La red sirve como proxy de lo "urbano" (CENTERPOINT/HCFCD frente a ASOS o aeropuertos).
- ✅ Paleta, colormaps, `rcParams` (`aplicar_estilo()`) y las funciones `hexbin_par`, `escala_comun` y `banda_por_hora` movidos a `notebooks/utils.py`. Los notebooks nuevos los importan con `from utils import ...`.

### 1. Distribución de los errores (`notebooks/distribucion-errores.ipynb`; los histogramas siguen en la secc. 8 de `relaciones-bivariadas.ipynb`)
- 🟡 Histograma por variable (4 paneles): hecho (fig. 08), con los bins alineados al paso de °F y mph. Falta superponer una normal de igual media y desvío (o una KDE).
- ✅ Un QQ-plot por variable contra la normal (`distribucion-errores.ipynb` secc. 2, fig. `distribucion/01_qqplot_error.png`), con recta por P25–P75 y `tabla_forma` (n, asimetría, curtosis en exceso). Las 4 tienen colas pesadas (curtosis 1,1–5,5); la temperatura y el viento tienen la cola larga en las sobreestimaciones y la HR y el punto de rocío en las subestimaciones.
- 🟡 Tabla resumen por variable: ya tiene n, sesgo, MAE y RMSE (`tabla_error`). Asimetría y curtosis ya están en `tabla_forma`. Falta mediana, desvío, P1/P5/P95/P99, % de atípicos según el criterio 1,5·RIC y % con |z| > 3.
- ⬜ Normalidad: test de D'Agostino o Jarque-Bera (`scipy.stats.normaltest`), aclarando que con n ≈ 180 000 cualquier desvío sale significativo, por lo que la conclusión se apoya en el QQ-plot y en la curtosis. Esto conecta con el pedido de "analizar limitaciones".
- ✅ Atípicos: `distribucion-errores.ipynb` secc. 3. Extremo = más de 3 RIC fuera de P25–P75 (2.232 casos, 0,3 %). Cada uno se clasifica en valor congelado (≥ 6 reportes iguales seguidos), estación con falla persistente (≥ 10 % de extremos), evento compartido (vecina a ≤ 50 km con extremo del mismo signo a la misma hora), aislado o sin vecinos. Resultado: en temperatura el 62 % son fallas del modelo (tardes de tormenta en Houston: pron. ~33 °C, obs. ~26 °C, PoP 37 %); en punto de rocío el 59 % son sensores. 11 estaciones sospechosas, todas de APRSWXNET salvo ALMW3 (fig. 02, `tabla_estaciones_sospechosas.csv`). Excluirlas casi no cambia las métricas globales (`tabla_efecto_exclusion.csv`). La HR observada se deriva de T y Td (Magnus), así que no sirve para validar el higrómetro. Lista completa: `data/out/distribucion/errores_extremos_clasificados.csv`.
- ✅ Error medio y MAE **por estación** en un mapa: `distribucion-errores.ipynb` secc. 4, figs. `distribucion/04_mapa_sesgo_estacion.png` y `05_mapa_mae_estacion.png` (4 variables, estaciones con ≥ 100 pares, recuadro de Houston como la fig. 07, sospechosas de la secc. 3 con borde negro). Tablas: `tabla_error_por_estacion.csv` y `tabla_resumen_por_estacion.csv`, más el resumen por red. Hallazgos: el MAE crece hacia el norte en las 4 variables (confundido con la red: TX es casi todo CENTERPOINT); temperatura y HR se espejan (Houston cálido y seco, Arkansas/ARDOT frío y húmedo); el 85 % de las estaciones sobreestima el viento, salvo ARDOT; en el borde este de Houston (bahía de Galveston) el pronóstico de viento es un 75 % más alto con el mismo viento observado, probablemente por representatividad. Usa geopandas: correr en `geo_env`.
- ✅ Caso del viento: cuantización en mph y calmas explicadas (secc. 3 y 8); distribución asimétrica con sesgo de +2,9 km/h (pron. − obs.).

### 2. Patrones temporales y cíclicos (nuevo: `notebooks/patrones-temporales.ipynb`)
- ✅ Sesgo por hora local con banda RIC y MAE para **las 4 variables** (fig. 09 y `tabla_error_hora`).
- 🟡 **Autocorrelación**: hecha la ACF hasta 72 h (`patrones-temporales.ipynb` secc. 2, fig. `temporales/01_acf_error.png`, `tabla_acf_error.csv`), por estación sobre la serie horaria reindexada con huecos (`missing='conservative'`, estaciones con ≥ 50 % de las horas), con el promedio entre estaciones (banda P25–P75) y la ACF de la serie media regional. Hallazgos: ACF a 1 h ≈ 0,73 por estación (0,59 en viento) y 0,81–0,89 en la serie regional, que se anula entre las 9 y las 12 h; picos en 24, 48 y 72 h y valles en 12 y 36 h en temperatura, HR y viento (ciclo diario del sesgo); el punto de rocío tiene pico diario por estación pero no regional. Implica un n efectivo mucho menor para los tests de la Etapa B. Falta la PACF.
- ✅ Serie temporal del sesgo y el MAE diarios (`patrones-temporales.ipynb` secc. 3, figs. `temporales/02_error_diario.png` y `03_sesgo_diario_estado.png`; `tabla_error_diario.csv` y `tabla_dias_atipicos.csv`). Día local en hora central, días con ≥ 20 h (01–27/09), banda P25–P75 entre estaciones, días atípicos con z robusto > 2,5, contexto con T/Td observados y PoP pronosticada, y heatmap estado × día del sesgo. Hallazgos: viento sobreestimado los 27 días (más en KS, MN, NE, IL; AR ≈ 0); período lluvioso en TX (01–06/09) con el peor MAE de temperatura (03 y 06/09) y el 02/09 frío y húmedo en TX y OK; dos frentes (17–18/09 y 25–27/09, caída del Td observado): el 17/09 el pronóstico es seco, cálido y ventoso y el 25–27/09 el Td tiene el peor MAE del mes (primero seco, después húmedo); desde el 17/09 sesgo húmedo diario en NE e IA; OK frío y húmedo casi todos los días. Sin precipitación observada, la lluvia solo se infiere de la PoP.
- ✅ Heatmap día × hora local del sesgo de temperatura y viento (`patrones-temporales.ipynb` secc. 4, fig. `temporales/04_sesgo_dia_hora.png`; `tabla_sesgo_dia_hora.csv` y `tabla_descomposicion_dia_hora.csv`), con el ciclo medio arriba, la media de cada día a la derecha, los días atípicos de la secc. 3 en negrita y una descomposición tipo ANOVA (ciclo horario / nivel del día / interacción) para toda la región, Texas y el resto. Hallazgos: en temperatura la interacción pesa más que el ciclo (46 % vs. 34 %): los días con peor MAE (03, 06, 10, 13 y 17/09) son tardes de Texas con +2 °C y PoP de tarde alta (tormentas); fuera de Texas domina un ciclo fijo (45 %) con la mañana más fría. En viento pesa más el nivel del día (33 %): días enteros sobreestimados (17/09, noche del 02/09, tarde del 01/09); solo 29 de 648 celdas son negativas. **A revisar:** a las 9 h el sesgo de temperatura es negativo en 25 de 27 días y salta a ≈ 0 a las 10 h en todas las redes principales; no es por el desfase de la observación (10 min constante), quizás por cómo el pronóstico arma la mañana.

### 3. Completar las relaciones bivariadas (editar `relaciones-bivariadas.ipynb`)
- 🟡 **Matriz de correlación con las columnas de error**: `elevation` y `latitud` ya están en la matriz Pearson/Spearman (fig. 01). Falta agregar los errores con signo y en valor absoluto. Alcanza con mostrar el bloque "variables × errores" para que sea legible.
- 🟡 Error vs. variable: hecho el sesgo según el valor observado para las 4 variables (fig. 10), que cubre error de temp. vs. temp. y error de HR vs. HR (saturación en 100 %). Falta error de temp. vs. HR y error de viento vs. viento **pronosticado**.
- 🟡 Error vs. atributos de estación: hecho para el viento por red (`tabla_viento_red`) y por franja de latitud (`tabla_viento_franja`). Hecho el sesgo y el MAE por estación vs. elevación para las 4 variables (fig. 11, `tabla_error_vs_elevacion`): el MAE crece con la elevación, pero está confundido con la latitud (ρ ≈ 0,8 entre ambas) y con la red APRSWXNET; puede estar influido por las estaciones sospechosas de falla del sensor (varias están por encima de 500 m) (`distribucion/tabla_estaciones_sospechosas.csv`), conviene repetirla sin ellas. Faltan los boxplots por estado y por red para las 4 variables.
- ✅ Conclusiones (secc. 9) y exportación (secc. 10) actualizadas con las figuras 08–10 y sus tablas.

### 4. Cierre de la Etapa A
- 🟡 Síntesis de los **sesgos direccionales**, que la Etapa B pide validar formalmente. Ya detectados (falta reunirlos en un apartado propio):
  - el viento se sobreestima (+2,9 km/h), más por la tarde y con calmas o vientos leves;
  - ciclo diario en temperatura y HR: el pronóstico es frío y húmedo a la mañana y cálido y seco por la tarde; el punto de rocío casi no tiene sesgo;
  - con PoP de 41–60 % el pronóstico sobreestima la temperatura;
  - el pronóstico no alcanza los extremos, sobre todo la HR en saturación y el punto de rocío alto;
  - el sesgo del viento es mayor por encima de 37° de latitud.
- 🟡 Exportar todas las figuras a `data/out/<sección>/` con el patrón `figuras = {}` + `savefig`: hecho para `bivariadas/` (figs. 01–11 y 6 tablas) y `distribucion/` (figs. 01–05, 8 tablas y la lista de extremos clasificados).
- ⬜ Volcar figuras y conclusiones al informe técnico en PDF (entregable 2).

## Archivos críticos
- `notebooks/relaciones-bivariadas.ipynb`: se edita (exportar el dataset limpio, sumar errores a la matriz y las secciones nuevas).
- `notebooks/distribucion-errores.ipynb`: inciso 1 (creado el 13/10, con el QQ-plot).
- `notebooks/patrones-temporales.ipynb`: nuevo.
- `data/raw/metadata_estaciones.json` y `data/processed/estaciones_central.geojson`: fuente de los atributos de estación.

## Verificación
- Ejecutar los notebooks de punta a punta sin errores (con `nbclient`, ver la nota sobre `geo_env` y el parche SSL).
- Comprobar que los n de cada tabla coinciden con los pares completos ya informados (≈ 186 k para temperatura).
- Comprobar que el sesgo y el MAE de los notebooks nuevos coinciden con `tabla_error_pronostico.csv` (con el signo invertido respecto de `tabla_observado_vs_pronosticado.csv`).
- Revisar que las PNG exportadas en `data/out/` sean legibles.