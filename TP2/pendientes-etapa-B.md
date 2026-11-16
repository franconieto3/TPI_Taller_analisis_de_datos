# Etapa B (TP2): qué está hecho y qué falta

_Última actualización: 13/10. Avance global estimado: **~15 %** (H7 testeada de punta a punta; falta H1–H6, H8 y el cierre)._

## Contexto

La Etapa B de la consigna (`consigna/Consigna TP2.pdf`) pide, sobre el dataset de errores de pronóstico ya construido en la Etapa A, tres cosas:

1. **Formulación de hipótesis**: plantear hipótesis formales sobre qué factores causan que el pronóstico sea peor.
2. **Pruebas de significancia**: aplicar tests estadísticos para contrastar las hipótesis y validar si las variables climáticas realmente se asocian al rendimiento del modelo.
3. **Validación de sesgo**: pruebas formales para comprobar si el sesgo direccional detectado en la Etapa A es estadísticamente significativo o si podría deberse al azar.

Esta etapa depende de trabajo que todavía no está cerrado:

- Los pendientes de la Etapa A (`pendientes-etapa-A.md`), en particular la **síntesis de sesgos direccionales** (sección "Pasos > 4. Cierre") que lista justamente los sesgos a validar acá: viento sobreestimado, ciclo diario de temperatura/HR, sesgo con PoP 41–60 %, no alcanza los extremos, sesgo de viento por encima de 37° de latitud.
- El hallazgo de autocorrelación en `patrones-temporales.ipynb` (ACF a 1 h ≈ 0,73–0,89 según la serie): el n efectivo para los tests es mucho menor que el n de filas, hay que decidir cómo se agrega antes de testear.
- El plan de atributos geográficos y urbanos (`plan-atributos-geograficos.md`): agua, ciudades y aeropuerto ya están implementados (`ETL/atributos_geograficos.py` → `estaciones_atributos.csv`) y testeados en H7; falta bioma, que haría falta para H8 si se recorta por zona.

> **Nota de fecha.** La consigna indica entrega el 07OCT2026 (campus habilitado hasta las 23:59 del 08OCT26). La fecha actual del sistema es 13/10, posterior a ambas. Conviene confirmar con la cátedra si hubo prórroga antes de planificar el resto del cronograma.

## Hipótesis candidatas (derivadas de los hallazgos de la Etapa A)

| # | Hipótesis | Test sugerido | Estado |
|---|---|---|---|
| H1 | El sesgo del viento pronosticado (+2,9 km/h) es sistemático, no producto del azar | Test t de una muestra o Wilcoxon signed-rank sobre el sesgo medio por estación (H0: media = 0) | ⬜ |
| H2 | El sesgo de temperatura y HR varía con la hora del día (frío/húmedo de mañana, cálido/seco de tarde) | ANOVA o Kruskal-Wallis del error por franja horaria | ⬜ |
| H3 | Con PoP pronosticada 41–60 % el pronóstico sobreestima más la temperatura que con otras PoP | Comparación de medias entre grupos de PoP (ANOVA/Kruskal-Wallis) | ⬜ |
| H4 | El pronóstico no alcanza los extremos observados (HR cerca de saturación, punto de rocío alto) | Regresión/correlación error vs. valor observado, test de pendiente ≠ 0 | ⬜ |
| H5 | El sesgo del viento es mayor por encima de 37° de latitud | Comparación de medias (t-test/Mann-Whitney) entre grupos de latitud | ⬜ |
| H6 | La elevación de la estación se asocia a mayor MAE, controlando por latitud y red | Correlación parcial o regresión múltiple | ⬜ |
| H7 | La distancia a agua, a área urbana o a la estación ASOS/aeropuerto más cercana se asocia al error | Correlación (Spearman) o regresión | 🟡 testeada en `notebooks/hipotesis-tests.ipynb`: confirmada para agua (temp./HR/rocío) y aeropuerto (ídem), parcial para área urbana (solo el viento sobrevive al controlar por red) |
| H8 | La red de la estación (proxy de lo urbano) se asocia al error, más allá de la ubicación | ANOVA/Kruskal-Wallis por red, controlando región | ⬜ |

## Pasos

Leyenda: ✅ hecho · 🟡 parcial · ⬜ pendiente.

### 0. Preparación
- ✅ Decidir la unidad de análisis para los tests: una fila por estación (agregando sesgo/MAE), no una fila por medición, para no inflar el n con observaciones autocorrelacionadas en el tiempo (ver ACF en `patrones-temporales.ipynb`). Aplicado en `hipotesis-tests.ipynb`; el mismo criterio sirve para el resto de las hipótesis.
- 🟡 Si se incluyen H7/H8 con atributos geográficos nuevos, implementar `ETL/descargar_externos.py` y `ETL/atributos_geograficos.py` según `plan-atributos-geograficos.md`. Hecho `atributos_geograficos.py` (agua, aeropuerto, área urbana → `estaciones_atributos.csv`); no hizo falta `descargar_externos.py` porque las capas se agregaron directamente a `data/raw/`. Bioma todavía no se cruzó (lo necesitaría H8 si se recorta por zona).

### 1. Formulación de hipótesis (nuevo notebook, ej. `notebooks/hipotesis-tests.ipynb`)
- 🟡 Redactar cada hipótesis en formato H0/H1 formal (ej. H0: sesgo medio del viento = 0; H1: ≠ 0). Hecho para H7; falta para H1–H6 y H8.
- 🟡 Justificar cada hipótesis citando el hallazgo de la Etapa A que la motiva (figura/tabla concreta). H7 se justifica con la disponibilidad de datos, no con un hallazgo previo (es una hipótesis nueva, no uno de los sesgos ya detectados); falta para el resto.
- 🟡 Para cada hipótesis, fijar variable dependiente (error con signo, |error| o MAE por estación) y variable independiente (hora, PoP, latitud, red, elevación, atributo geográfico). Hecho para H7 (dist_agua_km, dist_aeropuerto_km, dist_area_urbana_km, sobre_agua, en_area_urbana); falta para el resto.

### 2. Pruebas de significancia
- 🟡 Elegir el test según la distribución real del error: los QQ-plots de la Etapa A muestran colas pesadas (curtosis 1,1–5,5), así que conviene preferir pruebas no paramétricas (Mann-Whitney, Kruskal-Wallis, Spearman) o bootstrap antes que asumir normalidad. Aplicado en H7 (Spearman, Mann-Whitney, Wilcoxon); falta para el resto.
- 🟡 Corregir por comparaciones múltiples si se testean varias hipótesis o variables a la vez (Bonferroni o FDR). Aplicado en H7 (FDR de Benjamini-Hochberg, por familia de tests); falta integrar con el resto de las hipótesis si se testean en el mismo notebook.
- 🟡 Reportar tamaño de efecto además del p-valor (diferencia de medias, ρ de Spearman), no solo "significativo / no significativo". Aplicado en H7.
- 🟡 Documentar el problema de autocorrelación temporal y espacial (ACF, y 43 % de las estaciones concentradas en Houston) y cómo se mitiga en cada test (agregación por estación/día, block bootstrap, etc.). En H7: la temporal se resuelve agregando a una fila por estación; la espacial (Houston/CENTERPOINT) se controla repitiendo el test sin esa red (secc. 4) y se declara como limitación, sin modelo espacial formal.

### 3. Validación del sesgo direccional
- ⬜ Aplicar a H1–H5 (los sesgos ya detectados en la Etapa A) el test correspondiente de la tabla de hipótesis.
- ⬜ Concluir, por cada uno, si el sesgo es estadísticamente significativo o compatible con el azar, al nivel α elegido.
- ⬜ Distinguir significancia estadística de relevancia práctica (ej. un sesgo de 0,1 °C puede dar significativo con n grande y ser irrelevante para el caso de uso).

### 4. Cierre de la Etapa B
- ⬜ Síntesis de qué hipótesis se confirmaron, cuáles no, y qué factores explican mejor/peor el pronóstico.
- ⬜ Sección de limitaciones de los tests usados (lo pide la consigna explícitamente): supuesto de normalidad, autocorrelación, n efectivo, variables confundidas (red vs. ubicación, elevación vs. latitud).
- ⬜ Volcar hallazgos y limitaciones al informe técnico en PDF (entregable 2), junto con el código documentado (entregable 1).

## Archivos críticos
- `notebooks/hipotesis-tests.ipynb`: H7 formulada, testeada (Spearman/Mann-Whitney/Wilcoxon + FDR) y con conclusión; falta sumar H1–H6 y H8.
- `ETL/atributos_geograficos.py`: cruza `estaciones.csv` con agua/aeropuerto/área urbana → `data/processed/estaciones_atributos.csv`, que usa H7.
- `pendientes-etapa-A.md`: pendientes que hay que cerrar antes (síntesis de sesgos, matriz con errores, boxplots).
- `plan-atributos-geograficos.md`: atributos geográficos/urbanos; agua/ciudades/aeropuerto ya implementados, bioma pendiente.
- `notebooks/utils.py`: reutilizar `cargar_mediciones_limpias()` y las funciones de estilo ya centralizadas.

## Verificación
- Cada test declara H0/H1, estadístico, p-valor, tamaño de efecto y conclusión en texto simple.
- Ningún test asume normalidad sin chequearla antes (reusar los QQ-plots / test de normalidad de la Etapa A).
- El n usado en cada test es el n efectivo (por estación o por día), no el n de filas crudo.
- Ejecutar el notebook de punta a punta sin errores.
