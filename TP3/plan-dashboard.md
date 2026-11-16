# TP3 · Plan del dashboard (Etapa C)

_Propuesta de estructura para el entregable 4, derivada del apunte `apuntes/Clase 13 - TDA1 - 2026.pdf`
("Cómo comunicar datos, modelos e incertidumbre"). Escrito 2026-11-16. Estado de la Etapa C al momento de
escribir esto: no empezada -ver `pasos-pendientes.md`, sección 4._

La Clase 13 da la estructura casi explícita: la p.38 define "nuestro dashboard" como *el pronóstico de temperatura
del curso*, para *un jurado mixto*, con **tres niveles: resumen guiado, explicabilidad y detalle**; la p.39 lo
justifica con la copa de martini (tramo angosto donde conduce el autor, después se abre a la exploración del
lector). Lo que sigue es ese marco aplicado a los resultados que ya produjimos.

## 0. Definiciones previas (p.38: propósito → audiencia → frecuencia → recién después, componentes)

| Definición | Decisión para el TP3 |
|---|---|
| Propósito | Comunicación y defensa de resultados (no monitoreo operativo: el test son 5 días cerrados, no un feed en vivo) |
| Audiencia | Jurado mixto: docente que evalúa método + el "tomador de decisiones" que pide la consigna |
| Decisión que habilita | ¿Conviene usar el pronóstico corregido en lugar del crudo, y en qué horas/regiones confiar menos? |
| Posición autor↔lector | Copa de martini: Nivel 1 cerrado y narrado; Niveles 2 y 3 abiertos con filtros |

## 1. Nivel 1 · Resumen guiado

Conduce el autor: títulos que enuncian la conclusión (p.37) y **sin filtros** -si el filtro cambia el dato, el
título-conclusión caduca, y la p.37 lo advierte explícitamente. Cuatro piezas fijas:

1. **Fila de KPIs con doble encuadre.** La p.35 pide elegir un encuadre y declarar el otro en el subtítulo.
   Con nuestros números: *"El modelo corregido baja el error 20,5%"* (relativo), subtítulo *"0,30 °C menos por hora:
   1,44 → 1,14 °C de MAE"* (absoluto), más `MASE 0,638` y el contraste contra el mejor baseline (naive estacional,
   1,591 °C). Cuarto KPI en encuadre de frecuencia, ya calculado por `dashboard/precomputo.py`: **53,3% de las
   horas del test con error ≤ 1 °C, contra 42,6% sin corregir** (el árbol interpretable queda en 50,3%).
2. **Barras de MAE: no corregir / árbol / Random Forest.** Barras **desde cero**, porque el MAE sí tiene cero
   absoluto (p.11). Corolario del mismo principio: la temperatura en °C **nunca** va en barras.
3. **Una estación, dos paneles apilados con el mismo eje temporal** (p.16): arriba observado + pronóstico +
   corregido; abajo el error con signo de ambos, con el cero marcado. La clase es tajante -si el mensaje es el
   error, se grafica el error, pero la solución buena es mostrar los dos apilados. Con 441 estaciones esto **tiene
   que ser una estación concreta**, no un promedio: promediar 441 series borra la dinámica que se quiere mostrar.
4. **Mapa de los 441 sitios coloreado por mejora relativa.** Responde "¿mejora en todas partes o sólo en algunas?"
   de un vistazo, y es el único canal razonable para 441 unidades espaciales (p.9: región espacial para identidad).

## 2. Nivel 2 · Explicabilidad

Se abre al lector; acá entran los filtros. Títulos en modo herramienta: formulan la pregunta, no la conclusión (p.37).

5. **Ranking de importancia, con las variables redundantes agrupadas *antes* del ranking** (p.31: "primero se
   agrupan las variables redundantes; después se construye el ranking"). Nos toca directo: `hora_sin` + `hora_cos`
   son **una** variable ("hora del día"), y `error_temp_c_lag6` + `lag24` + `roll3_mean` son "historia reciente del
   error". Sin agrupar, el ranking reparte la hora en dos barras y subestima su peso.

   **Lo que apareció al agrupar** (ya implementado en `app.py`, permutation importance sumada por grupo): los dos
   modelos se apoyan en bloques distintos, algo que el ranking por feature suelta no deja ver. El árbol
   interpretable se apoya más en la **historia del error** (0,239 contra 0,235 de la hora), y el Random Forest al
   revés, más en la **hora del día** (0,210 contra 0,172). Ungrouped, en los dos modelos `hora_sin` aparecía
   primera y el contraste entre bloques quedaba invisible. Va al informe como diferencia de comportamiento entre
   los dos entregables. Advertencia honesta que el panel declara: sumar la importancia dentro de un grupo aproxima
   la permutación conjunta del grupo, no la calcula.
6. **Beeswarm de SHAP del Random Forest** al lado de las barras de |SHAP| medio (p.31: las barras dan magnitud, el
   beeswarm agrega el *sentido* del efecto), con la leyenda de una línea sobre cómo se lee.
7. **Permutation importance de los dos modelos en el mismo gráfico** -la tabla ya tiene las dos columnas. Son dos
   series en la misma escala: se **superponen** (p.15), no se yuxtaponen.
8. **MAE por hora, pre y post corrección, en orden horario** -no ordenado por valor (p.12: el ciclo conserva su
   orden natural). Es nuestro hallazgo más honesto: mejora +26% a +39% en la mañana pero sólo +10,5% a +18,3% de
   19h a 23h, y menos en la ventana 10-14h. Título sugerido: *"¿A qué hora corrige menos el modelo?"*
9. **MAE por bioma, pre/post.** Categórico sin orden propio → ordenado por valor (p.2). Cuidado con la p.22
   (Simpson): *Northwestern Forested Mountains* tiene MAE 2,17 °C pero muy pocas estaciones; hay que mostrar el n de
   cada bioma al lado, o la comparación es tramposa.
10. **Explicación local de un caso**, con la doble presentación de la p.32: cascada de SHAP para quien audita **y la
    frase contrafactual** al lado para quien decide. El caso del mediodía ya está exportado
    (`tabla_shap_local_caso_mediodia.csv`). La advertencia de la p.32 -"importancia no es causalidad: SHAP describe
    al modelo, no a la atmósfera"- debería ir escrita en el panel, no sólo en el informe.

## 3. Nivel 3 · Detalle técnico

Cerrado por defecto, se expande.

11. **Diagnóstico de residuos**: ACF del error. La p.30 dice literalmente que su lugar es el detalle técnico del
    dashboard. Ya existe: `data/out/correlaciones_error_temp_c/01_autocorrelacion_propia.png` (notebook 3b).
12. **Distribución del residuo, no sólo el promedio** (p.20): histograma del error con signo y **percentil 90 por
    hora**. El MAE de 1,14 °C esconde las colas, y si el jurado pregunta "¿cuánto puede errar en el peor caso?", el
    promedio no responde. La p.21 pide además probar al menos dos anchos de clase e informar cuál se usó.
13. **Tabla completa de los 7 modelos** con hiperparámetros ganadores y `cv_MAE_std` -ya está en
    `data/out/modelado_global/tabla_comparacion_modelos_temp_c.csv`.
14. **Tabla de validez del target** (79,6% de filas válidas) y resumen del split. Es el pie de página metodológico
    que blinda ante la pregunta obvia sobre cobertura de datos.

## 4. Filtros: un control por cada pregunta real de la audiencia (p.40)

| Control | Pregunta que responde | ¿Va? |
|---|---|---|
| Estación (selector + mapa) | ¿Cómo se comporta en *mi* sitio? | Sí -es el eje del panel 3 |
| Franja horaria | ¿De noche corrige menos? | Sí |
| Estado / bioma | ¿Funciona igual en toda la región? | Sí -lo pide la consigna |
| `fcst_pop_pct` alto / bajo | ¿Sirve en días de lluvia? | Sí -es nuestro proxy de "días de lluvia" de la consigna |
| Hora puntual (explicación local) | ¿Por qué falló esa hora? | Sí, en el Nivel 2 |
| **Período (mes)** | — | **No.** La clase lo conserva porque su test va del 20/3 al 31/5; el nuestro son 5 días (10/11 al 15/11). No hay estacionalidad que filtrar y el control mentiría sobre el alcance |
| Humedad, viento, elevación | Ninguna que formule el jurado | No -van a la notebook |

**Confirmación explícita (agregado 2026-11-16, a pedido)**: los controles viven dentro de un `st.form` con botones
**Aplicar** y **Limpiar**. Sin el formulario, cada clic dispara un rerun y el tablero se recalcula varias veces
mientras el lector arma una sola pregunta; con él, la selección se confirma en bloque.

**Qué alcanza cada filtro, y por qué se veían "sin efecto"**: los filtros alcanzan a Explicabilidad y Detalle, pero
no al Resumen (decisión de la p.37: sus títulos son conclusiones sobre todo el test) ni a los dos primeros paneles
de Explicabilidad (importancia y beeswarm se calculan sobre todo el test en la notebook 8). Eso es correcto por
diseño, pero mirando el Resumen -que es el tab que se abre primero- mover un filtro no cambiaba nada y parecía que
no funcionaban. Tres agregados para que el alcance sea evidente en pantalla, no sólo en un pie de figura:

- el Resumen abre con un aviso de que **no se filtra**, y por qué;
- los dos niveles filtrables abren con el n seleccionado (métrica) y un banner que lista los filtros activos;
- los dos paneles estáticos de Explicabilidad llevan su propio aviso de que no responden a los filtros.

## 5. Dos desajustes entre la clase y nuestro proyecto, a resolver a conciencia

### El Bloque 3 (incertidumbre) no tenía insumo en el pipeline — ✅ resuelto en el precómputo

La clase trabaja con "intervalos calibrados" que declara haber *agregado para esa clase*, y la consigna pide un
dashboard que muestre "la confianza del pronóstico"; el pipeline sólo producía predicciones puntuales.
`dashboard/precomputo.py` ahora genera un **intervalo del 80% por (bioma, franja) estilo split-conformal**, con la
cobertura observada que pide el panel B de la p.29 ("abanico y cobertura: 89%").

**Resultado: cobertura observada 81,3% contra 80% nominal, con un ancho medio de banda de 3,72 °C.**

El camino hasta ahí importa para el informe, porque el primer intento estaba mal y se puede mostrar por qué:

- **Intento descartado — calibrar con los residuos de train**: el Random Forest está sobreajustado en train (MAE
  0,661 contra 1,141 en test), así que sus residuos de train son demasiado angostos. La banda salía de ±1,0 °C y la
  **cobertura observada en test fue del 55,4%** -una banda del 80% que en realidad cubre poco más de la mitad de los
  casos es peor que no mostrar banda, porque induce exactamente el exceso de confianza que el Bloque 3 advierte.
- **Implementado — calibrar con un tramo del test**: split-conformal necesita un set de calibración que el modelo no
  haya visto. Lo canónico sería reentrenar dejando afuera una ventana, pero el bosque pesa 1,1 GB; la alternativa
  correcta y barata es partir el test en dos tramos cronológicos (2 días de calibración, 3 de evaluación) y medir la
  cobertura sólo fuera del tramo de calibración. La columna `tramo` del parquet declara cuáles son cuáles.

Dato que vale la pena contar en el dashboard: la banda del 80% mide **±1,9 °C**, ancha contra un MAE de 1,14 °C. Es
el argumento del Bloque 3 en una sola cifra -el valor puntual oculta el riesgo, y la decisión del productor depende
del ancho, no del promedio.

### No tenemos una variable de "frente frío"

El relato de la clase se apoya en sus 13 episodios de frente. Nuestros 5 días de test seguramente contienen algún
pasaje frontal, pero no está etiquetado. El sustituto más defendible es un flag de cambio brusco de masa de aire
(salto horario de `obs_temp_c` o de `obs_dewpoint_c` por encima de un umbral), que permitiría replicar el ejemplo
resuelto de la p.41 -"con frente el error sube de X a Y, y aun así le gana a no corregir"-, que es el cierre
argumental más sólido de todo el apunte.

## 6. Implementación: el precómputo como frontera

Streamlit, y **precomputar**. La razón es cuantitativa: el `.joblib` del modelo de mínimo error pesa **1,1 GB** (el
interpretable, 40 KB) y `dataset_modelado.csv` 170 MB. Streamlit reejecuta el script entero ante cada interacción
del usuario, así que un dashboard que cargara el modelo en vivo tardaría minutos por clic y exigiría `sklearn`,
`shap` y el modelo de 1,1 GB en el entorno de despliegue -que además no admite un archivo de ese tamaño.

`dashboard/precomputo.py` (✅ hecho, 2026-11-16) traza esa frontera: **todo lo que necesita el modelo ocurre una
sola vez, offline, en este script; el dashboard queda siendo un lector de tablas de pocos MB.** Cumple cuatro
funciones concretas:

1. **Replica el preprocesamiento de la Etapa A sin reentrenar.** Repite la misma carga y la misma máscara de
   validez que `7_modelado_global.ipynb` y `8_interpretabilidad.ipynb` (los `.joblib` traen su `ColumnTransformer`
   ajustado a esos nombres de columna, así que las listas de features tienen que coincidir exactamente) y hace
   `joblib.load` + `predict`. **Validación**: los MAE que imprime -1,4435 / 1,2147 / 1,1410- coinciden hasta el
   cuarto decimal con `data/out/interpretabilidad/tabla_resultados.csv`, que es la prueba de que la réplica es fiel.
2. **Fija una única fuente de verdad numérica** para los KPIs del Nivel 1 (`metricas.json`), en vez de dejar que
   cada panel los recalcule y se desincronice del informe.
3. **Materializa la incertidumbre de la sección 5**: banda del 80% por (bioma, franja) estilo split-conformal,
   calibrada en los primeros 2 días del test y con cobertura medida en los 3 restantes -81,3% observado contra 80%
   nominal (el panel B de la p.29 del apunte).
4. **Deja los SHAP por fila** que el beeswarm necesita y que el dashboard no puede calcular: TreeSHAP sobre este
   Random Forest es tan caro que la notebook 8 tuvo que bajar la muestra de 500 a 100 filas.

Salidas en `data/processed/dashboard/`:

| Archivo | Contenido | Para qué nivel |
|---|---|---|
| `predicciones_test.parquet` (3,3 MB) | 47.151 filas × 432 estaciones: `obs_temp_c`, `fcst_temp_c`, las dos correcciones, los tres errores con signo, banda `obs_lo`/`obs_hi`, `tramo`, y las columnas de filtro (`hora_local`, `franja`, `estado`, `bioma`, `fcst_pop_pct`, `obs_imputado`) | 1 y 2 |
| `estaciones.parquet` (20 KB) | 432 filas: lat/lon, estado, bioma, elevación, horas en test | mapa y selector |
| `metricas.json` | MAE, RMSE, sesgo medio, P90, MASE, % de horas con error ≤ 1 °C, cobertura observada y ancho medio de banda | KPIs del Nivel 1 |
| `intervalos_empiricos.csv` | cuantiles del residuo por (bioma, franja), del tramo de calibración | banda del Nivel 1 |
| `shap_muestra.parquet` (70 KB) | valores SHAP y valor de cada feature, por fila, para la muestra | beeswarm del Nivel 2 |

Total: **3,4 MB** contra los 170 MB del dataset y 1,1 GB del modelo.

Nota sobre el n: el test del split son 54.243 filas, pero al dashboard entran **47.151** -las que además pasan la
máscara de features completas, el mismo criterio con el que se entrenó y se evaluó. Es el número que corresponde
citar en el dashboard y en el informe.

Uso:

```
python TP3/dashboard/precomputo.py                        # corrida completa (~1 min + los SHAP)
python TP3/dashboard/precomputo.py --muestra-shap 0       # sin el beeswarm: el paso lento
python TP3/dashboard/precomputo.py --dias-calibracion 3   # mover el corte de calibración de la banda
```

El paso costoso es TreeSHAP (varios minutos para 100 filas sobre un bosque de 1,1 GB); todo lo demás tarda alrededor
de un minuto. Requiere `pyarrow` (instalado 2026-11-16) y `shap` sólo si se pide el beeswarm.

## 7. La app — ✅ hecha (2026-11-16)

`dashboard/app.py` (+ `dashboard/paleta.py`, `dashboard/.streamlit/config.toml`). Streamlit + Altair; no importa
`sklearn` ni toca los `.joblib`. Para correrla:

```
python -m streamlit run TP3/dashboard/app.py
```

Implementa los 14 componentes de las secciones 1-3 en tres `st.tabs` (Resumen / Explicabilidad / Detalle técnico) y
la fila de filtros de la sección 4 en la barra lateral, delimitando sólo los dos niveles abiertos -el Resumen es
fijo, porque sus títulos enuncian conclusiones y un filtro las haría caducar (p.37).

Decisiones de visualización que vale la pena dejar escritas:

- **Paleta**: los mismos tres colores que `notebooks/utils.py` (azul `#2a78d6`, naranja `#eb6834`, verde
  `#1baf7a`), con color fijo por entidad -filtrar no repinta a las series que sobreviven. Validada sobre superficie
  clara y con todos los pares: banda de luminosidad, piso de croma, separación bajo deuteranopia/protanopia (peor
  par ΔE 9,2) y piso de visión normal (24,0) pasan; el verde queda en 2,74:1 de contraste, por debajo de 3:1, y por
  eso **todos** los gráficos llevan etiqueta visible o su tabla equivalente en un `expander`, no sólo tooltip.
- **El tema se fija en claro** (`.streamlit/config.toml`): la paleta se validó contra ese fondo, y dejarla a merced
  del tema del sistema la renderizaría sobre una superficie para la que no se verificó el contraste.
- **El histograma se agrega en pandas, no en el navegador** (47.151 filas × 3 series exceden el límite de Altair), y
  va como contorno escalonado en vez de tres rellenos translúcidos superpuestos, que se ocluyen entre sí.
- **Un eje por gráfico**, nunca dos escalas: temperatura y error van en dos paneles apilados que comparten el eje
  temporal, que es además lo que pide la p.16.

Verificación: `streamlit.testing.v1.AppTest` ejecuta el script completo y reporta excepciones. Corridas sin
excepción en 11 caminos -base, cada filtro por separado, los tres anchos de clase, cambio de estación, banda
apagada, y el caso borde de una combinación (estado, bioma) sin ninguna fila, donde los dos niveles filtrables
avisan y el Resumen sigue en pie.

Pendiente de la Etapa C: nada bloqueante. Queda el informe (entregable 3).
