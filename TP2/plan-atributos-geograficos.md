# Atributos geográficos y urbanos de las estaciones: agua, ciudades, bioma y aeropuertos

## Contexto

La consigna pide "utilizar distintos atributos (geográficos, climáticos, urbanos, etc.) para distinguir zonas con mejores o peores pronósticos", y la Etapa B pide hipótesis sobre qué factores empeoran el pronóstico. Hoy `data/processed/estaciones.csv` tiene lat/lon, `elevation`, estado, zona horaria y red. La idea es sumar cuatro atributos: cercanía a cuerpos de agua, cercanía a ciudades, bioma y si la estación está en un aeropuerto. Abajo se detalla qué se puede aproximar con lo que ya hay, qué datasets externos hacen falta y cómo se calcularía cada atributo.

## Qué se puede sacar de los datos actuales (proxies)

Revisé `estaciones.csv` (500 estaciones, 345 `con_datos`).

| Atributo | Proxy disponible hoy | Limitación |
|---|---|---|
| Agua | Boyas de los Grandes Lagos (IDs `45xxx`, sin provider: 4 con datos), `MARITIME` (2), `NOS-NWLON` (1); `origen_estado = 'estado más cercano (agua)'` (28 estaciones fuera de todo polígono estatal, es decir, sobre el agua); nombres con *Bayou/Lake/River/Bay/Harbor*; franja de latitud 25–31° (Golfo) | Sirve para marcar las que están **sobre** el agua, no para medir distancias. Las estaciones de agua con datos son muy pocas (~7). |
| Ciudades | Red como proxy urbano: `CENTERPOINT` (148, área de Houston), `HCFCD` (Houston), `WEATHERSTEM` (escuelas y municipios), `APRSWXNET` (casas de radioaficionados); el código de condado FIPS está en la URL `county` de la metadata cruda (`TXC201` → Harris) | La red queda confundida con la ubicación (Houston es casi toda una sola red). Con el FIPS se pueden unir tablas por condado, pero las tablas son externas. |
| Bioma | `latitud`, `longitud`, `elevation`, `estado`; `RAWS` (12, estaciones de incendios en zonas silvestres) | Son gradientes continuos, no una clasificación. |
| Aeropuerto | **Ninguno.** No hay estaciones ASOS/AWOS: ningún ID tiene la forma ICAO `Kxxx` (todos tienen 5 caracteres), y el único nombre con "Airport" es una pista privada (`115CE Mario's Flying Pizza Airport`) | Un "es aeropuerto = sí/no" quedaría ≈ 0 en todas. Hay que redefinirlo como **distancia al aeropuerto o ASOS más cercano**. |

Por qué conviene la distancia a un ASOS: el modelo del NWS (NDFD/NBM) se calibra con observaciones ASOS. Es razonable plantear como hipótesis que **el error crece con la distancia a la estación ASOS más cercana**. Además es una pregunta que no se puede responder con los datos actuales.

## Datasets a agregar (recomendados; todos públicos y gratuitos)

| Atributo | Dataset | Formato / tamaño | Métricas por estación |
|---|---|---|---|
| Agua | **Natural Earth 10m** `coastline` + `lakes` (+ `rivers_lake_centerlines` opcional) | Shapefile, unos pocos MB | `dist_costa_km` (Golfo), `dist_lago_grande_km` (Grandes Lagos y lagos grandes), `dist_agua_km` = mínimo de ambas; `sobre_agua` (bool) |
| Ciudades | **Census Gazetteer – Places 2020** (CSV con lat/lon y población) + **TIGER/Line Urban Areas 2020** (polígonos) | CSV ~3 MB; shapefile ~100 MB | `en_area_urbana` (bool), `dist_area_urbana_km`, `dist_ciudad_50k_km` (lugar ≥ 50 000 hab.), `pob_ciudad_cercana` |
| Bioma | **EPA Ecoregions Level III** (CONUS) y, como alternativa climática, **Köppen-Geiger 1 km** (Beck et al. 2023) | Shapefile ~30 MB; GeoTIFF | `ecoregion_l2`/`ecoregion_l3` (categórica); `koppen` (Cfa, Dfa, BSk…) |
| Aeropuerto | **OurAirports `airports.csv`** (tipo large/medium/small, lat/lon) + lista de estaciones ASOS del **IEM** (Iowa Environmental Mesonet) | CSV ~12 MB; CSV/GeoJSON por estado | `dist_aeropuerto_km` (medium/large), `dist_asos_km`, `en_aeropuerto` (< 1 km, se espera ≈ 0) |

Opcional (más pesado, un solo raster que cubre tres atributos): **NLCD 2021** (land cover a 30 m, ~2 GB para CONUS). Con estadísticas zonales en un buffer de 1–5 km da `%_agua` (clase 11), `%_urbano` y `impermeabilidad_%` (isla de calor) y la cobertura dominante. Es la medida más fina, pero requiere `rasterio`/`rasterstats` y mucho disco. Queda para una segunda etapa.

Sobre la elección de Ecoregiones Level III frente a WWF Biomes: en la región central el mapa de biomas de WWF solo da 2 o 3 clases (pastizal templado, bosque latifoliado templado, algo de desierto en el oeste de TX), con poca variación. Level III distingue unas 30–40 regiones y Level II unas 10, lo que deja grupos con suficientes estaciones.

## Implementación propuesta

1. **Descarga** a `data/raw/externos/` con un script `ETL/descargar_externos.py` (URLs fijas y `curl`, porque el SSL de Python de este equipo falla; mismo patrón que `_get_json` en `ETL/clasificar_estaciones.py`).
2. **Nuevo script `ETL/atributos_geograficos.py`**, con el mismo estilo que `ETL/armar_estaciones.py`:
   - parte de `data/processed/estaciones.csv` → `GeoDataFrame` en `CRS_METRICO = 'EPSG:5070'` (el mismo que usan `analisis_estaciones.ipynb` y `relaciones-bivariadas.ipynb`);
   - distancias con `gpd.sjoin_nearest(..., distance_col=...)` (patrón ya usado en `analisis_estaciones.ipynb` para asignar el estado); para puntos sobre el agua, `dist_agua = 0`;
   - pertenencia (área urbana, ecorregión) con `gpd.sjoin(..., predicate='within')`; las boyas quedan sin ecorregión → `'agua'`;
   - salida `data/processed/estaciones_atributos.csv` (una fila por `station_id`, sin geometría).
3. **Análisis** (sección nueva en `relaciones-bivariadas.ipynb` o un notebook propio), usando `utils.py`:
   - agregar el error a **una fila por estación** (sesgo y MAE por variable) para no inflar el n con 180 mil filas correlacionadas;
   - dispersión MAE vs. `dist_agua_km`, `dist_asos_km` y `dist_area_urbana_km` (eje log), y Spearman;
   - boxplots de MAE por `ecoregion_l2`/`koppen` y por `en_area_urbana`;
   - **controlar por red**: repetir sin `CENTERPOINT` y sin `APRSWXNET`, o estratificar, porque la red y la ubicación están confundidas (ya se vio con el viento por franja de latitud).
4. Agregar las columnas nuevas a la matriz de correlación pendiente (punto 3 de `pendientes-etapa-A.md`).

## Archivos que se tocan

| Archivo | Nuevo/editar | Qué hace |
|---|---|---|
| `ETL/descargar_externos.py` | nuevo | Baja las capas externas a `data/raw/externos/` (ignorado por git, como el resto de `data/`) |
| `ETL/atributos_geograficos.py` | nuevo | **Une las capas con las estaciones**: lee `data/processed/estaciones.csv` y las capas, y escribe `data/processed/estaciones_atributos.csv` (`station_id` + columnas nuevas) |
| `ETL/armar_estaciones.py` | sin cambios | Sigue generando la base liviana (solo pandas). No se mezcla con geopandas ni con las descargas, así puede correr aunque falten las capas externas |
| `notebooks/utils.py` | editar | Suma `cargar_estaciones(processed)`: devuelve `estaciones.csv` unido a `estaciones_atributos.csv` (si existe) por `station_id`, en un solo lugar para todos los notebooks |
| `notebooks/relaciones-bivariadas.ipynb` | editar | La celda de hora local (`b3c4a253`) hoy une `estaciones_central.geojson[['id','estado','timeZone']]`, y la celda de redes (`a6e0b4f1`) une `cobertura_estaciones.csv`. Las dos pasan a usar `cargar_estaciones()` y un solo `mp.merge(..., on='station_id', how='left')`. A partir de ahí, `mp` ya trae `elevation`, la red y los atributos nuevos |
| notebooks nuevos (`distribucion-errores`, `patrones-temporales`) | — | Usan el mismo `cargar_estaciones()` |

Las mediciones no se unen en el ETL: `mediciones-pronosticos.csv` tiene 55 MB y unirla allí duplicaría los atributos en cada fila. La unión con las mediciones se hace en el notebook, después de la limpieza.

## Limitaciones a declarar en el informe
- Solo 345 estaciones con datos y ~43 % en Houston: hay autocorrelación espacial y pocas estaciones en algunos grupos (agua ≈ 7, RAWS = 12).
- El pronóstico es una celda de grilla de 2,5 km. En la costa, la celda mezcla tierra y agua, y eso explica parte del error, no el modelo en sí.
- Natural Earth es de escala 1:10M: alcanza para la costa y los lagos grandes, pero no para ríos o lagos chicos (para eso harían falta NHD o NLCD).
- Ecorregión, latitud y elevación están correlacionadas entre sí, así que conviene reportarlas juntas y no como efectos independientes.

## Verificación
- `python ETL/atributos_geograficos.py` imprime un resumen: estaciones por ecorregión y Köppen, y distribución de cada distancia.
- Controles de sentido común: boyas `45xxx` con `dist_agua_km ≈ 0`; estaciones `CENTERPOINT` con `en_area_urbana = True` en su mayoría; ninguna estación con `dist_asos_km > ~100 km`; `ecoregion` sin nulos salvo las estaciones sobre el agua.
- Mapa rápido de las estaciones coloreadas por cada atributo (reutilizando el mapa de la sección 7 de `relaciones-bivariadas.ipynb`) para revisar a ojo.
- Ejecutar los notebooks de punta a punta en `geo_env`.
