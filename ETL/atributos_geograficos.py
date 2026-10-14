"""Atributos geográficos por estación: agua, aeropuerto/ASOS y área urbana.

Cruza data/processed/estaciones.csv con las capas externas de data/raw/ para
sumar, por estación:

Agua (data/raw/cuerpos_de_agua, Natural Earth 10m):
  - dist_costa_km:        distancia a la línea de costa más cercana (Golfo de
                          México, para las estaciones de la región central).
  - dist_lago_grande_km:  distancia al lago grande más cercano (scalerank == 0
                          en ne_10m_lakes: Grandes Lagos, Great Salt Lake, Lake
                          of the Woods y similares; no todos los lagos).
  - dist_agua_km:         mínimo de las dos anteriores; 0 si la estación está
                          sobre el agua.
  - sobre_agua:           True si el punto cae dentro de un polígono de
                          ne_10m_lakes (cualquier lago, no solo los grandes),
                          o si estaciones.csv ya la había marcado fuera de todo
                          estado (origen_estado == 'estado más cercano (agua)':
                          boyas del Golfo o de los Grandes Lagos sin costa ni
                          lago propio en estas capas; ver armar_estaciones.py).
  - lago_cercano:         nombre del lago en el que cae la estación (NaN si no
                          cae en ninguno, incluidas las boyas del Golfo).

Aeropuerto / ASOS (data/raw/aeropuertos/airports.csv, OurAirports):
  - dist_aeropuerto_km:   distancia al aeropuerto medium/large más cercano.
                          No hay una lista separada de estaciones ASOS (la
                          red del NWS que calibra el modelo): se usa el tipo
                          de aeropuerto como proxy, porque los ASOS suelen
                          estar en aeropuertos medium/large (ver
                          plan-atributos-geograficos.md, fila "Aeropuerto").

Área urbana (data/raw/ciudades/tl_2020_us_uac20.shp, TIGER/Line 2020):
  - en_area_urbana:       True si el punto cae dentro de un Urban Area del
                          Census (urbanizada o cluster).
  - dist_area_urbana_km:  distancia al Urban Area más cercano (0 si está adentro).
  - dist_ciudad_50k_km:   ídem, pero solo contra las Urbanized Areas (UATYP20
                          == 'U': >= 50.000 habitantes). El Gazetteer de
                          lugares (2020_Gaz_place_national.txt) no trae
                          población, así que "ciudad grande" se aproxima con
                          esta categoría del Census en lugar de un umbral de
                          población.

Natural Earth es escala 1:10M: alcanza para el Golfo y los lagos grandes, no
para ríos o lagos chicos (haría falta NHD o NLCD para esos).

Requiere geopandas (correr en el entorno con geopandas instalado, no el base).

Uso (desde la raíz del proyecto):
    python ETL/atributos_geograficos.py

Salida:
    data/processed/estaciones_atributos.csv  -> station_id + las columnas de arriba
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
ESTACIONES = RAIZ / 'data/processed/estaciones.csv'
COSTA = RAIZ / 'data/raw/cuerpos_de_agua/ne_10m_coastline.shp'
LAGOS = RAIZ / 'data/raw/cuerpos_de_agua/ne_10m_lakes.shp'
AEROPUERTOS = RAIZ / 'data/raw/aeropuertos/airports.csv'
AREAS_URBANAS = RAIZ / 'data/raw/ciudades/tl_2020_us_uac20.shp'
SALIDA = RAIZ / 'data/processed/estaciones_atributos.csv'

CRS_METRICO = 'EPSG:5070'  # Albers equivalente para EE.UU. contiguo (igual que analisis_estaciones.ipynb)
SCALERANK_LAGO_GRANDE = 0  # Natural Earth: 0 = lagos más prominentes del mundo
TIPOS_AEROPUERTO = ['medium_airport', 'large_airport']  # proxy de ASOS: no hay lista separada de la IEM


def _a_gdf(estaciones):
    return gpd.GeoDataFrame(
        estaciones,
        geometry=gpd.points_from_xy(estaciones['longitud'], estaciones['latitud']),
        crs='EPSG:4326',
    )


def _distancia_km(puntos, capa):
    """Distancia (km) de cada punto al elemento más cercano de `capa` (0 si el punto cae adentro)."""
    cercano = gpd.sjoin_nearest(
        puntos.to_crs(CRS_METRICO)[['geometry']],
        capa.to_crs(CRS_METRICO)[['geometry']],
        how='left', distance_col='dist_m',
    )
    cercano = cercano[~cercano.index.duplicated()]  # empates a igual distancia
    return (cercano['dist_m'] / 1000).round(3)


def calcular_atributos_agua(estaciones, puntos):
    costa = gpd.read_file(COSTA)
    lagos = gpd.read_file(LAGOS)
    lagos_grandes = lagos[lagos['scalerank'] == SCALERANK_LAGO_GRANDE]

    atributos = estaciones[['station_id']].copy()
    atributos['dist_costa_km'] = _distancia_km(puntos, costa)
    atributos['dist_lago_grande_km'] = _distancia_km(puntos, lagos_grandes)

    dentro_de_lago = gpd.sjoin(puntos[['geometry']], lagos[['name', 'geometry']],
                               how='left', predicate='within')
    dentro_de_lago = dentro_de_lago[~dentro_de_lago.index.duplicated()]
    atributos['lago_cercano'] = dentro_de_lago['name']

    # Boyas del Golfo o de los Grandes Lagos que no cayeron dentro de ningún
    # polígono de ne_10m_lakes (el Golfo no tiene polígono en estas capas):
    # ya quedaron marcadas en armar_estaciones.py al no caer en ningún estado.
    boya_en_golfo_o_lago = estaciones['origen_estado'].eq('estado más cercano (agua)')
    atributos['sobre_agua'] = atributos['lago_cercano'].notna() | boya_en_golfo_o_lago

    atributos['dist_agua_km'] = atributos[['dist_costa_km', 'dist_lago_grande_km']].min(axis=1)
    atributos.loc[atributos['sobre_agua'], 'dist_agua_km'] = 0.0

    return atributos[['station_id', 'dist_costa_km', 'dist_lago_grande_km',
                      'dist_agua_km', 'sobre_agua', 'lago_cercano']]


def calcular_atributos_aeropuerto(puntos):
    aeropuertos = pd.read_csv(AEROPUERTOS, usecols=['type', 'iso_country', 'latitude_deg', 'longitude_deg'])
    aeropuertos = aeropuertos[aeropuertos['iso_country'].eq('US') & aeropuertos['type'].isin(TIPOS_AEROPUERTO)]
    aeropuertos_gdf = gpd.GeoDataFrame(
        aeropuertos,
        geometry=gpd.points_from_xy(aeropuertos['longitude_deg'], aeropuertos['latitude_deg']),
        crs='EPSG:4326',
    )
    return _distancia_km(puntos, aeropuertos_gdf).rename('dist_aeropuerto_km')


def calcular_atributos_urbanos(puntos):
    areas = gpd.read_file(AREAS_URBANAS)[['UATYP20', 'geometry']]
    areas_grandes = areas[areas['UATYP20'].eq('U')]  # Urbanized Area (Census): >= 50.000 hab.

    dist_area_urbana_km = _distancia_km(puntos, areas)
    dist_ciudad_50k_km = _distancia_km(puntos, areas_grandes)

    return pd.DataFrame({
        'en_area_urbana': dist_area_urbana_km.eq(0),
        'dist_area_urbana_km': dist_area_urbana_km,
        'dist_ciudad_50k_km': dist_ciudad_50k_km,
    })


def main():
    estaciones = pd.read_csv(ESTACIONES)
    puntos = _a_gdf(estaciones)

    atributos = pd.concat([
        calcular_atributos_agua(estaciones, puntos),
        calcular_atributos_aeropuerto(puntos),
        calcular_atributos_urbanos(puntos),
    ], axis=1)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    atributos.to_csv(SALIDA, index=False)

    # Resumen
    print(f'Estaciones sobre el agua: {atributos["sobre_agua"].sum()} de {len(atributos)}')
    print(atributos.loc[atributos['sobre_agua'], 'lago_cercano'].value_counts(dropna=False).to_string())
    print(f'\nEstaciones en área urbana (Census Urban Area): {atributos["en_area_urbana"].sum()} de {len(atributos)}')
    print('\nDistancias (km):')
    print(atributos[['dist_costa_km', 'dist_lago_grande_km', 'dist_agua_km',
                     'dist_aeropuerto_km', 'dist_area_urbana_km', 'dist_ciudad_50k_km']]
          .describe().round(1).to_string())
    print(f'\nArchivo escrito: {SALIDA.relative_to(RAIZ)} ({len(atributos)} estaciones)')


if __name__ == '__main__':
    main()
