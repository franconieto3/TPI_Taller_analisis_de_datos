"""Clasifica las estaciones según qué variables meteorológicas reportan realmente.

Muchas estaciones de la metadata son hidrológicas (pluviómetros, medidores de
nivel de ríos: HADS, HCFCD, AUSTIN, ...). api.weather.gov les publica
observaciones, pero con todas las variables meteorológicas en null, y el ETL
las guarda igual. Este script mide la cobertura de cada estación en el dataset
consolidado y exporta las que conviene excluir.

Uso (desde la raíz del proyecto):
    python ETL/clasificar_estaciones.py [--umbral 0.01] [--verificar-api]

Salidas:
    data/processed/estaciones_sin_meteo.csv   -> lista para filtrar
    data/processed/cobertura_estaciones.csv   -> cobertura por variable de las 500 estaciones
"""
import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
MEDICIONES = RAIZ / 'data/raw/mediciones-pronosticos.csv'
METADATA = RAIZ / 'data/raw/metadata_estaciones.json'
SALIDA_EXCLUIR = RAIZ / 'data/processed/estaciones_sin_meteo.csv'
SALIDA_COBERTURA = RAIZ / 'data/processed/cobertura_estaciones.csv'

# obs_heat_index_c queda afuera: es derivada y la API la deja en null cuando no
# hace calor, así que su ausencia no indica falta de sensor.
VARIABLES_OBS = [
    'obs_temp_c', 'obs_dewpoint_c', 'obs_rh_pct',
    'obs_windspeed_kmh', 'obs_wind_direction_d',
]
UMBRAL_INTERMITENTE = 0.90

# Verificación contra la API (mismo User-Agent y backoff que get_observations.py)
USER_AGENT = 'MiProyectoTallerDatos/1.0 (analista@taller.com)'
HTTP_TIMEOUT = 20
WORKERS = 4
VARIABLES_API = ['temperature', 'dewpoint', 'relativeHumidity', 'windSpeed']


def calcular_cobertura(mediciones):
    """Fracción de horas con dato por estación y variable."""
    agrupado = mediciones.groupby('station_id')
    cobertura = agrupado[VARIABLES_OBS].agg(lambda s: s.notna().mean())
    cobertura.columns = [f'cob_{c.removeprefix("obs_")}' for c in VARIABLES_OBS]
    cobertura.insert(0, 'n_horas', agrupado.size())
    cobertura['cobertura_max'] = cobertura.filter(like='cob_').max(axis=1)
    return cobertura.reset_index()


def clasificar(fila, umbral):
    if pd.isna(fila['n_horas']):
        return 'sin_registros', 'sin filas en el dataset (le falta el pronóstico o la observación)', ''

    columnas_cob = [c for c in fila.index if c.startswith('cob_')]
    sin_dato = [c.removeprefix('cob_') for c in columnas_cob if fila[c] < umbral]

    if fila['cobertura_max'] < umbral:
        return 'sin_meteo', 'nunca reporta variables meteorológicas', ''
    if fila['cobertura_max'] < UMBRAL_INTERMITENTE:
        return 'intermitente', 'reporta, pero con huecos frecuentes', ','.join(sin_dato)
    if sin_dato:
        return 'parcial_por_variable', 'no mide algunas variables', ','.join(sin_dato)
    return 'ok', '', ''


def _get_json(url, intentos=3):
    """GET vía curl: el ssl de Python en este equipo no valida api.weather.gov."""
    ultimo_error = None
    for intento in range(intentos):
        res = subprocess.run(
            ['curl', '-s', '-f', '-m', str(HTTP_TIMEOUT), '-H', f'User-Agent: {USER_AGENT}', url],
            capture_output=True,
        )
        if res.returncode == 0:
            return json.loads(res.stdout.decode('utf-8'))
        ultimo_error = f'curl exit {res.returncode}'
        time.sleep(2 ** intento)
    raise RuntimeError(ultimo_error)


def verificar_en_api(station_id):
    """Fracción de las últimas 50 observaciones con alguna variable meteorológica.

    0 confirma que la estación no mide. Un valor bajo pero positivo indica una
    estación que sí mide pero intercala registros vacíos (la foto horaria de
    /observations/latest suele caer en uno vacío). None = sin observaciones.
    """
    url = f'https://api.weather.gov/stations/{station_id}/observations?limit=50'
    try:
        features = _get_json(url).get('features', [])
    except RuntimeError:
        return None
    if not features:
        return None
    con_dato = sum(
        any((f['properties'].get(v) or {}).get('value') is not None for v in VARIABLES_API)
        for f in features
    )
    return round(con_dato / len(features), 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--umbral', type=float, default=0.01,
                        help='cobertura mínima para considerar que una variable se mide (default 0.01)')
    parser.add_argument('--verificar-api', action='store_true',
                        help='confirma contra api.weather.gov las estaciones a excluir')
    args = parser.parse_args()

    # dtype=str: alcanza con saber si la celda está vacía (evita lidiar con los '-5.56)
    mediciones = pd.read_csv(MEDICIONES, usecols=['station_id', *VARIABLES_OBS], dtype=str)
    metadata = pd.DataFrame(json.loads(METADATA.read_text(encoding='utf-8')))
    metadata = metadata[['id', 'nombre', 'provider', 'subProvider', 'estado']].rename(columns={'id': 'station_id'})

    # Merge desde la metadata para que también aparezcan las estaciones sin filas
    estaciones = metadata.merge(calcular_cobertura(mediciones), on='station_id', how='left')
    estaciones[['categoria', 'motivo', 'variables_sin_dato']] = estaciones.apply(
        lambda f: pd.Series(clasificar(f, args.umbral)), axis=1)
    estaciones['n_horas'] = estaciones['n_horas'].astype('Int64')

    excluir = estaciones[estaciones['categoria'].isin(['sin_meteo', 'sin_registros'])].copy()
    if args.verificar_api:
        print(f'Verificando {len(excluir)} estaciones contra api.weather.gov...')
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            excluir['api_frac_con_dato'] = list(pool.map(verificar_en_api, excluir['station_id']))

    columnas_excluir = ['station_id', 'nombre', 'provider', 'subProvider', 'estado',
                        'n_horas', 'cobertura_max', 'categoria', 'motivo']
    if args.verificar_api:
        columnas_excluir.append('api_frac_con_dato')

    SALIDA_EXCLUIR.parent.mkdir(parents=True, exist_ok=True)
    excluir[columnas_excluir].sort_values(['categoria', 'station_id']).to_csv(SALIDA_EXCLUIR, index=False)
    estaciones.round(4).sort_values('station_id').to_csv(SALIDA_COBERTURA, index=False)

    # Resumen
    print('\nEstaciones por categoría:')
    print(estaciones['categoria'].value_counts().to_string())
    print('\nCategoría por proveedor:')
    print(pd.crosstab(estaciones['provider'].replace('', '(sin proveedor)'), estaciones['categoria']).to_string())
    if args.verificar_api:
        print('\nVerificación en la API (fracción de las últimas 50 obs. con dato; 0 = no mide):')
        print(excluir.groupby('categoria')['api_frac_con_dato'].describe()[['count', 'min', '50%', 'max']].to_string())
        print('Excluidas que en la API sí tienen datos (revisar):')
        print(excluir.loc[excluir['api_frac_con_dato'] > 0, ['station_id', 'nombre', 'categoria', 'api_frac_con_dato']].to_string(index=False))

    vacias = mediciones[VARIABLES_OBS].isna().all(axis=1)
    quedan = ~mediciones['station_id'].isin(excluir['station_id'])
    print(f'\nFilas sin ninguna observación: {vacias.mean():.1%} del total '
          f'-> {vacias[quedan].mean():.1%} tras excluir ({(~quedan).sum():,} filas fuera)')
    print(f'\nArchivos escritos:\n  {SALIDA_EXCLUIR.relative_to(RAIZ)} ({len(excluir)} estaciones)'
          f'\n  {SALIDA_COBERTURA.relative_to(RAIZ)} ({len(estaciones)} estaciones)')


if __name__ == '__main__':
    main()
