"""Precómputo de la fuente de datos del dashboard (Etapa C del TP3).

Qué hace: replica exactamente la carga de features de `7_modelado_global.ipynb` / `8_interpretabilidad.ipynb`,
carga los dos modelos ya entrenados de `data/processed/modelos/`, predice una sola vez, y deja en
`data/processed/dashboard/` archivos chicos y autocontenidos que el dashboard sólo lee y filtra.

Por qué existe (ver `plan-dashboard.md`, sección 6): el modelo de mínimo error pesa ~1,1 GB y
`dataset_modelado.csv` 170 MB. Si el dashboard cargara eso en vivo, cada rerun de Streamlit tardaría minutos y
necesitaría sklearn, shap y los .joblib en el entorno de despliegue. Este script traza la frontera: todo lo que
requiere el modelo pasa acá (offline, una vez), y el dashboard queda siendo un lector de tablas de pocos MB.

Uso (desde cualquier directorio):
    python TP3/dashboard/precomputo.py
    python TP3/dashboard/precomputo.py --muestra-shap 0 --sin-intervalos   # corrida rápida, sin SHAP ni bandas

Convención de signos, igual que en todo el pipeline: error = pronosticado − observado (positivo = sobreestima).
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent          # .../TP3
PROCESSED = RAIZ / 'data' / 'processed'
MODELOS_DIR = PROCESSED / 'modelos'
SALIDA_DEFAULT = PROCESSED / 'dashboard'

VARIABLE_OBJETIVO = 'temp_c'
COL_ERROR = f'error_{VARIABLE_OBJETIVO}'

# Mismas listas que 7_modelado_global.ipynb y 8_interpretabilidad.ipynb. Tienen que coincidir exactamente: cada
# .joblib incluye su propio ColumnTransformer ya ajustado a estos nombres de columna.
FEATURES_NUMERICAS = [
    'error_temp_c_lag6', 'error_temp_c_lag24', 'error_temp_c_roll3_mean',
    'error_temp_c_vecinas_lag6_mean',
    'obs_wind_direction_sin_lag6', 'obs_wind_direction_cos_lag6',
    'obs_rh_pct_lag6', 'obs_dewpoint_c_lag6', 'obs_windspeed_kmh_lag6',
    'hora_sin', 'hora_cos',
    'elevation', 'dist_agua_km', 'dist_aeropuerto_km', 'dist_area_urbana_km', 'area_urbana_km2',
    'fcst_temp_c', 'fcst_pop_pct', 'fcst_rh_pct', 'fcst_windspeed_kmh',
]
FEATURES_BINARIAS = ['sobre_agua', 'en_area_urbana']
FEATURES_CATEGORICAS = ['bioma']
FEATURES = FEATURES_NUMERICAS + FEATURES_BINARIAS + FEATURES_CATEGORICAS

# Columnas de dataset_modelado.csv: las mismas que usan las notebooks 7 y 8, más las que el dashboard necesita
# para filtrar y mapear (obs, lat/lon, estado) y para declarar qué filas tienen la observación imputada.
COLS_MODELADO = [
    'station_id', 'target_hour_utc', COL_ERROR, f'abs_{COL_ERROR}', f'target_valido_{VARIABLE_OBJETIVO}',
    'hora_local', 'hora_sin', 'hora_cos', 'elevation',
    'dist_agua_km', 'dist_aeropuerto_km', 'dist_area_urbana_km', 'area_urbana_km2',
    'sobre_agua', 'en_area_urbana', 'bioma', 'fcst_temp_c', 'fcst_pop_pct',
    'fcst_rh_pct', 'fcst_windspeed_kmh', 'split',
    'obs_temp_c', 'obs_temp_c_imputado', 'latitud', 'longitud', 'estado',
]

# Franja horaria: ordinal, para el filtro del Nivel 2 y para agrupar los intervalos empíricos.
FRANJAS = [(0, 6, 'madrugada (0-5h)'), (6, 12, 'mañana (6-11h)'),
           (12, 18, 'tarde (12-17h)'), (18, 24, 'noche (18-23h)')]

UMBRAL_KPI = 1.0        # °C: umbral del KPI en encuadre de frecuencia ("X de cada 10 horas con error <= 1 °C")
NIVEL_INTERVALO = 0.80  # cobertura nominal de la banda empírica


def franja_horaria(hora: pd.Series) -> pd.Series:
    """Hora local (0-23) -> franja ordinal con las categorías en orden natural, no alfabético."""
    etiquetas = [nombre for _, _, nombre in FRANJAS]
    salida = pd.Series(pd.NA, index=hora.index, dtype='object')
    for desde, hasta, nombre in FRANJAS:
        salida[(hora >= desde) & (hora < hasta)] = nombre
    return pd.Categorical(salida, categories=etiquetas, ordered=True)


def cargar_datos() -> pd.DataFrame:
    """Replica la carga de las notebooks 7 y 8: panel de features + columnas de dataset_modelado, misma máscara."""
    features = pd.read_csv(PROCESSED / 'panel_features_error_temp_c.csv', parse_dates=['target_hour_utc'])
    # features trae su propia versión de error_temp_c (con interpolación acotada, para el diagnóstico ACF, no como
    # target): se dropea para que la única fuente del target sea dataset_modelado.csv (hallazgo 1).
    features = features.drop(columns=COL_ERROR)

    modelado = pd.read_csv(PROCESSED / 'dataset_modelado.csv', usecols=COLS_MODELADO,
                           parse_dates=['target_hour_utc'])

    assert not features.duplicated(['station_id', 'target_hour_utc']).any(), \
        'panel_features_error_temp_c tiene pares estación+hora repetidos'
    assert not modelado.duplicated(['station_id', 'target_hour_utc']).any(), \
        'dataset_modelado tiene pares estación+hora repetidos'

    datos = features.merge(modelado, on=['station_id', 'target_hour_utc'])
    datos[['sobre_agua', 'en_area_urbana']] = datos[['sobre_agua', 'en_area_urbana']].astype(int)

    mascara = datos[f'target_valido_{VARIABLE_OBJETIVO}'] & datos[FEATURES].notna().all(axis=1)
    datos = datos[mascara].copy().sort_values('target_hour_utc').reset_index(drop=True)
    print(f'{len(datos):,} filas válidas tras el merge y la máscara de validez')
    return datos


def predecir(datos: pd.DataFrame, modelos: dict, split: str) -> pd.DataFrame:
    """Predice el error con cada modelo sobre un split y devuelve la tabla lista para el dashboard."""
    d = datos[datos['split'] == split]
    if d.empty:
        return pd.DataFrame()
    X = d[FEATURES]
    print(f'  prediciendo sobre {split}: {len(d):,} filas')

    salida = pd.DataFrame({
        'station_id': d['station_id'].values,
        'target_hour_utc': d['target_hour_utc'].values,
        'hora_local': d['hora_local'].astype('int8').values,
        'franja': franja_horaria(d['hora_local']),
        'estado': d['estado'].values,
        'bioma': d['bioma'].values,
        'fcst_pop_pct': d['fcst_pop_pct'].values,
        'obs_temp_c': d['obs_temp_c'].values,
        'obs_imputado': d['obs_temp_c_imputado'].astype(bool).values,
        'fcst_temp_c': d['fcst_temp_c'].values,
        'error_crudo': d[COL_ERROR].values,                      # fcst - obs, sin corregir
        'error_naive_lag6': d['error_temp_c_lag6'].values,        # referencia de persistencia, para el MASE
    })

    for clave, modelo in modelos.items():
        pred = modelo.predict(X)
        salida[f'pred_error_{clave}'] = pred
        salida[f'fcst_corregido_{clave}'] = salida['fcst_temp_c'] - pred   # porque error = fcst - obs
        salida[f'error_corregido_{clave}'] = salida['error_crudo'] - pred  # residuo, en la misma convención
    return salida


def bandas_empiricas(test: pd.DataFrame, clave: str, dias_calibracion: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Intervalo empírico estilo split-conformal, calibrado sobre los primeros `dias_calibracion` días del test.

    Por qué no se calibra en train (primera versión de este script, 2026-11-16): el Random Forest está sobreajustado
    en train (MAE 0,661 contra 1,141 en test), así que sus residuos de train son demasiado angostos -la banda daba
    ±1,0 °C y la cobertura observada en test fue del 55,4% contra el 80% nominal. Split-conformal necesita un set de
    calibración que el modelo no haya visto. Reentrenar el bosque dejando afuera una ventana de calibración sería lo
    canónico, pero pesa 1,1 GB: la alternativa barata y correcta es partir el test en dos tramos cronológicos,
    calibrar en el primero y medir cobertura en el segundo.

    La banda se aplica a todas las filas del test (el dashboard las muestra en todo el período), pero la columna
    `tramo` declara cuáles se usaron para calibrar: la cobertura sólo se informa sobre `evaluacion`.
    """
    col = f'error_corregido_{clave}'
    alfa = (1 - NIVEL_INTERVALO) / 2

    corte = test['target_hour_utc'].min() + pd.Timedelta(days=dias_calibracion)
    test = test.copy()
    test['tramo'] = np.where(test['target_hour_utc'] < corte, 'calibracion', 'evaluacion')
    calib = test[test['tramo'] == 'calibracion']
    if calib.empty or (test['tramo'] == 'evaluacion').sum() == 0:
        raise ValueError(f'--dias-calibracion {dias_calibracion} no deja los dos tramos: el test va de '
                         f'{test["target_hour_utc"].min()} a {test["target_hour_utc"].max()}')

    cuantiles = (calib.groupby(['bioma', 'franja'], observed=True)[col]
                 .quantile([alfa, 1 - alfa]).unstack()
                 .rename(columns={alfa: 'q_lo', 1 - alfa: 'q_hi'}).reset_index())
    q_lo_global, q_hi_global = calib[col].quantile([alfa, 1 - alfa])

    test = test.merge(cuantiles, on=['bioma', 'franja'], how='left')
    test['q_lo'] = test['q_lo'].fillna(q_lo_global)
    test['q_hi'] = test['q_hi'].fillna(q_hi_global)
    # obs = fcst_corregido - error_corregido, así que el intervalo del error se invierte al pasarlo a temperatura.
    test['obs_lo'] = test[f'fcst_corregido_{clave}'] - test['q_hi']
    test['obs_hi'] = test[f'fcst_corregido_{clave}'] - test['q_lo']
    test = test.drop(columns=['q_lo', 'q_hi'])
    return test, cuantiles


def calcular_metricas(test: pd.DataFrame, claves: dict, cobertura: float | None, dias_calibracion: int) -> dict:
    """KPIs del Nivel 1 del dashboard, calculados una sola vez acá para que los paneles no los recalculen."""
    mae_crudo = test['error_crudo'].abs().mean()
    mae_naive = (test['error_crudo'] - test['error_naive_lag6']).abs().mean()

    modelos = {'cero (no corregir)': test['error_crudo']}
    for clave, etiqueta in claves.items():
        modelos[etiqueta] = test[f'error_corregido_{clave}']

    filas = []
    for etiqueta, serie in modelos.items():
        mae = serie.abs().mean()
        filas.append({
            'modelo': etiqueta,
            'MAE': round(float(mae), 4),
            'RMSE': round(float(np.sqrt((serie ** 2).mean())), 4),
            'sesgo_medio': round(float(serie.mean()), 4),
            'p90_abs': round(float(serie.abs().quantile(0.90)), 4),
            'mejora_vs_cero_pct': round(float(100 * (1 - mae / mae_crudo)), 2),
            'mejora_vs_cero_c': round(float(mae_crudo - mae), 4),
            f'pct_error_menor_{UMBRAL_KPI}c': round(float(100 * (serie.abs() <= UMBRAL_KPI).mean()), 2),
            'MASE': round(float(mae / mae_naive), 4),
        })

    return {
        'variable': VARIABLE_OBJETIVO,
        'filas_test': int(len(test)),
        'estaciones_test': int(test['station_id'].nunique()),
        'desde': str(test['target_hour_utc'].min()),
        'hasta': str(test['target_hour_utc'].max()),
        'mae_naive_persistencia': round(float(mae_naive), 4),
        'umbral_kpi_c': UMBRAL_KPI,
        'nivel_intervalo': NIVEL_INTERVALO,
        'dias_calibracion': dias_calibracion,
        'cobertura_observada': None if cobertura is None else round(float(cobertura), 4),
        'ancho_medio_banda_c': (None if 'obs_hi' not in test
                                else round(float((test['obs_hi'] - test['obs_lo']).mean()), 4)),
        'modelos': filas,
    }


def shap_muestra(modelo, datos_test: pd.DataFrame, n: int, semilla: int = 0) -> pd.DataFrame | None:
    """Valores SHAP por fila para el beeswarm del Nivel 2, sobre una muestra del test.

    TreeSHAP escala con el tamaño de los árboles además de con la cantidad de filas, y el Random Forest reentrenado
    sobre el dataset extendido es muy grande (~1,1 GB): la notebook 8 bajó la muestra de 500 a 100 por eso mismo.
    Mantener n chico; n=0 saltea el paso.
    """
    if n <= 0:
        return None
    try:
        import shap
    except ImportError:
        print('  shap no instalado: se saltea el beeswarm (el dashboard puede usar las tablas de data/out/)')
        return None

    prep, estimador = modelo.named_steps['prep'], modelo.named_steps['modelo']
    if not hasattr(estimador, 'estimators_') and not hasattr(estimador, 'tree_'):
        print(f'  {type(estimador).__name__} no es de árbol/ensamble: TreeExplainer no aplica, se saltea')
        return None

    nombres = prep.get_feature_names_out()
    X = datos_test[FEATURES]
    n = min(n, len(X))
    X_m = X.sample(n, random_state=semilla)
    X_t = prep.transform(X_m)
    if hasattr(X_t, 'toarray'):
        X_t = X_t.toarray()

    print(f'  SHAP (TreeExplainer) sobre {n:,} filas -puede tardar varios minutos')
    valores = shap.TreeExplainer(estimador).shap_values(X_t)

    largo = pd.DataFrame(valores, columns=[f'shap__{c}' for c in nombres], index=X_m.index)
    largo = pd.concat([pd.DataFrame(X_t, columns=[f'valor__{c}' for c in nombres], index=X_m.index), largo], axis=1)
    largo.insert(0, 'station_id', datos_test.loc[X_m.index, 'station_id'].values)
    largo.insert(1, 'target_hour_utc', datos_test.loc[X_m.index, 'target_hour_utc'].values)
    return largo.reset_index(drop=True)


def contorno_estados(salida: Path, tolerancia: float = 0.02) -> None:
    """Contorno de los 15 estados, simplificado, para el mapa del dashboard.

    `estados_central.geojson` pesa 8,2 MB: demasiado para que Altair lo dibuje en cada rerun. A la escala del mapa
    (una región de ~20° de ancho en una figura de ~700 px) una tolerancia de 0,02° es imperceptible.
    """
    try:
        import geopandas as gpd
    except ImportError:
        print('  geopandas no instalado: se saltea el contorno (el mapa queda sin bordes de estado)')
        return
    estados = gpd.read_file(PROCESSED / 'estados_central.geojson')
    estados['geometry'] = estados.geometry.simplify(tolerancia, preserve_topology=True)
    destino = salida / 'estados.geojson'
    estados.to_file(destino, driver='GeoJSON')
    print(f'  {len(estados)} estados simplificados: {destino.stat().st_size / 1e6:.2f} MB '
          f'(original {(PROCESSED / "estados_central.geojson").stat().st_size / 1e6:.1f} MB)')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--salida', type=Path, default=SALIDA_DEFAULT, help='directorio de salida')
    ap.add_argument('--muestra-shap', type=int, default=100,
                    help='filas para los SHAP por fila del beeswarm (0 = saltear)')
    ap.add_argument('--dias-calibracion', type=int, default=2,
                    help='días iniciales del test que se usan para calibrar las bandas (el resto mide cobertura)')
    ap.add_argument('--sin-intervalos', action='store_true',
                    help='saltea las bandas empíricas y la cobertura')
    args = ap.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)

    print('1. Cargando features')
    datos = cargar_datos()

    print('2. Cargando modelos de Etapa A (no se reentrena nada)')
    modelos = {
        'interpretable': joblib.load(MODELOS_DIR / f'modelo_interpretable_{VARIABLE_OBJETIVO}.joblib'),
        'min_error': joblib.load(MODELOS_DIR / f'modelo_min_error_{VARIABLE_OBJETIVO}.joblib'),
    }
    claves = {clave: f'{clave} ({type(m.named_steps["modelo"]).__name__})' for clave, m in modelos.items()}
    for etiqueta in claves.values():
        print(f'  {etiqueta}')

    print('3. Prediciendo')
    test = predecir(datos, modelos, 'test')

    cobertura = None
    if args.sin_intervalos:
        print('4. Bandas empíricas: salteadas (--sin-intervalos)')
    else:
        print(f'4. Bandas empíricas: calibración en los primeros {args.dias_calibracion} día(s) del test')
        test, cuantiles = bandas_empiricas(test, 'min_error', args.dias_calibracion)
        evaluacion = test[test['tramo'] == 'evaluacion']
        dentro = (evaluacion['obs_temp_c'] >= evaluacion['obs_lo']) & (evaluacion['obs_temp_c'] <= evaluacion['obs_hi'])
        cobertura = float(dentro.mean())
        cuantiles.to_csv(args.salida / 'intervalos_empiricos.csv', index=False)
        print(f'  calibración: {len(test) - len(evaluacion):,} filas | evaluación: {len(evaluacion):,} filas')
        print(f'  ancho medio de la banda: {(test["obs_hi"] - test["obs_lo"]).mean():.2f} °C')
        print(f'  cobertura observada fuera de calibración: {100 * cobertura:.1f}% '
              f'(nominal {100 * NIVEL_INTERVALO:.0f}%)')

    print('5. Escribiendo tablas del dashboard')
    test.to_parquet(args.salida / 'predicciones_test.parquet', index=False)

    estaciones = (datos[datos['split'] == 'test']
                  .groupby('station_id', as_index=False)
                  .agg(estado=('estado', 'first'), bioma=('bioma', 'first'),
                       latitud=('latitud', 'first'), longitud=('longitud', 'first'),
                       elevation=('elevation', 'first'), dist_agua_km=('dist_agua_km', 'first'),
                       horas_test=('target_hour_utc', 'size')))
    estaciones.to_parquet(args.salida / 'estaciones.parquet', index=False)

    metricas = calcular_metricas(test, claves, cobertura, args.dias_calibracion)
    (args.salida / 'metricas.json').write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding='utf-8')

    contorno_estados(args.salida)

    print('6. SHAP por fila (modelo de mínimo error)')
    shap_df = shap_muestra(modelos['min_error'], datos[datos['split'] == 'test'], args.muestra_shap)
    if shap_df is not None:
        shap_df.to_parquet(args.salida / 'shap_muestra.parquet', index=False)

    print(f'\nListo. Archivos en {args.salida}')
    for archivo in sorted(args.salida.iterdir()):
        print(f'  {archivo.name:32s} {archivo.stat().st_size / 1e6:7.2f} MB')
    for fila in metricas['modelos']:
        print(f'  {fila["modelo"]:45s} MAE {fila["MAE"]:.3f} °C  ({fila["mejora_vs_cero_pct"]:+.1f}% vs. no corregir)')


if __name__ == '__main__':
    main()
