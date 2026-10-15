"""Selecciona las estaciones que usan get_observations y get_forecasts.

Reemplaza la lista armada con /stations?state=... a secas, que tenía dos
problemas: más de 150 estaciones que no miden variables meteorológicas
(pluviómetros y limnímetros: HADS, HCFCD, ...) y una distribución muy despareja
(Houston concentraba buena parte de Texas). Esta lambda elige N_ESTACIONES con
dos condiciones:

1. Uniformidad geográfica. Cada estado recibe un cupo proporcional a su área
   (método del mayor resto) y, dentro de los cupos, las estaciones se eligen
   por muestreo del punto más lejano (farthest point sampling): en cada paso
   entra la candidata más alejada de las ya elegidas, de cualquier estado, así
   que tampoco se amontonan sobre las fronteras. La distancia se pondera por
   red (PESO_RED) para preferir aeropuertos (ASOS/AWOS) sobre estaciones
   ciudadanas (APRSWXNET) cuando están a distancias parecidas.

2. Estación activa y con variables. Antes de aceptar una candidata se
   verifica en la API que:
     - su última observación tenga menos de MAX_HORAS_SIN_DATO horas;
     - en al menos UMBRAL_HORAS_COMPLETAS de las últimas VENTANA_HORAS horas,
       la última observación de la hora (la que habría devuelto
       /observations/latest, que es lo que pide get_observations) traiga todas
       las VARIABLES_REQUERIDAS. windDirection no se exige porque la API la
       deja en null con calma o viento variable;
     - /points devuelva forecastHourly (las boyas del Golfo, por ejemplo, no
       tienen grilla de pronóstico).
   La verificación es perezosa: solo se verifican las candidatas que están por
   entrar (de a WORKERS en paralelo), no las ~12.000 de la región.

Antes de verificar se descartan, sin llamar a la API, las estaciones sin
proveedor (no publican observaciones) y las redes de REDES_EXCLUIDAS, que en un
muestreo del 14/10/2026 nunca reportaron las variables. Es solo un ahorro de
llamadas: la verificación igual descartaría cualquier estación que no mida.

Salidas en S3 (no pisan los archivos actuales; para usarlas, apuntar la
variable de entorno METADATA_KEY de get_observations y de get_forecasts a
METADATA_KEY y GRIDS_KEY de esta lambda, respectivamente):
    METADATA_KEY   -> mismo esquema que metadata_estaciones.json (campo 'id'),
                      más cobertura_horaria y dist_vecina_km
    GRIDS_KEY      -> station_id + forecastHourly por estación (lo que lee
                      get_forecasts)
    MANIFIESTO_KEY -> cupos, descartes por motivo y distancias entre vecinas

Configurar la lambda con timeout de 15 min. Si se acerca el límite, corta y
guarda lo elegido hasta ahí (por cómo funciona el muestreo, ese subconjunto
también queda bien repartido), y el manifiesto lo marca como incompleto.
"""
import heapq
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from statistics import mean, quantiles

import boto3

s3_client = boto3.client('s3')

BUCKET = os.environ.get('BUCKET_NAME', 'clima-dataset-grupo2')
METADATA_KEY = os.environ.get('METADATA_KEY', 'raw/estaciones/v2/metadata_estaciones.json')
GRIDS_KEY = os.environ.get('GRIDS_KEY', 'raw/estaciones/v2/grids_por_estacion.json')
MANIFIESTO_KEY = os.environ.get('MANIFIESTO_KEY', 'raw/estaciones/v2/_manifiesto.json')
N_ESTACIONES = int(os.environ.get('N_ESTACIONES', '500'))
ESTADOS = os.environ.get('ESTADOS', 'TX,LA,AR,MI,WI,IN,MN,IL,IA,OK,KS,NE,MO,ND,SD').split(',')
VARIABLES_REQUERIDAS = os.environ.get(
    'VARIABLES_REQUERIDAS', 'temperature,dewpoint,relativeHumidity,windSpeed').split(',')
VENTANA_HORAS = int(os.environ.get('VENTANA_HORAS', '6'))
UMBRAL_HORAS_COMPLETAS = float(os.environ.get('UMBRAL_HORAS_COMPLETAS', '0.8'))
MAX_HORAS_SIN_DATO = float(os.environ.get('MAX_HORAS_SIN_DATO', '2'))
REDES_EXCLUIDAS = set(os.environ.get(
    'REDES_EXCLUIDAS',
    'HADS,NOS-NWLON,MesoWest/UPR,MesoWest/HCFCD,MesoWest/AUSTIN,MesoWest/JCDD6,MesoWest/GBRA,'
    'MesoWest/DATAWISE,MesoWest/GPTX,MesoWest/SCAN,MesoWest/CRN,MesoWest/MIDCoEMO').split(','))
HTTP_TIMEOUT = int(os.environ.get('HTTP_TIMEOUT', '20'))
WORKERS = int(os.environ.get('WORKERS', '8'))
USER_AGENT = os.environ.get('USER_AGENT', 'MiProyectoTallerDatos/1.0 (analista@taller.com)')

API = 'https://api.weather.gov'
MARGEN_SEGUNDOS = 60  # tiempo reservado para escribir en S3 antes del timeout

# Área de cada estado en km² (EPSG:5070, data/processed/estados_central.geojson).
AREA_KM2 = {
    'TX': 685700, 'MN': 218500, 'KS': 213100, 'NE': 200300, 'SD': 199700,
    'ND': 183100, 'OK': 181000, 'MO': 180500, 'MI': 150500, 'IL': 145900,
    'IA': 145700, 'WI': 145300, 'AR': 137700, 'LA': 120900, 'IN': 93700,
}

# Multiplica la distancia en el muestreo: con peso 0.75, una estación APRS solo
# entra antes que un aeropuerto si está 1/0.75 = 1,33 veces más aislada.
PESO_RED = {
    'ASOS': 1.0, 'ASOS-HFM': 1.0, 'OTHER-MTR': 1.0, 'NonFedAWOS': 1.0,
    'APRSWXNET': 0.75,
}
PESO_RED_DEFAULT = 0.9  # RAWS, mesoredes (MesoWest/*), MARITIME, ...


def _get_json(url, intentos=3):
    """GET con timeout explícito y backoff para 429/5xx (api.weather.gov los devuelve seguido)."""
    ultimo_error = None
    for intento in range(intentos):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as res:
                return json.loads(res.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            ultimo_error = f"HTTP {e.code}"
            if e.code not in (429, 500, 502, 503, 504):
                break
        except Exception as e:
            ultimo_error = f"{type(e).__name__}: {e}"
        time.sleep(2 ** intento)
    raise RuntimeError(ultimo_error)


def _parse(ts):
    return datetime.fromisoformat(ts.replace('Z', '+00:00'))


def _distancia_km(lat1, lon1, lat2, lon2):
    """Haversine."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def _red(estacion):
    if estacion['subProvider']:
        return f"{estacion['provider']}/{estacion['subProvider']}"
    return estacion['provider']


def listar_candidatas():
    """Todas las estaciones de ESTADOS, con el esquema de metadata_estaciones.json."""
    candidatas, vistas = [], set()
    for estado in ESTADOS:
        url = f"{API}/stations?state={estado}&limit=500"
        while url:
            datos = _get_json(url)
            features = datos.get('features') or []
            if not features:  # la última página igual trae 'next', pero vacía
                break
            for f in features:
                p = f['properties']
                station_id = p['stationIdentifier']
                if station_id in vistas:  # unas pocas aparecen en dos estados
                    continue
                vistas.add(station_id)
                longitud, latitud = f['geometry']['coordinates'][:2]
                candidatas.append({
                    'id': station_id,
                    'elevation': (p.get('elevation') or {}).get('value'),
                    'nombre': p.get('name'),
                    'county': p.get('county'),
                    'forecast': p.get('forecast'),
                    'fireWeatherZone': p.get('fireWeatherZone'),
                    'estado': estado,
                    'timeZone': p.get('timeZone'),
                    'provider': p.get('provider') or '',
                    'subProvider': p.get('subProvider') or '',
                    'longitud': longitud,
                    'latitud': latitud,
                })
            url = (datos.get('pagination') or {}).get('next')
    return candidatas


def prefiltrar(candidatas):
    """Descarta sin llamar a la API las que se sabe que no sirven."""
    quedan, descartes = [], Counter()
    for est in candidatas:
        if not est['provider']:
            descartes['sin_proveedor'] += 1
        elif est['provider'] in REDES_EXCLUIDAS or _red(est) in REDES_EXCLUIDAS:
            descartes['red_excluida'] += 1
        else:
            quedan.append(est)
    return quedan, descartes


def repartir_cupos(areas, total):
    """Cupos proporcionales al área por el método del mayor resto."""
    suma = sum(areas.values())
    exactos = {e: total * a / suma for e, a in areas.items()}
    cupos = {e: int(x) for e, x in exactos.items()}
    sobrante = total - sum(cupos.values())
    for e in sorted(exactos, key=lambda e: exactos[e] - cupos[e], reverse=True)[:sobrante]:
        cupos[e] += 1
    return cupos


def verificar(estacion, ahora):
    """Chequea actividad y variables en /observations, y la grilla en /points."""
    station_id = estacion['id']
    hora_fin = ahora.replace(minute=0, second=0, microsecond=0)
    inicio = hora_fin - timedelta(hours=VENTANA_HORAS)

    try:
        obs = _get_json(f"{API}/stations/{station_id}/observations"
                        f"?start={inicio.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    except RuntimeError as e:
        return {'ok': False, 'motivo': f'observaciones: {e}'}

    # Última observación de cada hora completa de la ventana: es la que habría
    # devuelto /observations/latest al correr get_observations en la hora en punto.
    ultima_por_hora, mas_reciente = {}, None
    for f in obs.get('features') or []:
        p = f['properties']
        ts = _parse(p['timestamp']).astimezone(timezone.utc)
        mas_reciente = max(mas_reciente or ts, ts)
        hora = ts.replace(minute=0, second=0, microsecond=0)
        if hora < hora_fin and (hora not in ultima_por_hora or ts > ultima_por_hora[hora][0]):
            completa = all((p.get(v) or {}).get('value') is not None for v in VARIABLES_REQUERIDAS)
            ultima_por_hora[hora] = (ts, completa)

    if mas_reciente is None:
        return {'ok': False, 'motivo': 'sin_observaciones'}
    if (ahora - mas_reciente).total_seconds() / 3600 > MAX_HORAS_SIN_DATO:
        return {'ok': False, 'motivo': 'inactiva'}
    cobertura = sum(completa for _, completa in ultima_por_hora.values()) / VENTANA_HORAS
    if cobertura < UMBRAL_HORAS_COMPLETAS:
        return {'ok': False, 'motivo': 'variables_incompletas'}

    try:
        punto = _get_json(f"{API}/points/{estacion['latitud']:.4f},{estacion['longitud']:.4f}")
    except RuntimeError as e:
        return {'ok': False, 'motivo': f'points: {e}'}
    props = punto.get('properties') or {}
    if not props.get('forecastHourly'):
        return {'ok': False, 'motivo': 'sin_forecastHourly'}

    return {
        'ok': True,
        'cobertura_horaria': round(cobertura, 2),
        'grilla': {
            'station_id': station_id,
            'gridId': props.get('gridId'),
            'gridX': props.get('gridX'),
            'gridY': props.get('gridY'),
            'forecastHourly': props['forecastHourly'],
        },
    }


def seleccionar(candidatas, n_objetivo, ahora, limite):
    """Muestreo del punto más lejano con cupo por estado y verificación perezosa.

    En cada paso se toma la candidata activa de mayor puntaje (distancia a la
    elegida más cercana x peso de la red). Si todavía no se verificó, se
    verifican en paralelo las WORKERS mejores sin verificar: las que pasan
    quedan en caché y, como suelen estar en otros huecos del mapa, entran en
    los pasos siguientes sin llamadas extra.
    """
    areas = {e: AREA_KM2[e] for e in ESTADOS}
    cupos = repartir_cupos(areas, n_objetivo)
    agotados = set()

    peso = [PESO_RED.get(est['provider'], PESO_RED_DEFAULT) for est in candidatas]
    # Arranque: distancia al centroide de las candidatas (la primera elegida
    # queda en un extremo de la región, como en el muestreo clásico).
    lat_c = mean(est['latitud'] for est in candidatas)
    lon_c = mean(est['longitud'] for est in candidatas)
    dmin = [_distancia_km(est['latitud'], est['longitud'], lat_c, lon_c) for est in candidatas]

    verificacion = {}
    elegidas = []
    por_estado = Counter()
    activas = set(range(len(candidatas)))
    incompleta = False

    def puntaje(i):
        return dmin[i] * peso[i]

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        while True:
            activas = {i for i in activas if por_estado[candidatas[i]['estado']] < cupos[candidatas[i]['estado']]}

            # Un estado sin candidatas válidas antes de llenar su cupo cede el
            # resto a los demás, que se reparte de nuevo por área.
            con_candidatas = Counter(candidatas[i]['estado'] for i in activas)
            nuevos_agotados = {e for e in cupos if e not in agotados
                               and por_estado[e] < cupos[e] and not con_candidatas[e]}
            if nuevos_agotados:
                agotados |= nuevos_agotados
                for e in agotados:
                    cupos[e] = por_estado[e]
                resto = repartir_cupos({e: a for e, a in areas.items() if e not in agotados},
                                       n_objetivo - sum(cupos[e] for e in agotados))
                cupos.update({e: max(c, por_estado[e]) for e, c in resto.items()})
                continue

            if not activas:
                break
            if time.monotonic() > limite:
                incompleta = True
                break

            mejor = max(activas, key=puntaje)
            if mejor not in verificacion:
                lote = heapq.nlargest(WORKERS, (i for i in activas if i not in verificacion), key=puntaje)
                for i, res in zip(lote, pool.map(lambda i: verificar(candidatas[i], ahora), lote)):
                    verificacion[i] = res
                    if not res['ok']:
                        activas.discard(i)
                continue

            activas.discard(mejor)
            elegidas.append(mejor)
            por_estado[candidatas[mejor]['estado']] += 1
            lat, lon = candidatas[mejor]['latitud'], candidatas[mejor]['longitud']
            for i in activas:
                dmin[i] = min(dmin[i], _distancia_km(lat, lon, candidatas[i]['latitud'], candidatas[i]['longitud']))

    return elegidas, verificacion, cupos, incompleta


def _distancias_a_vecina(estaciones):
    """Distancia (km) de cada estación elegida a la elegida más cercana."""
    return [
        min(_distancia_km(a['latitud'], a['longitud'], b['latitud'], b['longitud'])
            for j, b in enumerate(estaciones) if j != i)
        for i, a in enumerate(estaciones)
    ]


def _put_json(key, objeto):
    s3_client.put_object(Bucket=BUCKET, Key=key,
                         Body=json.dumps(objeto, ensure_ascii=False, indent=2).encode('utf-8'),
                         ContentType='application/json')


def lambda_handler(event, context):
    inicio = time.monotonic()
    restante = context.get_remaining_time_in_millis() / 1000 if context else float('inf')
    limite = inicio + restante - MARGEN_SEGUNDOS
    ahora = datetime.now(timezone.utc)
    n_objetivo = int((event or {}).get('n_estaciones', N_ESTACIONES))

    candidatas = listar_candidatas()
    filtradas, descartes_prefiltro = prefiltrar(candidatas)
    print(f"{len(candidatas)} estaciones en {len(ESTADOS)} estados; "
          f"{len(filtradas)} candidatas tras el prefiltro {dict(descartes_prefiltro)}.")

    elegidas, verificacion, cupos, incompleta = seleccionar(filtradas, n_objetivo, ahora, limite)

    metadata, grids = [], []
    for i in sorted(elegidas, key=lambda i: filtradas[i]['id']):
        res = verificacion[i]
        metadata.append({**filtradas[i], 'cobertura_horaria': res['cobertura_horaria']})
        grids.append(res['grilla'])
    distancias = _distancias_a_vecina(metadata) if len(metadata) > 1 else []
    for est, d in zip(metadata, distancias):
        est['dist_vecina_km'] = round(d, 1)

    _put_json(METADATA_KEY, metadata)
    _put_json(GRIDS_KEY, grids)

    descartes_verificacion = Counter(
        res['motivo'].split(':')[0] for res in verificacion.values() if not res['ok'])
    seleccionadas_por_red = Counter(_red(est) for est in metadata)
    manifiesto = {
        'fecha_utc': ahora.isoformat(),
        'duracion_s': round(time.monotonic() - inicio),
        'incompleta': incompleta,
        'parametros': {
            'n_estaciones': n_objetivo, 'estados': ESTADOS,
            'variables_requeridas': VARIABLES_REQUERIDAS, 'ventana_horas': VENTANA_HORAS,
            'umbral_horas_completas': UMBRAL_HORAS_COMPLETAS,
            'max_horas_sin_dato': MAX_HORAS_SIN_DATO,
            'redes_excluidas': sorted(REDES_EXCLUIDAS), 'peso_red': PESO_RED,
            'peso_red_default': PESO_RED_DEFAULT,
        },
        'estaciones_listadas': len(candidatas),
        'descartes_prefiltro': dict(descartes_prefiltro),
        'verificadas': len(verificacion),
        'descartes_verificacion': dict(descartes_verificacion.most_common()),
        'seleccionadas': len(metadata),
        'por_estado': {
            e: {'cupo': cupos[e], 'seleccionadas': sum(est['estado'] == e for est in metadata),
                'area_km2': AREA_KM2[e]}
            for e in ESTADOS
        },
        'por_red': dict(seleccionadas_por_red.most_common()),
        # Uniformidad: con una grilla perfecta todas las distancias serían
        # iguales (CV = 0); una selección al azar ronda CV ~ 0,5.
        'dist_vecina_km': {
            'media': round(mean(distancias), 1),
            'p10_p50_p90': [round(q, 1) for q in quantiles(distancias, n=10)[::4]],
            'cv': round((sum((d - mean(distancias)) ** 2 for d in distancias) / len(distancias)) ** 0.5
                        / mean(distancias), 3),
        } if len(distancias) > 1 else None,
        'salidas': {'metadata': METADATA_KEY, 'grids': GRIDS_KEY},
    }
    _put_json(MANIFIESTO_KEY, manifiesto)

    print(json.dumps(manifiesto, ensure_ascii=False)[:3000])

    if not metadata:
        raise RuntimeError("No se seleccionó ninguna estación.")

    return {'statusCode': 200, 'body': json.dumps(manifiesto, ensure_ascii=False)}
