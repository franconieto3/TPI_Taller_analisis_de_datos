import json
import os
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import boto3

s3_client = boto3.client('s3')

BUCKET = os.environ.get('BUCKET_NAME', 'clima-dataset-grupo2')
METADATA_KEY = os.environ.get('METADATA_KEY', 'raw/estaciones/grids_por_estacion.json')
PREFIJO = os.environ.get('PREFIJO_PRONOSTICOS', 'raw/pronosticos')
HORIZONTES = [int(h) for h in os.environ.get('HORIZONTES_HORAS', '6').split(',')]
HTTP_TIMEOUT = int(os.environ.get('HTTP_TIMEOUT', '15'))
WORKERS = int(os.environ.get('WORKERS', '8'))
USER_AGENT = os.environ.get('USER_AGENT', 'MiProyectoTallerDatos/1.0 (analista@taller.com)')


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
    # NWS usa offsets (-05:00); .fromisoformat no acepta 'Z' antes de Python 3.11.
    return datetime.fromisoformat(ts.replace('Z', '+00:00'))


def obtener_pronosticos(grid, hora_corrida):
    station_id = grid.get('station_id')
    forecast_url = grid.get('forecastHourly')
    if not forecast_url:
        return None, {'station_id': station_id, 'motivo': 'sin_forecastHourly'}

    try:
        datos = _get_json(forecast_url)
    except RuntimeError as e:
        return None, {'station_id': station_id, 'motivo': str(e)}

    props = datos.get('properties', {}) or {}
    periodos = props.get('periods', []) or []

    registros = []
    faltantes = []
    for h in HORIZONTES:
        objetivo = hora_corrida + timedelta(hours=h)
        elegido = None
        for p in periodos:
            try:
                if _parse(p['startTime']) <= objetivo < _parse(p['endTime']):
                    elegido = p
                    break
            except (ValueError, TypeError, KeyError):
                continue
        if elegido is None:
            faltantes.append(h)
            continue
        registros.append({
            'station_id': station_id,
            'fecha_extraccion_utc': hora_corrida.isoformat(),
            'horizonte_horas': h,
            'valido_desde_utc': _parse(elegido['startTime']).astimezone(timezone.utc).isoformat(),
            # Metadatos del modelo: sirven para detectar pronósticos "rancios" en el diagnóstico.
            'update_time': props.get('updateTime'),
            'generated_at': props.get('generatedAt'),
            'forecast': elegido,
        })

    error = {'station_id': station_id, 'motivo': f'sin_periodo_para_horizontes={faltantes}'} if faltantes else None
    return registros, error


def lambda_handler(event, context):
    # Anclada a la hora en punto: hace que reintentos y ejecuciones manuales
    # escriban SIEMPRE la misma key (idempotencia).
    if event and event.get('fecha'):
        hora_corrida = _parse(event['fecha'])
    else:
        hora_corrida = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    response = s3_client.get_object(Bucket=BUCKET, Key=METADATA_KEY)
    grids = json.loads(response['Body'].read().decode('utf-8'))
    if isinstance(grids, dict):
        grids = list(grids.values())

    print(f"Extrayendo pronósticos para {len(grids)} estaciones (horizontes={HORIZONTES}).")

    registros, errores = [], []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for regs, err in pool.map(lambda g: obtener_pronosticos(g, hora_corrida), grids):
            if regs:
                registros.extend(regs)
            if err:
                errores.append(err)

    # Particiones ASCII y en minúscula: Glue/Athena rechazan nombres como "año".
    base = (f"{PREFIJO}/year={hora_corrida.year}/month={hora_corrida.month:02d}"
            f"/day={hora_corrida.day:02d}/hour={hora_corrida.hour:02d}")

    # JSONL: append-friendly y directamente legible por Spark/pandas/Athena.
    cuerpo = '\n'.join(json.dumps(r, ensure_ascii=False) for r in registros)
    s3_client.put_object(Bucket=BUCKET, Key=f"{base}/pronosticos.jsonl",
                         Body=cuerpo.encode('utf-8'), ContentType='application/x-ndjson')

    # Manifiesto de calidad: sin esto, una estación caída desaparece sin dejar rastro.
    manifiesto = {
        'hora_corrida_utc': hora_corrida.isoformat(),
        'estaciones_solicitadas': len(grids),
        'estaciones_ok': len(grids) - len(errores),
        'registros': len(registros),
        'errores': errores,
    }
    s3_client.put_object(Bucket=BUCKET, Key=f"{base}/_manifiesto.json",
                         Body=json.dumps(manifiesto, ensure_ascii=False, indent=2).encode('utf-8'),
                         ContentType='application/json')

    print(json.dumps(manifiesto, ensure_ascii=False)[:2000])

    if not registros:
        raise RuntimeError("Ninguna estación devolvió pronóstico; se fuerza el reintento del scheduler.")

    return {'statusCode': 200, 'body': json.dumps(manifiesto, ensure_ascii=False)}