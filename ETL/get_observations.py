import json
import os
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import boto3

s3_client = boto3.client('s3')

# Configuración por variables de entorno (Idéntico a pronósticos)
BUCKET = os.environ.get('BUCKET_NAME', 'clima-dataset-grupo2')
METADATA_KEY = os.environ.get('METADATA_KEY', 'raw/estaciones/metadata_estaciones.json')
PREFIJO = os.environ.get('PREFIJO_OBSERVACIONES', 'raw/observaciones')
HTTP_TIMEOUT = int(os.environ.get('HTTP_TIMEOUT', '15'))
WORKERS = int(os.environ.get('WORKERS', '8'))
USER_AGENT = os.environ.get('USER_AGENT', 'MiProyectoTallerDatos/1.0 (analista@taller.com)')

def _get_json(url, intentos=3):
    """GET con timeout explícito y backoff para 429/5xx."""
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
        
        # Backoff exponencial: 1s, 2s, 4s...
        time.sleep(2 ** intento) 
    raise RuntimeError(ultimo_error)

def _parse(ts):
    return datetime.fromisoformat(ts.replace('Z', '+00:00'))

def obtener_mediciones(estacion):
    station_id = estacion.get('id')
    if not station_id:
         return None, {'station_id': 'desconocido', 'motivo': 'sin_id'}

    obs_url = f"https://api.weather.gov/stations/{station_id}/observations/latest"
    
    try:
        datos_obs = _get_json(obs_url)
    except RuntimeError as e:
        return None, {'station_id': station_id, 'motivo': str(e)}

    propiedades = datos_obs.get('properties', {})
    
    if not propiedades:
        return None, {'station_id': station_id, 'motivo': 'sin_propiedades_recientes'}
    
    # Aseguramos que el ID exista dentro del registro final para cruces en Athena
    propiedades['station_id'] = station_id
    
    return propiedades, None

def lambda_handler(event, context):
    # Idempotencia: anclaje horario para evitar duplicados en reintentos
    if event and event.get('fecha'):
        hora_corrida = _parse(event['fecha'])
    else:
        hora_corrida = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    try:
        response = s3_client.get_object(Bucket=BUCKET, Key=METADATA_KEY)
        estaciones = json.loads(response['Body'].read().decode('utf-8'))
        
        if isinstance(estaciones, dict):
            estaciones = list(estaciones.values())

        print(f"Extrayendo observaciones para {len(estaciones)} estaciones concurrentemente.")

        registros, errores = [], []
        
        # Ejecución en paralelo
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for reg, err in pool.map(obtener_mediciones, estaciones):
                if reg:
                    registros.append(reg)
                if err:
                    errores.append(err)

        # Particionado estándar en minúsculas y sin caracteres especiales
        base = (f"{PREFIJO}/year={hora_corrida.year}/month={hora_corrida.month:02d}"
                f"/day={hora_corrida.day:02d}/hour={hora_corrida.hour:02d}")

        if registros:
            # Exportación en formato JSONL para lectura optimizada
            cuerpo = '\n'.join(json.dumps(r, ensure_ascii=False) for r in registros)
            s3_client.put_object(
                Bucket=BUCKET, 
                Key=f"{base}/observaciones.jsonl",
                Body=cuerpo.encode('utf-8'), 
                ContentType='application/x-ndjson'
            )

        manifiesto = {
            'hora_corrida_utc': hora_corrida.isoformat(),
            'estaciones_solicitadas': len(estaciones),
            'estaciones_ok': len(estaciones) - len(errores),
            'registros': len(registros),
            'errores': errores,
        }
        
        # Metadatos del Job guardados en la misma partición
        s3_client.put_object(
            Bucket=BUCKET, 
            Key=f"{base}/_manifiesto.json",
            Body=json.dumps(manifiesto, ensure_ascii=False, indent=2).encode('utf-8'),
            ContentType='application/json'
        )

        if not registros:
            raise RuntimeError("Ninguna estación devolvió observaciones; se fuerza el reintento del scheduler.")

        return {'statusCode': 200, 'body': json.dumps(manifiesto, ensure_ascii=False)}

    except Exception as e:
        print(f"Error crítico durante la ingesta: {str(e)}")
        raise e