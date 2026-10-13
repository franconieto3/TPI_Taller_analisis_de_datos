"""Arma el dataset de atributos de estación, con la elevación.

Parte de data/processed/estaciones_central.geojson (y no de la metadata cruda)
porque ahí el estado ya está corregido: la metadata trae 31 estaciones con
estado 'N/A', que analisis_estaciones.ipynb asignó por polígono o por el estado
más cercano (columna origen_estado).

Elevación: viene de api.weather.gov en metros (son pies convertidos, por eso
los valores tipo 14.0208 = 46 ft). Las redes ARDOT y RDOT no la informan y la
API devuelve 0: son 36 estaciones de Arkansas tierra adentro, así que el 0 se
pasa a NaN.

Uso (desde la raíz del proyecto):
    python ETL/armar_estaciones.py

Salida:
    data/processed/estaciones.csv   -> una fila por estación (500)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
ESTACIONES_GEO = RAIZ / 'data/processed/estaciones_central.geojson'
SALIDA = RAIZ / 'data/processed/estaciones.csv'

COLUMNAS = [
    'station_id', 'nombre', 'estado', 'origen_estado', 'timeZone',
    'provider', 'subProvider', 'latitud', 'longitud', 'elevation', 'categoria',
]


def main():
    features = json.loads(ESTACIONES_GEO.read_text(encoding='utf-8'))['features']
    estaciones = pd.DataFrame([f['properties'] for f in features]).rename(columns={'id': 'station_id'})

    estaciones['nombre'] = estaciones['nombre'].str.strip()
    ceros = estaciones['elevation'] == 0
    estaciones['elevation'] = estaciones['elevation'].mask(ceros, np.nan)

    estaciones = estaciones[COLUMNAS].sort_values('station_id')
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    estaciones.to_csv(SALIDA, index=False)

    # Resumen
    print(f'Elevación en 0 -> NaN: {ceros.sum()} estaciones')
    print(estaciones.loc[ceros, 'subProvider'].value_counts().to_string())
    print('\nElevación (m) por categoría:')
    print(estaciones.groupby('categoria')['elevation'].describe()[['count', 'min', '50%', 'max']].round(1).to_string())
    print(f'\nArchivo escrito: {SALIDA.relative_to(RAIZ)} ({len(estaciones)} estaciones)')


if __name__ == '__main__':
    main()
