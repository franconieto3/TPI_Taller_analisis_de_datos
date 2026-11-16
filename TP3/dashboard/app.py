"""Dashboard de resultados del TP3 (Etapa C, entregable 4).

Estructura: tres niveles, como la Clase 13 (p.38-39) -copa de martini: el Resumen conduce con títulos que enuncian
la conclusión; Explicabilidad y Detalle se abren al lector y llevan títulos que formulan la pregunta (p.37).

No importa `sklearn` ni toca los `.joblib`: lee lo que `precomputo.py` dejó en `data/processed/dashboard/`
(3,6 MB en total, contra 170 MB de dataset y 1,1 GB de modelo). Si falta ese directorio, correr primero:

    python TP3/dashboard/precomputo.py

Convención de signos en todo el tablero: error = pronosticado − observado (positivo = sobreestima).
"""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from paleta import (AZUL, BORDE, DIVERGENTE, GRIS_NEUTRO, GRIS_TEXTO, INTERPRETABLE, MIN_ERROR, NARANJA,
                    ORDEN_FRANJAS, ORDEN_MODELOS, SECUENCIAL, SIN_CORREGIR, SUPERFICIE, escala_modelos,
                    registrar_tema)

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / 'data' / 'processed' / 'dashboard'
INTERPRETABILIDAD = RAIZ / 'data' / 'out' / 'interpretabilidad'
MODELADO_GLOBAL = RAIZ / 'data' / 'out' / 'modelado_global'
MODELADO = RAIZ / 'data' / 'out' / 'modelado'

# Columna de error con signo por modelo. El color de cada entidad es fijo (paleta.py): un filtro que cambia
# cuántas series quedan no repinta a las que sobreviven.
COL_ERROR = {
    SIN_CORREGIR: 'error_crudo',
    INTERPRETABLE: 'error_corregido_interpretable',
    MIN_ERROR: 'error_corregido_min_error',
}

# Agrupación de features redundantes ANTES del ranking (Clase 13, p.31): hora_sin/hora_cos son una sola variable,
# y los tres lags del error son "historia reciente del error". Sin agrupar, el ranking parte la hora en dos barras
# y subestima su peso.
GRUPOS_FEATURES = {
    'hora del día': ['hora_sin', 'hora_cos'],
    'historia del error': ['error_temp_c_lag6', 'error_temp_c_lag24', 'error_temp_c_roll3_mean'],
    'error de estaciones vecinas': ['error_temp_c_vecinas_lag6_mean'],
    'pronóstico externo': ['fcst_temp_c', 'fcst_pop_pct', 'fcst_rh_pct', 'fcst_windspeed_kmh'],
    'sensores locales (lag 6h)': ['obs_rh_pct_lag6', 'obs_dewpoint_c_lag6', 'obs_windspeed_kmh_lag6',
                                   'obs_wind_direction_sin_lag6', 'obs_wind_direction_cos_lag6'],
    'geografía': ['elevation', 'dist_agua_km', 'dist_aeropuerto_km', 'dist_area_urbana_km', 'area_urbana_km2',
                   'sobre_agua', 'en_area_urbana', 'bioma'],
}
A_GRUPO = {f: g for g, fs in GRUPOS_FEATURES.items() for f in fs}


# ---------------------------------------------------------------- carga

@st.cache_data
def cargar_predicciones() -> pd.DataFrame:
    d = pd.read_parquet(DATOS / 'predicciones_test.parquet')
    d['franja'] = pd.Categorical(d['franja'].astype(str), categories=ORDEN_FRANJAS, ordered=True)
    return d


@st.cache_data
def cargar_estaciones() -> pd.DataFrame:
    return pd.read_parquet(DATOS / 'estaciones.parquet')


@st.cache_data
def cargar_metricas() -> dict:
    return json.loads((DATOS / 'metricas.json').read_text(encoding='utf-8'))


@st.cache_data
def cargar_estados() -> list | None:
    archivo = DATOS / 'estados.geojson'
    if not archivo.exists():
        return None
    return json.loads(archivo.read_text(encoding='utf-8'))['features']


@st.cache_data
def cargar_shap() -> pd.DataFrame | None:
    archivo = DATOS / 'shap_muestra.parquet'
    return pd.read_parquet(archivo) if archivo.exists() else None


@st.cache_data
def cargar_csv(ruta: Path, **kwargs) -> pd.DataFrame | None:
    return pd.read_csv(ruta, **kwargs) if ruta.exists() else None


# ---------------------------------------------------------------- helpers

def error_largo(d: pd.DataFrame, por: list[str] | None = None) -> pd.DataFrame:
    """Pasa las tres columnas de error a formato largo (una fila por modelo), opcionalmente agregando por `por`."""
    por = por or []
    ids = [c for c in ['station_id', 'target_hour_utc'] if c not in por]
    piezas = []
    for modelo, col in COL_ERROR.items():
        p = d[ids + por].copy()
        p['modelo'] = modelo
        p['error'] = d[col].values
        piezas.append(p)
    largo = pd.concat(piezas, ignore_index=True)
    if not por:
        return largo
    agg = (largo.groupby(por + ['modelo'], observed=True)
           .agg(MAE=('error', lambda s: s.abs().mean()), sesgo=('error', 'mean'),
                p90=('error', lambda s: s.abs().quantile(0.90)), n=('error', 'size'))
           .reset_index())
    agg['modelo'] = pd.Categorical(agg['modelo'], categories=ORDEN_MODELOS, ordered=True)
    return agg.sort_values(por + ['modelo'])


def con_tabla(grafico: alt.Chart, tabla: pd.DataFrame, clave: str) -> None:
    """Dibuja el gráfico y deja su tabla equivalente a un clic.

    No es un adorno: el verde de la paleta queda en 2,74:1 contra el fondo (ver `paleta.py`), y la regla es que un
    contraste por debajo de 3:1 obliga a un canal de alivio -etiquetas visibles o la tabla. Además es la versión
    accesible de cualquier gráfico cuyo valor sólo se leería del tooltip.
    """
    st.altair_chart(grafico, width='stretch')
    with st.expander('Ver tabla', expanded=False):
        st.dataframe(tabla, width='stretch', hide_index=True, key=f'tabla_{clave}')


# ---------------------------------------------------------------- Nivel 1

def barras_mae(resumen: pd.DataFrame) -> alt.Chart:
    """MAE por modelo: barras desde cero -el error sí tiene cero absoluto (Clase 13, p.11)- con etiqueta directa.

    Un solo valor por categoría, así que el color no codifica magnitud: distingue el modelo de la referencia que
    hay que ganar (énfasis), no "más oscuro es más grande".
    """
    base = alt.Chart(resumen).encode(
        y=alt.Y('modelo:N', sort=alt.EncodingSortField('MAE', order='ascending'), title=None),
        x=alt.X('MAE:Q', title='MAE (°C)', scale=alt.Scale(domainMin=0, nice=True)),
        tooltip=[alt.Tooltip('modelo:N', title='modelo'), alt.Tooltip('MAE:Q', title='MAE (°C)', format='.3f'),
                 alt.Tooltip('mejora_vs_cero_pct:Q', title='mejora vs. no corregir (%)', format='.1f'),
                 alt.Tooltip('MASE:Q', format='.3f')],
    )
    barras = base.mark_bar(height=22).encode(
        color=alt.Color('modelo:N', scale=escala_modelos(resumen['modelo'].tolist()), legend=None))
    etiquetas = base.mark_text(align='left', dx=6, color=GRIS_TEXTO, fontSize=11).encode(
        text=alt.Text('etiqueta:N'))
    return (barras + etiquetas).properties(height=130)


def panel_monitoreo(d: pd.DataFrame, estacion: str, mostrar_banda: bool) -> alt.VConcatChart:
    """Dos paneles apilados con el mismo eje temporal (Clase 13, p.16): arriba nivel de temperatura, abajo el error.

    La clase es tajante: si el mensaje es el error, se grafica el error -pero la solución buena es mostrar los dos,
    porque el panel de arriba conserva la forma del día y el de abajo no obliga a estimar distancias a ojo.
    """
    s = d[d['station_id'] == estacion].sort_values('target_hour_utc')
    temp = s.melt(id_vars='target_hour_utc', value_vars=['obs_temp_c', 'fcst_temp_c', 'fcst_corregido_min_error'],
                  var_name='serie', value_name='temp_c')
    temp['serie'] = temp['serie'].map({'obs_temp_c': 'observado', 'fcst_temp_c': 'pronóstico',
                                        'fcst_corregido_min_error': 'pronóstico corregido'})

    capas = []
    if mostrar_banda and 'obs_lo' in s:
        capas.append(alt.Chart(s).mark_area(opacity=0.18, color=AZUL).encode(
            x=alt.X('target_hour_utc:T', title=None),
            y=alt.Y('obs_lo:Q', title='temperatura (°C)'), y2='obs_hi:Q'))
    capas.append(alt.Chart(temp).mark_line(interpolate='monotone').encode(
        x=alt.X('target_hour_utc:T', title=None),
        y=alt.Y('temp_c:Q', title='temperatura (°C)', scale=alt.Scale(zero=False)),
        color=alt.Color('serie:N', scale=alt.Scale(
            domain=['observado', 'pronóstico', 'pronóstico corregido'], range=[GRIS_TEXTO, NARANJA, AZUL])),
        strokeDash=alt.StrokeDash('serie:N', scale=alt.Scale(
            domain=['observado', 'pronóstico', 'pronóstico corregido'], range=[[1, 0], [4, 2], [1, 0]]),
            legend=None),
        tooltip=[alt.Tooltip('target_hour_utc:T', title='hora (UTC)', format='%d/%m %H:%M'),
                 alt.Tooltip('serie:N', title=None), alt.Tooltip('temp_c:Q', title='°C', format='.2f')]))
    arriba = alt.layer(*capas).properties(height=210, title=alt.Title(
        'Temperatura: observada, pronosticada y corregida',
        subtitle='la banda es el intervalo del 80% del modelo de mínimo error'))

    err = s.melt(id_vars='target_hour_utc', value_vars=['error_crudo', 'error_corregido_min_error'],
                 var_name='serie', value_name='error')
    err['serie'] = err['serie'].map({'error_crudo': SIN_CORREGIR, 'error_corregido_min_error': MIN_ERROR})
    cero = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color=GRIS_NEUTRO, strokeWidth=1).encode(y='y:Q')
    abajo = (cero + alt.Chart(err).mark_line(interpolate='monotone').encode(
        x=alt.X('target_hour_utc:T', title='hora (UTC)'),
        y=alt.Y('error:Q', title='error (°C)'),
        color=alt.Color('serie:N', scale=escala_modelos([SIN_CORREGIR, MIN_ERROR])),
        tooltip=[alt.Tooltip('target_hour_utc:T', title='hora (UTC)', format='%d/%m %H:%M'),
                 alt.Tooltip('serie:N', title=None), alt.Tooltip('error:Q', title='error (°C)', format='.2f')])
        ).properties(height=150, title=alt.Title(
            'El mismo período, graficado como error',
            subtitle='positivo = el pronóstico sobreestima; el cero es el acierto'))

    return alt.vconcat(arriba, abajo, spacing=8).resolve_scale(x='shared', color='independent')


def mapa_mejora(por_estacion: pd.DataFrame, estados: list | None) -> alt.Chart:
    """432 estaciones coloreadas por mejora relativa: polaridad (mejora o empeora), así que divergente con gris
    en el centro -no un secuencial, que no distinguiría el signo."""
    limite = float(por_estacion['mejora_pct'].abs().quantile(0.95))
    puntos = alt.Chart(por_estacion).mark_circle(size=42, stroke=GRIS_NEUTRO, strokeWidth=0.3).encode(
        longitude='longitud:Q', latitude='latitud:Q',
        color=alt.Color('mejora_pct:Q', title='mejora (%)',
                        scale=alt.Scale(range=DIVERGENTE, domain=[-limite, limite], clamp=True)),
        tooltip=[alt.Tooltip('station_id:N', title='estación'), alt.Tooltip('estado:N'),
                 alt.Tooltip('bioma:N'), alt.Tooltip('MAE_sin:Q', title='MAE sin corregir', format='.2f'),
                 alt.Tooltip('MAE_con:Q', title='MAE corregido', format='.2f'),
                 alt.Tooltip('mejora_pct:Q', title='mejora (%)', format='.1f'),
                 alt.Tooltip('n:Q', title='horas en test')])
    if estados:
        fondo = alt.Chart(alt.Data(values=estados)).mark_geoshape(
            fill='#f4f3f0', stroke=BORDE, strokeWidth=0.8)
        return alt.layer(fondo, puntos).project(type='mercator').properties(height=430)
    return puntos.project(type='mercator').properties(height=430)



# ---------------------------------------------------------------- Nivel 2

def barras_importancia_agrupada(perm: pd.DataFrame) -> tuple[alt.Chart, pd.DataFrame]:
    """Permutation importance agregada por grupo de variables, para los dos modelos (dos series, misma escala:
    se superponen en barras agrupadas, no se yuxtaponen -Clase 13, p.15)."""
    d = perm.rename(columns={perm.columns[0]: 'feature'}).melt(
        id_vars='feature', var_name='modelo', value_name='importancia')
    d['grupo'] = d['feature'].map(A_GRUPO).fillna('otras')
    d['modelo'] = d['modelo'].str.replace(r'^interpretable.*', INTERPRETABLE, regex=True) \
                             .str.replace(r'^mínimo error.*', MIN_ERROR, regex=True)
    agrupado = d.groupby(['grupo', 'modelo'], as_index=False)['importancia'].sum()
    orden = (agrupado[agrupado['modelo'] == MIN_ERROR].sort_values('importancia', ascending=False)['grupo'].tolist())

    grafico = alt.Chart(agrupado).mark_bar(height=11).encode(
        y=alt.Y('grupo:N', sort=orden, title=None),
        x=alt.X('importancia:Q', title='aumento del MAE al permutar el grupo (°C)',
                scale=alt.Scale(domainMin=0, nice=True)),
        yOffset=alt.YOffset('modelo:N', sort=[INTERPRETABLE, MIN_ERROR]),
        color=alt.Color('modelo:N', scale=escala_modelos([INTERPRETABLE, MIN_ERROR])),
        tooltip=[alt.Tooltip('grupo:N', title='grupo'), alt.Tooltip('modelo:N', title=None),
                 alt.Tooltip('importancia:Q', title='Δ MAE (°C)', format='.4f')],
    ).properties(height=260)
    return grafico, agrupado.round(4)


def beeswarm_shap(shap_df: pd.DataFrame, top: int = 10) -> tuple[alt.Chart, pd.DataFrame]:
    """Beeswarm: magnitud **y sentido** del efecto, que las barras de |SHAP| medio no muestran (Clase 13, p.31)."""
    cols = [c for c in shap_df.columns if c.startswith('shap__')]
    largo = []
    for c in cols:
        nombre = c.replace('shap__', '').replace('num__', '').replace('cat__', '')
        valor_col = c.replace('shap__', 'valor__')
        v = shap_df[valor_col]
        rango = v.max() - v.min()
        largo.append(pd.DataFrame({
            'feature': nombre, 'shap': shap_df[c],
            'valor_rel': (v - v.min()) / rango if rango else 0.5, 'valor': v}))
    d = pd.concat(largo, ignore_index=True)

    orden = d.assign(abs_shap=d['shap'].abs()).groupby('feature')['abs_shap'].mean().sort_values(ascending=False)
    d = d[d['feature'].isin(orden.head(top).index)]

    grafico = alt.Chart(d).mark_circle(size=26, opacity=0.6).encode(
        y=alt.Y('feature:N', sort=orden.head(top).index.tolist(), title=None),
        x=alt.X('shap:Q', title='contribución SHAP al error pronosticado (°C)'),
        yOffset=alt.YOffset('jitter:Q', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('valor_rel:Q', title='valor de la variable',
                        scale=alt.Scale(range=SECUENCIAL), legend=alt.Legend(format='.0%')),
        tooltip=[alt.Tooltip('feature:N', title='variable'),
                 alt.Tooltip('valor:Q', title='valor', format='.2f'),
                 alt.Tooltip('shap:Q', title='SHAP (°C)', format='.3f')],
    ).transform_calculate(jitter='random()').properties(height=320)
    return grafico, orden.head(top).reset_index().rename(columns={'abs_shap': '|SHAP| promedio'}).round(4)


def lineas_por_hora(agg: pd.DataFrame) -> alt.Chart:
    """MAE por hora local en orden horario -las categorías se ordenan por valor, los ciclos conservan su orden
    natural (Clase 13, p.12). Ordenar por valor acá mostraría el ranking y destruiría la forma del día."""
    seleccion = alt.selection_point(fields=['hora_local'], nearest=True, on='pointerover', empty=False)
    base = alt.Chart(agg).encode(
        x=alt.X('hora_local:Q', title='hora local', axis=alt.Axis(values=list(range(0, 24, 3)))),
        y=alt.Y('MAE:Q', title='MAE (°C)', scale=alt.Scale(domainMin=0, nice=True)),
        color=alt.Color('modelo:N', scale=escala_modelos()))
    lineas = base.mark_line(interpolate='monotone')
    puntos = base.mark_point(filled=True, size=70, stroke=SUPERFICIE, strokeWidth=2).encode(
        opacity=alt.condition(seleccion, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip('hora_local:Q', title='hora local'), alt.Tooltip('modelo:N', title=None),
                 alt.Tooltip('MAE:Q', title='MAE (°C)', format='.3f'), alt.Tooltip('n:Q', title='horas')],
    ).add_params(seleccion)
    return (lineas + puntos).properties(height=300)


def barras_por_bioma(agg: pd.DataFrame) -> alt.Chart:
    """MAE por bioma: categórico sin orden propio, así que ordenado por valor. El n va en la etiqueta porque sin
    controlar la composición la comparación entre grupos engaña (Clase 13, p.22, Simpson)."""
    orden = (agg[agg['modelo'] == MIN_ERROR].sort_values('MAE', ascending=False)['bioma'].tolist())
    etiquetas = (agg[agg['modelo'] == MIN_ERROR].assign(
        etiqueta=lambda x: 'n=' + (x['n'] / 1000).round(1).astype(str) + 'k'))
    barras = alt.Chart(agg).mark_bar(height=9).encode(
        y=alt.Y('bioma:N', sort=orden, title=None),
        x=alt.X('MAE:Q', title='MAE (°C)', scale=alt.Scale(domainMin=0, nice=True)),
        yOffset=alt.YOffset('modelo:N', sort=ORDEN_MODELOS),
        color=alt.Color('modelo:N', scale=escala_modelos()),
        tooltip=[alt.Tooltip('bioma:N'), alt.Tooltip('modelo:N', title=None),
                 alt.Tooltip('MAE:Q', title='MAE (°C)', format='.3f'), alt.Tooltip('n:Q', title='horas')])
    texto = alt.Chart(etiquetas).mark_text(align='left', dx=4, color=GRIS_TEXTO, fontSize=10).encode(
        y=alt.Y('bioma:N', sort=orden, title=None), x=alt.X('MAE:Q'), text='etiqueta:N')
    return (barras + texto).properties(height=230)


def cascada_shap_local(caso: pd.DataFrame, top: int = 8) -> tuple[alt.Chart, pd.DataFrame]:
    """Contribuciones con signo de un caso puntual: longitud desde cero, divergente por el signo (Clase 13, p.13)."""
    d = caso.rename(columns={caso.columns[0]: 'feature', caso.columns[1]: 'contribucion'}).copy()
    d['feature'] = d['feature'].str.replace('num__', '').str.replace('cat__', '')
    d = d.reindex(d['contribucion'].abs().sort_values(ascending=False).index).head(top)
    grafico = alt.Chart(d).mark_bar(height=16).encode(
        y=alt.Y('feature:N', sort=alt.EncodingSortField('contribucion', order='descending'), title=None),
        x=alt.X('contribucion:Q', title='contribución al error pronosticado (°C)'),
        color=alt.Color('contribucion:Q', scale=alt.Scale(range=DIVERGENTE, domainMid=0), legend=None),
        tooltip=[alt.Tooltip('feature:N', title='variable'),
                 alt.Tooltip('contribucion:Q', title='SHAP (°C)', format='.3f')],
    ).properties(height=240)
    return grafico, d.round(4)


# ---------------------------------------------------------------- Nivel 3

def histograma_error(d: pd.DataFrame, ancho: float) -> tuple[alt.Chart, pd.DataFrame]:
    """Distribución del error con signo, como polígono de frecuencias escalonado.

    El ancho de clase es un control explícito: la ventana puede crear un patrón o suprimirlo, y la regla es probar
    al menos dos anchos y declarar cuál se usó (Clase 13, p.21).

    Dos decisiones de forma: el conteo se agrega **en pandas**, no en el navegador (son 47.151 filas × 3 series:
    mandarlas crudas excede el límite de Altair y no hay razón para hacerlo), y las tres series van como contorno
    escalonado en vez de barras translúcidas superpuestas -tres rellenos encimados se ocluyen entre sí, tres líneas
    en la misma escala no (p.17).
    """
    largo = error_largo(d)
    borde = (largo['error'].abs().quantile(0.999) // ancho + 1) * ancho
    bins = pd.interval_range(start=-borde, end=borde, freq=ancho)
    largo = largo[largo['error'].between(-borde, borde)]
    clases = pd.cut(largo['error'], bins=bins)
    conteo = (largo.assign(centro=clases.map(lambda i: i.mid).astype(float))
              .groupby(['modelo', 'centro'], observed=True).size().reset_index(name='horas'))
    conteo['modelo'] = pd.Categorical(conteo['modelo'], categories=ORDEN_MODELOS, ordered=True)

    cero = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(color=GRIS_NEUTRO, strokeWidth=1).encode(x='x:Q')
    linea = alt.Chart(conteo).mark_line(interpolate='step-after').encode(
        x=alt.X('centro:Q', title=f'error (°C) · clases de {ancho} °C'),
        y=alt.Y('horas:Q', title='horas', scale=alt.Scale(domainMin=0, nice=True)),
        color=alt.Color('modelo:N', scale=escala_modelos()),
        tooltip=[alt.Tooltip('modelo:N', title=None), alt.Tooltip('centro:Q', title='error (°C)', format='.2f'),
                 alt.Tooltip('horas:Q', title='horas', format=',')])
    return (cero + linea).properties(height=280), conteo.sort_values(['modelo', 'centro'])


def lineas_p90(agg: pd.DataFrame) -> alt.Chart:
    """Percentil 90 del error absoluto por hora: el promedio resume, pero si la decisión depende de los extremos
    corresponde mirar la cola (Clase 13, p.20)."""
    return alt.Chart(agg).mark_line(interpolate='monotone', point=False).encode(
        x=alt.X('hora_local:Q', title='hora local', axis=alt.Axis(values=list(range(0, 24, 3)))),
        y=alt.Y('p90:Q', title='P90 del |error| (°C)', scale=alt.Scale(domainMin=0, nice=True)),
        color=alt.Color('modelo:N', scale=escala_modelos()),
        tooltip=[alt.Tooltip('hora_local:Q', title='hora'), alt.Tooltip('modelo:N', title=None),
                 alt.Tooltip('p90:Q', title='P90 (°C)', format='.3f')],
    ).properties(height=280)




# ---------------------------------------------------------------- filtros

FILTROS_DEFECTO = {'estado': [], 'bioma': [], 'franja': [], 'lluvia': 'todas las horas', 'imputadas': False}


def controles_filtros(d: pd.DataFrame, metricas: dict) -> dict:
    """Los controles de filtrado, dentro de un formulario con botón "Aplicar".

    El formulario existe para que la selección se confirme en bloque: sin él, cada clic dispara un rerun y el
    tablero se recalcula cinco veces mientras el lector arma una sola pregunta. "Limpiar" regenera las claves de
    los widgets (`gen`), que es la forma segura de volverlos a su valor por defecto sin pisar `session_state` de
    un widget ya instanciado.
    """
    st.session_state.setdefault('gen', 0)
    g = st.session_state['gen']

    with st.sidebar:
        st.header('Filtros')
        st.caption('Alcanzan a **Explicabilidad** y **Detalle técnico**. El **Resumen no se filtra**: sus títulos '
                   'enuncian conclusiones y un filtro las haría caducar (Clase 13, p.37).')
        with st.form('filtros', border=False):
            estado = st.multiselect('Estado', sorted(d['estado'].dropna().unique()), key=f'estado_{g}',
                                     help='¿Funciona igual en toda la región?')
            bioma = st.multiselect('Bioma', sorted(d['bioma'].dropna().unique()), key=f'bioma_{g}',
                                    help='¿El error depende del tipo de terreno?')
            franja = st.multiselect('Franja horaria', ORDEN_FRANJAS, key=f'franja_{g}',
                                     help='¿De noche se equivoca más?')
            lluvia = st.radio('Lluvia pronosticada',
                               ['todas las horas', 'con lluvia (PoP ≥ 50%)', 'sin lluvia (PoP < 50%)'],
                               key=f'lluvia_{g}',
                               help='El filtro de "días de lluvia" que pide la consigna, vía fcst_pop_pct')
            imputadas = st.checkbox('Excluir horas con observación imputada', key=f'imputadas_{g}',
                                     help='Honestidad metodológica: esas filas tienen el target interpolado')
            c1, c2 = st.columns([2, 1])
            aplicar = c1.form_submit_button('Aplicar', type='primary', width='stretch')
            limpiar = c2.form_submit_button('Limpiar', width='stretch')

        if limpiar:
            st.session_state['gen'] = g + 1
            st.rerun()
        if aplicar:
            st.toast('Filtros aplicados', icon='✅')

        st.divider()
        st.caption(f'Fuente: `data/processed/dashboard/` ({metricas["filas_test"]:,} filas). '
                   'Generado por `precomputo.py`; el dashboard no reentrena ni carga los modelos.')

    return {'estado': estado, 'bioma': bioma, 'franja': franja, 'lluvia': lluvia, 'imputadas': imputadas}


def aplicar_filtros(d: pd.DataFrame, sel: dict) -> pd.DataFrame:
    f = d
    if sel['estado']:
        f = f[f['estado'].isin(sel['estado'])]
    if sel['bioma']:
        f = f[f['bioma'].isin(sel['bioma'])]
    if sel['franja']:
        f = f[f['franja'].astype(str).isin(sel['franja'])]
    if sel['lluvia'].startswith('con'):
        f = f[f['fcst_pop_pct'] >= 50]
    elif sel['lluvia'].startswith('sin'):
        f = f[f['fcst_pop_pct'] < 50]
    if sel['imputadas']:
        f = f[~f['obs_imputado']]
    return f


def describir_filtros(sel: dict) -> list[str]:
    """Los filtros activos, en texto. Sin esto el lector no tiene forma de saber si su selección llegó o no."""
    activos = []
    if sel['estado']:
        activos.append('estados: ' + ', '.join(sel['estado']))
    if sel['bioma']:
        activos.append('biomas: ' + ', '.join(sel['bioma']))
    if sel['franja']:
        activos.append('franjas: ' + ', '.join(sel['franja']))
    if sel['lluvia'] != FILTROS_DEFECTO['lluvia']:
        activos.append(sel['lluvia'])
    if sel['imputadas']:
        activos.append('sin horas imputadas')
    return activos


def encabezado_filtrado(f: pd.DataFrame, d: pd.DataFrame, sel: dict) -> None:
    """Cuánto quedó seleccionado y con qué filtros: la confirmación visible de que la selección se aplicó."""
    activos = describir_filtros(sel)
    c1, c2 = st.columns([1, 3])
    c1.metric('Horas-estación seleccionadas', f'{len(f):,}',
              delta=None if len(f) == len(d) else f'{100 * len(f) / len(d):.1f}% del test',
              delta_color='off')
    with c2:
        if activos:
            st.success('**Filtros aplicados** · ' + ' · '.join(activos), icon='🔎')
        else:
            st.info('**Sin filtros**: todo el test. Elegí filtros a la izquierda y confirmá con **Aplicar**.',
                    icon='🔎')


# ---------------------------------------------------------------- los tres niveles

def nivel_resumen(d: pd.DataFrame, estaciones: pd.DataFrame, metricas: dict, estados: list | None) -> None:
    """Nivel 1: conduce el autor. Sin filtros y con títulos que enuncian la conclusión (Clase 13, p.37)."""
    rf = next(m for m in metricas['modelos'] if m['modelo'].startswith('min_error'))
    arbol = next(m for m in metricas['modelos'] if m['modelo'].startswith('interpretable'))
    cero = next(m for m in metricas['modelos'] if m['modelo'].startswith('cero'))

    st.info('Este nivel es el resumen fijo: **no responde a los filtros**, a propósito -sus títulos enuncian '
            'conclusiones sobre todo el test y un filtro las haría caducar (Clase 13, p.37). Para explorar con '
            'los filtros, pasá a **Explicabilidad** o **Detalle técnico**.', icon='📌')
    st.subheader(f'Corregir el pronóstico baja el error {rf["mejora_vs_cero_pct"]:.0f}%')
    st.caption(f'En términos absolutos: {rf["mejora_vs_cero_c"]:.2f} °C menos de error por hora '
               f'({cero["MAE"]:.2f} → {rf["MAE"]:.2f} °C de MAE). El encuadre relativo sirve para comparar '
               'contra la referencia; el absoluto, para juzgar si la mejora alcanza.')

    k1, k2, k3, k4 = st.columns(4)
    k1.metric('MAE del pronóstico corregido', f'{rf["MAE"]:.2f} °C',
              delta=f'{-rf["mejora_vs_cero_c"]:.2f} °C vs. no corregir', delta_color='inverse')
    k2.metric(f'Horas con error ≤ {metricas["umbral_kpi_c"]:.0f} °C', f'{rf["pct_error_menor_1.0c"]:.0f}%',
              delta=f'{rf["pct_error_menor_1.0c"] - cero["pct_error_menor_1.0c"]:+.0f} pp')
    k3.metric('MASE (vs. persistencia)', f'{rf["MASE"]:.3f}',
              help='Menor que 1 = mejora a la persistencia del error')
    if metricas['cobertura_observada'] is not None:
        k4.metric(f'Cobertura del intervalo {metricas["nivel_intervalo"]:.0%}',
                  f'{metricas["cobertura_observada"]:.0%}',
                  delta=f'ancho medio {metricas["ancho_medio_banda_c"]:.1f} °C', delta_color='off',
                  help='Calibrado en los primeros días del test y medido en los restantes')

    st.divider()
    st.markdown('##### Los dos modelos le ganan a no corregir, y el de mínimo error por más')
    resumen = pd.DataFrame(metricas['modelos'])
    resumen['modelo'] = resumen['modelo'].map(
        lambda m: SIN_CORREGIR if m.startswith('cero') else
        (INTERPRETABLE if m.startswith('interpretable') else MIN_ERROR))
    resumen['etiqueta'] = resumen.apply(
        lambda r: f'{r["MAE"]:.3f} °C' + ('' if r['mejora_vs_cero_pct'] == 0
                                          else f'  ({r["mejora_vs_cero_pct"]:+.1f}%)'), axis=1)
    con_tabla(barras_mae(resumen), resumen[['modelo', 'MAE', 'RMSE', 'sesgo_medio', 'p90_abs',
                                             'mejora_vs_cero_pct', 'MASE']], 'mae')

    st.divider()
    st.markdown('##### Vista de monitoreo: pronóstico contra pronóstico corregido')
    c1, c2 = st.columns([3, 1])
    orden_est = estaciones.sort_values('horas_test', ascending=False)['station_id'].tolist()
    estacion = c1.selectbox('Estación', orden_est,
                             help='Selector de entidad, no un filtro: elige qué estación se dibuja. '
                                  'Promediar las 432 series borraría la dinámica que este panel muestra.')
    banda = c2.checkbox('Mostrar intervalo 80%', value=True)
    info = estaciones[estaciones['station_id'] == estacion].iloc[0]
    st.caption(f'{info["estado"]} · {info["bioma"]} · {info["elevation"]:.0f} m · '
               f'{info["horas_test"]} horas en el test')
    st.altair_chart(panel_monitoreo(d, estacion, banda), width='stretch')

    st.divider()
    st.markdown('##### ¿Mejora en todas partes?')
    por_est = (error_largo(d, ['station_id'])
               .pivot(index='station_id', columns='modelo', values='MAE').reset_index())
    por_est.columns = [str(c) for c in por_est.columns]
    por_est = por_est.rename(columns={SIN_CORREGIR: 'MAE_sin', MIN_ERROR: 'MAE_con'})
    por_est['mejora_pct'] = 100 * (1 - por_est['MAE_con'] / por_est['MAE_sin'])
    por_est = por_est.merge(estaciones, on='station_id').rename(columns={'horas_test': 'n'})
    empeoran = int((por_est['mejora_pct'] < 0).sum())
    st.caption(f'Divergente centrado en cero: azul mejora, rojo empeora. '
               f'{len(por_est) - empeoran} de {len(por_est)} estaciones mejoran; {empeoran} empeoran.')
    con_tabla(mapa_mejora(por_est, estados),
              por_est[['station_id', 'estado', 'bioma', 'MAE_sin', 'MAE_con', 'mejora_pct', 'n']]
              .sort_values('mejora_pct').round(3), 'mapa')


def nivel_explicabilidad(f: pd.DataFrame, d: pd.DataFrame, sel: dict) -> None:
    """Nivel 2: se abre al lector. Responde a los filtros y los títulos formulan la pregunta."""
    encabezado_filtrado(f, d, sel)
    st.caption(f'{f["station_id"].nunique()} estaciones en la selección.')

    st.markdown('##### ¿Qué variables usa el modelo para corregir?')
    st.warning('Calculado sobre **todo** el test en la notebook 8: este panel y el siguiente no responden a los '
               'filtros.', icon='📌')
    perm = cargar_csv(INTERPRETABILIDAD / 'tabla_permutation_importance.csv')
    if perm is None:
        st.info('Falta `data/out/interpretabilidad/tabla_permutation_importance.csv` (notebook 8).')
    else:
        st.caption('Permutation importance, agrupando antes del ranking las variables redundantes: '
                   '`hora_sin`/`hora_cos` son una sola variable y los tres lags del error son la misma '
                   'señal. Es *model-agnostic*, así que los dos modelos son comparables en la misma escala. '
                   'Dos advertencias: sumar la importancia dentro de un grupo aproxima la permutación '
                   'conjunta, y esto se calculó sobre **todo** el test, así que no responde a los filtros.')
        g, t = barras_importancia_agrupada(perm)
        con_tabla(g, t, 'perm')

    st.divider()
    st.markdown('##### ¿En qué sentido influye cada variable?')
    shap_df = cargar_shap()
    if shap_df is None:
        st.info('Falta `shap_muestra.parquet`. Generarlo con `python TP3/dashboard/precomputo.py`.')
    else:
        st.caption(f'Beeswarm de SHAP del modelo de mínimo error sobre una muestra de {len(shap_df)} horas. '
                   'Cada punto es una hora: la posición horizontal es cuánto empujó esa variable al error '
                   'pronosticado, y el color, si la variable valía poco (claro) o mucho (oscuro). '
                   'Las barras de |SHAP| medio dan el ranking; esto agrega el **sentido** del efecto. '
                   'Importancia no es causalidad: SHAP describe al modelo, no a la atmósfera.')
        g, t = beeswarm_shap(shap_df)
        con_tabla(g, t, 'shap')

    st.divider()
    st.markdown('##### ¿A qué hora corrige menos el modelo?')
    por_hora = error_largo(f, ['hora_local'])
    st.caption('En orden horario, no ordenado por valor: el ranking se leería más rápido pero se perdería la '
               'forma del día, que es justo donde está el hallazgo.')
    con_tabla(lineas_por_hora(por_hora), por_hora.round(3), 'hora')

    st.divider()
    st.markdown('##### ¿Funciona igual en toda la región?')
    por_bioma = error_largo(f, ['bioma'])
    st.caption('El n de cada bioma va en la etiqueta: sin controlar la composición, un grupo con pocas '
               'estaciones puede parecer peor (o mejor) de lo que es.')
    con_tabla(barras_por_bioma(por_bioma), por_bioma.round(3), 'bioma')

    st.divider()
    st.markdown('##### ¿Por qué falló en una hora puntual?')
    caso = cargar_csv(INTERPRETABILIDAD / 'tabla_shap_local_caso_mediodia.csv')
    if caso is None:
        st.info('Falta `tabla_shap_local_caso_mediodia.csv` (notebook 8).')
    else:
        cg, ct = cascada_shap_local(caso)
        cc1, cc2 = st.columns([2, 1])
        with cc1:
            st.altair_chart(cg, width='stretch')
        with cc2:
            st.markdown('**Para quien audita el modelo**: el gráfico, que es exacto y suma hasta la '
                        'predicción.\n\n**Para quien decide**: una frase contrafactual, que no exige saber '
                        'qué es "el promedio del modelo".')
            st.info('Caso de la ventana del mediodía (notebook 8): el modelo tenía señal en la historia '
                    'reciente del error, pero apuntaba en la dirección equivocada. Es la limitación de fondo '
                    'del feature set actual: las anomalías puntuales no vienen avisadas.')
        with st.expander('Ver tabla', expanded=False):
            st.dataframe(ct, width='stretch', hide_index=True)


def nivel_detalle(f: pd.DataFrame, d: pd.DataFrame, metricas: dict, sel: dict) -> None:
    """Nivel 3: el diagnóstico técnico, para quien mantiene el modelo (Clase 13, p.30)."""
    encabezado_filtrado(f, d, sel)
    st.caption('Los gráficos técnicos del diagnóstico, para quien mantiene el modelo. '
               'Responden a los mismos filtros que Explicabilidad.')

    st.markdown('##### Distribución del error, no sólo su promedio')
    ancho = st.radio('Ancho de clase', [0.25, 0.5, 1.0], index=1, horizontal=True,
                      format_func=lambda v: f'{v} °C',
                      help='Se declara explícitamente: la ventana puede crear un patrón o suprimirlo')
    st.caption(f'Clases de {ancho} °C. El MAE de {metricas["modelos"][-1]["MAE"]:.2f} °C esconde las colas: '
               f'el P90 del error absoluto está en {metricas["modelos"][-1]["p90_abs"]:.2f} °C.')
    hg, ht = histograma_error(f, ancho)
    con_tabla(hg, ht, 'histograma')

    st.divider()
    st.markdown('##### La cola del error por hora (P90)')
    con_tabla(lineas_p90(error_largo(f, ['hora_local'])),
              error_largo(f, ['hora_local']).round(3), 'p90')

    st.divider()
    st.markdown('##### Las 7 familias de modelos de la Etapa A')
    comparacion = cargar_csv(MODELADO_GLOBAL / 'tabla_comparacion_modelos_temp_c.csv')
    if comparacion is not None:
        st.dataframe(comparacion, width='stretch', hide_index=True)
        st.caption('`cv_MAE_std` es la dispersión entre los 5 folds de `TimeSeriesSplit` para los '
                   'hiperparámetros ganadores: la señal barata de si el MAE de test es representativo.')

    st.divider()
    st.markdown('##### Cobertura de datos y validez del target')
    cv1, cv2 = st.columns(2)
    validez = cargar_csv(MODELADO / 'tabla_validez_target.csv')
    split = cargar_csv(MODELADO / 'tabla_resumen_split.csv')
    if validez is not None:
        cv1.dataframe(validez, width='stretch', hide_index=True)
    if split is not None:
        cv2.dataframe(split, width='stretch', hide_index=True)
    st.caption(f'Al dashboard entran {metricas["filas_test"]:,} filas de test: las que además de tener target '
               'válido no tienen ningún faltante en las features, el mismo criterio con el que se entrenó y '
               'se evaluó.')

    st.divider()
    st.markdown('##### Cómo se construyó el intervalo del 80%')
    st.markdown(f"""
Split-conformal: los cuantiles 10 y 90 del residuo se calibran por (bioma, franja) sobre los primeros
**{metricas['dias_calibracion']} días** del test y se aplican a los días restantes, donde se mide la cobertura.
Resultado: **{metricas['cobertura_observada']:.1%} observado contra {metricas['nivel_intervalo']:.0%} nominal**,
con un ancho medio de **{metricas['ancho_medio_banda_c']:.2f} °C**.

Calibrar con los residuos de *train* -el intento anterior- daba una banda de ±1,0 °C y una cobertura del 55,4%:
el Random Forest está sobreajustado en train (MAE 0,661 contra 1,141 en test), así que sus residuos de ahí son
demasiado angostos. Una banda del 80% que cubre la mitad de los casos induce exactamente el exceso de confianza
que un intervalo debería evitar.
""")
    cuantiles = cargar_csv(DATOS / 'intervalos_empiricos.csv')
    if cuantiles is not None:
        with st.expander('Ver los cuantiles por bioma y franja'):
            st.dataframe(cuantiles.round(3), width='stretch', hide_index=True)



# ---------------------------------------------------------------- app

def main() -> None:
    st.set_page_config(page_title='TP3 · Corrección del pronóstico', page_icon='🌡️', layout='wide')
    registrar_tema()
    st.markdown("""<style>
        [data-testid="stMetricValue"] { font-size: 2.1rem; font-weight: 600; }
        [data-testid="stMetricLabel"] { color: #52514e; }
        </style>""", unsafe_allow_html=True)

    if not (DATOS / 'predicciones_test.parquet').exists():
        st.error(f'Falta `{DATOS}`. Correr primero:\n\n`python TP3/dashboard/precomputo.py`')
        st.stop()

    d = cargar_predicciones()
    estaciones = cargar_estaciones()
    metricas = cargar_metricas()
    estados = cargar_estados()

    st.title('Corrección del error de pronóstico de temperatura')
    st.caption(f"{metricas['filas_test']:,} horas-estación · {metricas['estaciones_test']} estaciones · "
               f"test del {metricas['desde'][:10]} al {metricas['hasta'][:10]} · "
               'error = pronosticado − observado')

    # --- Filtros: un solo lugar, delimitando todo lo que alcanzan (Explicabilidad y Detalle). Un control por
    # cada pregunta real de la audiencia; lo demás corresponde a la notebook (Clase 13, p.40). Van dentro de un
    # `st.form`: así la selección no se aplica sola a medio armar, sino al confirmar con "Aplicar".
    seleccion = controles_filtros(d, metricas)
    f = aplicar_filtros(d, seleccion)

    resumen_tab, explica_tab, detalle_tab = st.tabs(['Resumen', 'Explicabilidad', 'Detalle técnico'])

    with resumen_tab:
        nivel_resumen(d, estaciones, metricas, estados)

    # Los dos niveles filtrables comparten la misma guarda: una selección vacía avisa, y no se pueden usar
    # st.stop() acá porque abortaría el script entero y dejaría los otros tabs en blanco.
    for tab, nivel in ((explica_tab, lambda: nivel_explicabilidad(f, d, seleccion)),
                       (detalle_tab, lambda: nivel_detalle(f, d, metricas, seleccion))):
        with tab:
            if f.empty:
                encabezado_filtrado(f, d, seleccion)
                st.warning('Los filtros no dejan ninguna fila. Ampliá la selección y volvé a tocar **Aplicar**.')
            else:
                nivel()


if __name__ == '__main__':
    main()
