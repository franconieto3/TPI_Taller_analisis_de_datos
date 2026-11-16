"""Paleta y helpers de gráficos del dashboard.

Los colores son los mismos que `notebooks/utils.py` usa en todas las figuras del TP, así que el dashboard y el
informe se ven como una sola pieza. Validados con el validador de la guía de visualización sobre superficie clara
(`#fcfcfb`), con los 3 slots que usa el dashboard y bajo `--pairs all`:

    lightness band PASS · chroma floor PASS · CVD separation PASS (peor par ΔE 9,2 deutan)
    normal-vision floor PASS (ΔE 24,0) · contrast WARN: #1baf7a queda en 2,74:1

El WARN de contraste del verde obliga a un canal de alivio: por eso **todos** los gráficos llevan etiquetas
visibles o su tabla equivalente en un `expander` ("Ver tabla"), no sólo el tooltip.
"""
import altair as alt

# Slots categóricos 1-3 (mismo orden que la guía). Cada entidad tiene su color fijo: filtrar no repinta a los que
# quedan -el lector que aprendió "el azul es Random Forest" no se confunde al cambiar de filtro.
AZUL = '#2a78d6'       # slot 1: modelo de mínimo error (Random Forest)
NARANJA = '#eb6834'    # slot 2: sin corregir (la referencia que hay que ganar)
VERDE = '#1baf7a'      # slot 3: modelo interpretable (árbol de decisión)

GRIS_TEXTO, GRIS_GRILLA = '#52514e', '#e4e3df'
SUPERFICIE = '#fcfcfb'
BORDE = '#dedcd7'         # hairline de los bordes de estado en el mapa
GRIS_NEUTRO = '#b4b2ad'   # para marcas de contexto que no son una serie

SIN_CORREGIR = 'sin corregir'
INTERPRETABLE = 'interpretable (árbol)'
MIN_ERROR = 'mínimo error (Random Forest)'

ORDEN_MODELOS = [SIN_CORREGIR, INTERPRETABLE, MIN_ERROR]
COLOR_MODELOS = {SIN_CORREGIR: NARANJA, INTERPRETABLE: VERDE, MIN_ERROR: AZUL}

# Secuencial de un solo tono (magnitud) y divergente con gris en el centro (polaridad), iguales a utils.py.
SECUENCIAL = ['#cde2fb', '#86b6ef', '#2a78d6', '#184f95', '#0d366b']
DIVERGENTE = ['#184f95', '#6da7ec', '#f0efec', '#ee9190', '#b8302f']

ORDEN_FRANJAS = ['madrugada (0-5h)', 'mañana (6-11h)', 'tarde (12-17h)', 'noche (18-23h)']


def escala_modelos(dominio=None) -> alt.Scale:
    """Escala categórica con color fijo por entidad, no por orden de aparición."""
    dominio = dominio or ORDEN_MODELOS
    return alt.Scale(domain=dominio, range=[COLOR_MODELOS[m] for m in dominio])


def tema_tp3():
    """Marcas finas, grilla y ejes en hairline recesivo, sin bordes superior/derecho -igual que utils.py."""
    return {
        'config': {
            'background': SUPERFICIE,
            'view': {'strokeWidth': 0, 'continuousWidth': 700, 'continuousHeight': 300},
            'font': 'Source Sans Pro, Segoe UI, sans-serif',
            'title': {'color': '#2e2d2b', 'fontSize': 13, 'fontWeight': 600, 'anchor': 'start',
                      'subtitleColor': GRIS_TEXTO, 'subtitleFontSize': 11},
            'axis': {'domainColor': GRIS_GRILLA, 'tickColor': GRIS_GRILLA, 'gridColor': GRIS_GRILLA,
                     'gridWidth': 0.6, 'gridDash': [], 'labelColor': GRIS_TEXTO, 'titleColor': GRIS_TEXTO,
                     'labelFontSize': 10, 'titleFontSize': 10, 'titleFontWeight': 'normal'},
            'legend': {'labelColor': GRIS_TEXTO, 'titleColor': GRIS_TEXTO, 'labelFontSize': 10,
                       'titleFontSize': 10, 'orient': 'top', 'direction': 'horizontal', 'title': None},
            'line': {'strokeWidth': 2},
            'point': {'size': 60},
            'bar': {'cornerRadiusEnd': 4},
            'range': {'category': [AZUL, NARANJA, VERDE], 'heatmap': SECUENCIAL, 'diverging': DIVERGENTE},
        }
    }


def registrar_tema() -> None:
    alt.theme.register('tp3', enable=True)(tema_tp3)
