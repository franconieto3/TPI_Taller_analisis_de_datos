"""Paleta, estilo de matplotlib, funciones de gráficos y carga del dataset limpio compartidas por los notebooks.

Uso (desde un notebook en notebooks/):
    from utils import AZUL, NARANJA, GRIS_TEXTO, aplicar_estilo, hexbin_par, cargar_mediciones_limpias
    aplicar_estilo()
    mp = cargar_mediciones_limpias()
"""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

# Misma paleta que en analisis_estaciones.ipynb
AZUL, NARANJA = '#2a78d6', '#eb6834'
GRIS_TEXTO, GRIS_GRILLA = '#52514e', '#e4e3df'

# Secuencial (densidad de puntos) y divergente (correlaciones, centrada en 0)
SECUENCIAL = LinearSegmentedColormap.from_list(
    'azules', ['#cde2fb', '#86b6ef', '#2a78d6', '#184f95', '#0d366b'])
DIVERGENTE = LinearSegmentedColormap.from_list(
    'azul_rojo', ['#184f95', '#6da7ec', '#f0efec', '#ee9190', '#b8302f'])


def aplicar_estilo():
    """rcParams comunes: sin bordes superior y derecho, grilla suave y textos en gris."""
    plt.rcParams.update({
        'figure.dpi': 110,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': GRIS_TEXTO, 'axes.labelcolor': GRIS_TEXTO,
        'xtick.color': GRIS_TEXTO, 'ytick.color': GRIS_TEXTO,
        'axes.grid': True, 'grid.color': GRIS_GRILLA, 'grid.linewidth': 0.6,
        'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.labelsize': 10,
    })


def hexbin_par(ax, datos, x, y, identidad=False, gridsize=60, texto_abajo=False):
    """Hexbin de densidad para un par de variables, con r de Pearson y rho de Spearman."""
    d = datos[[x, y]].dropna()
    hb = ax.hexbin(d[x], d[y], gridsize=gridsize, bins='log', mincnt=1,
                   cmap=SECUENCIAL, linewidths=0)
    if identidad:
        lims = [d[x].min(), d[x].max()]
        ax.plot(lims, lims, color=NARANJA, lw=1.5, ls='--', label='y = x')
        ax.legend(loc='lower right', frameon=False)
    r = d[x].corr(d[y])
    rho = d[x].corr(d[y], method='spearman')
    ax.text(0.03, 0.03 if texto_abajo else 0.97, f'r = {r:.2f}   ρ = {rho:.2f}\nn = {len(d):,}',
            transform=ax.transAxes, va='bottom' if texto_abajo else 'top', fontsize=9, color=GRIS_TEXTO,
            bbox=dict(fc='white', ec='none', alpha=0.85))
    ax.grid(False)
    return hb


def escala_comun(hexbins):
    """Misma escala de color en todos los paneles, para que la barra de color valga para todos."""
    vmax = max(hb.get_array().max() for hb in hexbins)
    for hb in hexbins:
        hb.set_norm(LogNorm(vmin=1, vmax=vmax))
    return hexbins[-1]


def banda_por_hora(ax, datos, variable, color, etiqueta):
    """Media y rango intercuartílico de una variable por hora local."""
    g = datos.groupby('hora_local')[variable]
    q = g.quantile([0.25, 0.75]).unstack()
    ax.fill_between(q.index, q[0.25], q[0.75], color=color, alpha=0.15, linewidth=0)
    ax.plot(q.index, g.mean(), color=color, lw=2, label=etiqueta)


def cargar_mediciones_limpias(ruta='../data/processed/mediciones_limpias.csv'):
    """Dataset limpio que exporta relaciones-bivariadas.ipynb (secc. 1), con los timestamps en UTC."""
    mp = pd.read_csv(ruta)
    for c in ['obs_timestamp', 'fcst_target_timestamp', 'fcst_generated_at']:
        mp[c] = pd.to_datetime(mp[c], utc=True)
    return mp
