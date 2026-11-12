"""Shared plotting style for the paper figures: every curve is distinguishable in
black and white (line style + sparse markers + luminance-separated colours),
reference slopes are thin dotted black lines with a text label."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# luminance-separated, colourblind-safe (no red/green pair, no light yellow)
COLORS = ['#000000', '#1f77b4', '#ff7f0e', '#7f7f7f', '#9467bd', '#17becf', '#8c564b', '#e377c2']
LINESTYLES = ['-', '--', '-.', ':']
MARKERS = ['o', 's', '^', 'D', 'x']


def setup(font=11):
    plt.rcParams.update({
        'font.size': font, 'axes.titlesize': font, 'axes.labelsize': font,
        'legend.fontsize': font - 2.5, 'xtick.labelsize': font - 1.5, 'ytick.labelsize': font - 1.5,
        'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.5,
        'lines.linewidth': 1.5, 'legend.framealpha': 0.9, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
        'axes.prop_cycle': matplotlib.cycler(color=COLORS),
    })


def style(i, n_points=None, frac=0.12, ms=4.5, lw=1.5, color=None, ls=None, marker=None):
    """kwargs for the i-th curve of a panel: colour, line style, marker, sparse markevery
    (offset shifted with i so markers of different curves do not sit on top of each other)."""
    kw = dict(color=COLORS[i % len(COLORS)] if color is None else color,
              ls=LINESTYLES[i % len(LINESTYLES)] if ls is None else ls,
              marker=MARKERS[i % len(MARKERS)] if marker is None else marker,
              ms=ms, lw=lw, markerfacecolor='none' if (marker or MARKERS[i % len(MARKERS)]) != 'x' else None,
              markeredgewidth=1.1)
    if n_points is not None and n_points > 1:
        step = max(1, int(round(n_points * frac)))
        kw['markevery'] = ((i * step) // 3 % step, step)
    return kw


def ref_slope(ax, K, c, p, label, x_text=None, dy=1.35, ha='left', fontsize=None):
    """thin dotted black reference line c*K^p with a text label next to it."""
    K = np.asarray(K, dtype=float)
    ax.plot(K, c * K ** p, ls=':', color='k', lw=0.9, zorder=1)
    if x_text is None:
        x_text = K[len(K) // 2]
    ax.text(x_text, c * x_text ** p * dy, label, ha=ha, va='bottom', fontsize=fontsize or plt.rcParams['legend.fontsize'])


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
