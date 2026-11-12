"""Figures for Section 7.1 of the paper (finite-state congestion game).

Reads finite_state_history.csv and finite_state_result.npz, written by
finite_state_rates.py, and writes plots/finite_state_rates.png and
plots/finite_state_equilibrium.png. No simulation is run. The figures are
styled for black-and-white printing: line styles, sparse markers and
luminance-separated colours, with the reference slopes labelled in the plot.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import bw_style as bw

src = Path(__file__).parent
plots = src / 'plots'
plots.mkdir(exist_ok=True)
bw.setup(font=14)
# Keep the smallest text above 7 pt after reduction to the paper's text width.
plt.rcParams.update({'legend.fontsize': 13, 'xtick.labelsize': 13, 'ytick.labelsize': 13})
W3, H3 = 10.5, 4.5        # three panels, full text width


def save_with_legends(fig, axes, path, bottom=0.26):
    """Reserve a readable legend row without covering curves or tick labels."""
    legends = [axis.get_legend() for axis in axes if axis.get_legend() is not None]
    for legend in legends:
        legend.set_loc('upper center')
        legend.set_bbox_to_anchor((0.5, -0.25))
        legend.set_frame_on(False)
        legend.set_in_layout(False)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    # Include external legends in savefig's tight bounding box after layout.
    for legend in legends:
        legend.set_in_layout(True)
    bw.save(fig, path)

# ------------------------------------------------ rates
h = np.genfromtxt(src / 'finite_state_history.csv', delimiter=',', names=True)
K = h['K']
n = len(K)
fig, ax = plt.subplots(1, 3, figsize=(W3, H3))
ax[0].loglog(K, h['residual_avg'], label='averaged iterate', **bw.style(0, n))
ax[0].loglog(K, h['residual_last'], label='last iterate', **bw.style(1, n))
bw.ref_slope(ax[0], K, 4, -1, r'$K^{-1}$', x_text=2.5, dy=1.05)
ax[0].set_xlabel('K'); ax[0].set_title('forward-equation residual'); ax[0].legend(loc='upper right')
ax[1].loglog(K, h['projection_distance'], label=r'$\Vert\hat\pi_K-\bar\pi_K\Vert_{\mathrm{TV},2}$', **bw.style(0, n))
bw.ref_slope(ax[1], K, 8, -1, r'$K^{-1}$', x_text=2.5, dy=1.05)
ax[1].set_xlabel('K'); ax[1].set_title('size of the correction'); ax[1].legend(loc='lower left')
ax[2].loglog(K, h['exploitability'], label=r'$\mathrm{Gap}_{\mathrm{S}}(\hat\pi_K)$', **bw.style(0, n))
bw.ref_slope(ax[2], K, 3.3, -1, r'$K^{-1}$', x_text=2.5, dy=1.05)
bw.ref_slope(ax[2], K, 0.3, -0.5, r'$K^{-1/2}$', x_text=4e3, dy=1.4)
ax[2].set_xlabel('K'); ax[2].set_title('Stampacchia gap'); ax[2].legend(loc='lower left')
save_with_legends(fig, ax, plots / 'finite_state_rates.png')

# ------------------------------------------------ equilibrium
# Keys of the saved result, in the paper's notation: q_hat[t, y, a] is the
# corrected average hat-pi_K, gam[t, y, a] the relaxed feedback alpha_t(a | y),
# and psi[t, y] the dual iterate u_K(t, y).
r = np.load(src / 'finite_state_result.npz')
tm, q_hat, gam, psi = r['tm'], r['q_hat'], r['gam'], r['psi']
mu = q_hat.sum(2)
nt = len(tm)
fig, ax = plt.subplots(1, 3, figsize=(W3, 4.7))
for y in range(mu.shape[1]):
    ax[0].plot(tm, mu[:, y], label=f'state {y+1}', **bw.style(y, nt, frac=0.1))
ax[0].set_xlabel('t'); ax[0].set_title(r'corrected flow $(\mathcal{M}\hat\pi_K)_t(y)$')
names = ['left', 'stay', 'right']
for a in range(3):          # action: colour + marker; state: solid (state 1) vs dashed (state 3)
    ax[1].plot(tm, gam[:, 0, a], label=rf'{names[a]}, $y=1$', **bw.style(a, nt, frac=0.1, ls='-'))
    ax[1].plot(tm, gam[:, 2, a], label=rf'{names[a]}, $y=3$', **bw.style(a, nt, frac=0.1, ls='--'))
ax[1].set_xlabel('t'); ax[1].set_title(r'relaxed feedback $\alpha_t(a\mid y)$')
for y in range(psi.shape[1]):
    ax[2].plot(tm, psi[:, y], label=f'state {y+1}', **bw.style(y, nt, frac=0.1))
ax[2].set_xlabel('t'); ax[2].set_title(r'dual iterate $u_K(t,y)$')
for axis in ax:
    handles, labels = axis.get_legend_handles_labels()
    if axis is ax[1]:
        order = [0, 2, 4, 1, 3, 5]  # one feedback-state column per y
        handles, labels = [handles[i] for i in order], [labels[i] for i in order]
    axis.legend(handles, labels, ncol=2, fontsize=13, loc='upper center', bbox_to_anchor=(0.5, -0.25),
                frameon=False, handlelength=1.1, columnspacing=0.6, handletextpad=0.35)
save_with_legends(fig, ax, plots / 'finite_state_equilibrium.png')
print('plots written to', plots)
