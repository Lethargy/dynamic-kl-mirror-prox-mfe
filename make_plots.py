"""Figures for the dynamic primal-dual paper (restyled for black-and-white printing:
line styles + sparse markers + luminance-separated colours; reference slopes labelled in the plot).
Reads the saved histories/results only; no simulation is run."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import bw_style as bw

src = Path(__file__).parent
plots = src.parent / 'plots'
plots.mkdir(exist_ok=True)
bw.setup(font=14)
# Keep the smallest text above 7 pt after reduction to the paper's text width.
plt.rcParams.update({'legend.fontsize': 13, 'xtick.labelsize': 13, 'ytick.labelsize': 13})
W3, H3 = 10.5, 4.5        # three panels, full text width
W2, H2 = 7.8, 4.2         # two panels, 0.8 text width


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

# ------------------------------------------------ finite state: rates
h = np.genfromtxt(src / 'finite_state_history.csv', delimiter=',', names=True)
K = h['K']
n = len(K)
fig, ax = plt.subplots(1, 3, figsize=(W3, H3))
ax[0].loglog(K, h['residual_avg'], label='averaged iterate', **bw.style(0, n))
ax[0].loglog(K, h['residual_last'], label='last iterate', **bw.style(1, n))
bw.ref_slope(ax[0], K, 4, -1, r'$K^{-1}$', x_text=2.5, dy=1.05)
ax[0].set_xlabel('K'); ax[0].set_title('quadrature residual'); ax[0].legend(loc='upper right')
ax[1].loglog(K, h['projection_distance'], label=r'$\Vert \hat x_K-\bar x_K\Vert _{\mathcal{X}}$', **bw.style(0, n))
bw.ref_slope(ax[1], K, 8, -1, r'$K^{-1}$', x_text=2.5, dy=1.05)
ax[1].set_xlabel('K'); ax[1].set_title('distance to projection'); ax[1].legend(loc='lower left')
ax[2].loglog(K, h['exploitability'], label=r'$\operatorname{Gap}_{\mathrm{S}}(\hat x_K)$', **bw.style(0, n))
bw.ref_slope(ax[2], K, 3.3, -1, r'$K^{-1}$', x_text=2.5, dy=1.05)
bw.ref_slope(ax[2], K, 0.3, -0.5, r'$K^{-1/2}$', x_text=4e3, dy=1.4)
ax[2].set_xlabel('K'); ax[2].set_title('Stampacchia gap'); ax[2].legend(loc='lower left')
save_with_legends(fig, ax, plots / 'finite_state_rates.png')

# ------------------------------------------------ finite state: equilibrium
r = np.load(src / 'finite_state_result.npz')
tm, q_hat, gam, psi, dpsi = r['tm'], r['q_hat'], r['gam'], r['psi'], r['dpsi']
mu = q_hat.sum(2)
nt = len(tm)
fig, ax = plt.subplots(1, 3, figsize=(W3, 4.7))
for y in range(mu.shape[1]):
    ax[0].plot(tm, mu[:, y], label=f'state {y+1}', **bw.style(y, nt, frac=0.1))
ax[0].set_xlabel('t'); ax[0].set_title(r'projected flow $\hat\mu_t(y)$')
names = ['left', 'stay', 'right']
for a in range(3):          # action: colour + marker; state: solid (state 1) vs dashed (state 3)
    ax[1].plot(tm, gam[:, 0, a], label=rf'{names[a]}, $y=1$', **bw.style(a, nt, frac=0.1, ls='-'))
    ax[1].plot(tm, gam[:, 2, a], label=rf'{names[a]}, $y=3$', **bw.style(a, nt, frac=0.1, ls='--'))
ax[1].set_xlabel('t'); ax[1].set_title(r'relaxed feedback $\hat\gamma_t(a\mid y)$')
for y in range(psi.shape[1]):
    ax[2].plot(tm, psi[:, y], label=f'state {y+1}', **bw.style(y, nt, frac=0.1))
ax[2].set_xlabel('t'); ax[2].set_title(r'dual iterate $\psi_K(t,y)$')
for axis in ax:
    handles, labels = axis.get_legend_handles_labels()
    if axis is ax[1]:
        order = [0, 2, 4, 1, 3, 5]  # one feedback-state column per y
        handles, labels = [handles[i] for i in order], [labels[i] for i in order]
    axis.legend(handles, labels, ncol=2, fontsize=13, loc='upper center', bbox_to_anchor=(0.5, -0.25),
                frameon=False, handlelength=1.1, columnspacing=0.6, handletextpad=0.35)
save_with_legends(fig, ax, plots / 'finite_state_equilibrium.png')

# ------------------------------------------------ diffusion
for tag, title in [('diffusion_g', r'$g(y)=-\tanh y$'), ('diffusion_0', r'$g\equiv0$')]:
    f = src / f'{tag}_result.npz'
    if not f.exists():
        continue
    d = np.load(f)
    hist = d['history']
    Kh = hist[:, 0]
    nh = len(Kh)
    fig, ax = plt.subplots(1, 3, figsize=(W3, H3))
    ax[0].loglog(Kh, hist[:, 1], label='averaged iterate', **bw.style(0, nh))
    ax[0].loglog(Kh, hist[:, 2], label='last iterate', **bw.style(1, nh))
    Kr = Kh[Kh >= 300]                     # reference slope only over the decaying part
    c1 = np.interp(1e3, Kh, hist[:, 1]) * 1e3 * 0.5
    bw.ref_slope(ax[0], Kr, c1, -1, r'$K^{-1}$', x_text=3e3, dy=0.35)
    ax[0].set_xlabel('K'); ax[0].set_title('restricted residual'); ax[0].legend(loc='lower left')
    ax[1].loglog(Kh, hist[:, 3], label=r'$\max_t|m_{\bar x_K}(t)|$', **bw.style(0, nh))
    ax[1].set_xlabel('K'); ax[1].set_title(r'$\max_t|m_{\bar x_K}(t)|$')
    ax[2].plot(d['tm'], d['m_pd'], label='PD-KL-MP average', **bw.style(0, len(d['tm']), frac=0.1))
    ax[2].plot(d['tfd'], d['m_fp'], label='FD fictitious play', **bw.style(1, len(d['tfd']), frac=0.08))
    ax[2].plot(d['tfd'], d['m_hat'], label='projected average', **bw.style(2, len(d['tfd']), frac=0.08))
    ax[2].set_xlabel('t'); ax[2].set_title(r'equilibrium moment $m(t)$' + '\n' + title); ax[2].legend()
    save_with_legends(fig, ax, plots / f'{tag}.png')

    # dual function and terminal law
    x, c, cT, yg = d['x'], d['c'], d['cT'], d['yg']
    fig, ax = plt.subplots(1, 2, figsize=(W2, H2))
    from numpy.polynomial.hermite_e import HermiteE
    from math import factorial
    Mb = c.shape[1]
    B = np.array([HermiteE.basis(m)(yg) / np.sqrt(factorial(m)) * np.exp(-yg ** 2 / 2) for m in range(Mb)]).T
    ny = len(yg)
    for j, i in enumerate([0, len(d['tm']) // 2, len(d['tm']) - 1]):
        ax[0].plot(yg, B @ c[i], label=f't={d["tm"][i]:.2f}', **bw.style(j, ny, frac=0.07))
    ax[0].plot(yg, B @ cT, label='t=1', **bw.style(3, ny, frac=0.07))
    ax[0].set_xlim(-4, 4); ax[0].set_xlabel('y'); ax[0].set_title(r'dual iterate $\psi_K(t,\cdot)$'); ax[0].legend()
    ax[1].plot(yg, d['dens_hat'][-1], label='projected average, t=1', **bw.style(0, ny, frac=0.07))
    ax[1].plot(yg, d['dens_fp'][-1], label='fictitious play, t=1', **bw.style(1, ny, frac=0.07))
    ax[1].set_xlim(-5, 5); ax[1].set_xlabel('y'); ax[1].set_title('terminal law'); ax[1].legend()
    save_with_legends(fig, ax, plots / f'{tag}_dual.png', bottom=0.30)
print('plots written to', plots)
