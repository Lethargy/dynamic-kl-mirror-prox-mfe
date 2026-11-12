"""Primal-dual KL Mirror-Prox for a continuous-time finite-state MFG.

Checks the rates of the paper: O(1/K) feasibility residual of the averaged
primal iterate, O(1/K) distance to its feasibility projection, and the
exploitability of the projected average.  H = R^E with the Euclidean inner
product, so the Riesz map is the identity and the dual step is the exact
time integration of the Kolmogorov defect (no Galerkin in time).  Time
integrals are evaluated by the midpoint rule on Nt intervals.
"""
from pathlib import Path
import numpy as np
import scipy.linalg as sla

out = Path(__file__).parent
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- model
N, A = 5, 3                       # states 0..4, actions (left, stay, right)
T, Nt = 1.0, 100
dt = T / Nt
tm = (np.arange(Nt) + 0.5) * dt   # midpoints
noise, move = 0.1, 0.5
Q = np.zeros((A, N, N))           # Q[a, y, y'] jump rates
for y in range(N):
    for a in range(A):
        if y > 0:
            Q[a, y, y - 1] += noise + (move if a == 0 else 0.0)
        if y < N - 1:
            Q[a, y, y + 1] += noise + (move if a == 2 else 0.0)
        Q[a, y, y] = -Q[a, y].sum()
qbar = float(np.max(-np.einsum('ayy->ay', Q)))
alpha, kappa = 2.0, 0.1           # congestion weight, action cost
moves = np.array([1.0, 0.0, 1.0])
g = 0.25 * (np.arange(N) - 2.0) ** 2        # terminal cost, population independent
rho = np.array([0.5, 0.0, 0.0, 0.0, 0.5])


def running_cost(mu):             # mu (Nt, N) -> (Nt, N, A)
    return alpha * mu[:, :, None] + kappa * moves[None, None, :]


# constants of the paper
cH, ellH = 1.0, 2 * qbar
LA = np.sqrt(2 * cH ** 2 + 2 * ellH ** 2 * T * (1 + T))
LC = alpha / 2                    # D_E = 1, L_f = alpha
lam = tau = 0.99 / (LC + LA)      # theta = lam*LC + sqrt(lam*tau)*LA <= 1
theta = lam * LC + np.sqrt(lam * tau) * LA
C0 = np.sqrt(g.max() ** 2 + T * (alpha + kappa) ** 2)
print(f'qbar={qbar:.3f} LA={LA:.3f} LC={LC:.3f} lam=tau={lam:.4f} theta={theta:.3f} C0={C0:.3f}')


# ------------------------------------------------------------ operators
def defects(nu, q):
    """Kolmogorov defects d_s (midpoints) and d_T, in H^* = R^N."""
    mu = q.sum(2)
    n = np.einsum('tya,ayz->tz', q, Q)                 # n_t = mu_t Q^{gamma_t}
    cum = np.cumsum(n, axis=0) * dt - 0.5 * dt * n     # int_0^{s_i} n
    d = mu - rho[None, :] - cum
    dT = nu - rho - n.sum(0) * dt
    return d, dT


def riesz(d, dT):
    """v(t) = d_T + int_t^T d_s ds at midpoints; v(T) = d_T; dv/dt = -d."""
    tail = np.cumsum(d[::-1], axis=0)[::-1] * dt - 0.5 * dt * d
    return dT[None, :] + tail


def residual_norm(nu, q):
    d, dT = defects(nu, q)
    return np.sqrt(np.sum(dT ** 2) + np.sum(d ** 2) * dt)


def slacks(nu, q, psiT, psi, dpsi):
    mu = q.sum(2)
    G = g - psiT
    Gam = running_cost(mu) + dpsi[:, :, None] + np.einsum('ayz,tz->tya', Q, psi)
    return G, Gam


def tilt(p, s, axes):
    z = np.log(p) - lam * s
    z -= z.max(axis=axes, keepdims=True)
    r = np.exp(z)
    return r / r.sum(axis=axes, keepdims=True)


def xnorm(nu, q, nu2, q2):
    return np.sqrt(np.abs(nu - nu2).sum() ** 2 + np.sum(np.abs(q - q2).sum((1, 2)) ** 2) * dt)


def ynorm(psiT, dpsi):
    return np.sqrt(np.sum(psiT ** 2) + np.sum(dpsi ** 2) * dt)


# -------------------------------------------------- feasibility projection
def project(nu, q):
    """Disintegrate q = mu (x) gamma, re-simulate the chain under gamma from rho.
    Returns the exact occupation measure of the piecewise-constant feedback
    (midpoint values of the flow) and the exact running cost integrals."""
    mu = q.sum(2)
    gam = q / np.maximum(mu[:, :, None], 1e-300)
    gam /= gam.sum(2, keepdims=True)
    m = rho.copy()
    q_hat = np.zeros_like(q)
    mu_int = np.zeros((Nt, N))     # int over interval of the flow
    for i in range(Nt):
        Qg = np.einsum('ya,ayz->yz', gam[i], Q)
        aug = np.zeros((2 * N, 2 * N))
        aug[:N, :N] = Qg
        aug[:N, N:] = np.eye(N)
        E = sla.expm(aug * dt)
        mu_int[i] = m @ E[:N, N:]
        q_hat[i] = (m @ sla.expm(Qg * dt / 2))[:, None] * gam[i]
        m = m @ E[:N, :N]
    return m, q_hat, gam, mu_int


def exploitability(nu_hat, q_hat, gam, mu_int):
    """Expl of a feasible point: own cost minus HJB value for the frozen flow."""
    mu = q_hat.sum(2)
    fcost = running_cost(mu)                                  # (Nt, N, A)
    cost = g @ nu_hat + np.sum(mu_int * np.einsum('tya,tya->ty', fcost, gam))
    V = g.copy()
    for i in reversed(range(Nt)):                             # RK4 backward, f frozen on interval
        def rhs(v):
            return -np.min(fcost[i] + np.einsum('ayz,z->ya', Q, v), axis=1)
        k1 = rhs(V); k2 = rhs(V - dt / 2 * k1); k3 = rhs(V - dt / 2 * k2); k4 = rhs(V - dt * k3)
        V = V - dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return cost - rho @ V, V


# ------------------------------------------------------------- iteration
nu = np.full(N, 1.0 / N)
q = np.full((Nt, N, A), 1.0 / (N * A))
psiT, psi, dpsi = np.zeros(N), np.zeros((Nt, N)), np.zeros((Nt, N))
m_nu, m_q = nu.min(), q.min()
Dbar = np.log(1 / m_nu) + T * np.log(1 / m_q)
print(f'Dbar={Dbar:.3f}')

avg_nu, avg_q = np.zeros(N), np.zeros_like(q)
Kmax = 20000
checkpoints = sorted(set(np.unique(np.logspace(0, np.log10(Kmax), 40).astype(int)).tolist()))
rows = []
for k in range(1, Kmax + 1):
    # predictor (centre z_k)
    d, dT = defects(nu, q); v = riesz(d, dT)
    G, Gam = slacks(nu, q, psiT, psi, dpsi)
    nu_h, q_h = tilt(nu, G, (0,)), tilt(q, Gam, (1, 2))
    psiT_h, psi_h, dpsi_h = psiT - tau * dT, psi - tau * v, dpsi + tau * d
    # corrector (operator at z_{k+1/2}, centre z_k)
    d2, dT2 = defects(nu_h, q_h); v2 = riesz(d2, dT2)
    G2, Gam2 = slacks(nu_h, q_h, psiT_h, psi_h, dpsi_h)
    nu, q = tilt(nu, G2, (0,)), tilt(q, Gam2, (1, 2))
    psiT, psi, dpsi = psiT - tau * dT2, psi - tau * v2, dpsi + tau * d2
    avg_nu += (nu_h - avg_nu) / k
    avg_q += (q_h - avg_q) / k
    if k in checkpoints:
        res_avg = residual_norm(avg_nu, avg_q)
        nu_hat, q_hat, gam, mu_int = project(avg_nu, avg_q)
        dist = xnorm(nu_hat, q_hat, avg_nu, avg_q)
        expl, V = exploitability(nu_hat, q_hat, gam, mu_int)
        rows.append([k, res_avg, residual_norm(nu, q), dist, expl, ynorm(psiT, dpsi)])
        print(f'K={k:6d}  res_avg={res_avg:.3e}  res_last={rows[-1][2]:.3e}  '
              f'|proj-avg|={dist:.3e}  expl={expl:.3e}  |psi_K|={rows[-1][5]:.3f}')

rows = np.array(rows)
np.savetxt(out / 'finite_state_history.csv', rows, delimiter=',',
           header='K,residual_avg,residual_last,projection_distance,exploitability,dual_norm', comments='')
nu_hat, q_hat, gam, mu_int = project(avg_nu, avg_q)
expl, V = exploitability(nu_hat, q_hat, gam, mu_int)
np.savez(out / 'finite_state_result.npz', tm=tm, avg_nu=avg_nu, avg_q=avg_q, nu_hat=nu_hat, q_hat=q_hat,
         gam=gam, psiT=psiT, psi=psi, dpsi=dpsi, V0=V, rho=rho, Q=Q, g=g,
         params=np.array([lam, tau, theta, LA, LC, C0, Dbar, qbar, alpha, kappa]))
print('final exploitability of projected average', expl)
