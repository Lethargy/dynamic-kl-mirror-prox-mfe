"""Periodic controlled diffusion: PD-KL Mirror-Prox with FFT spatial Riesz solves.

Model on R/(2*pi Z), T=1, X_0=0:
    dX_t = a_t dt + sqrt(2) dW_t, a in {-1,+1},
    f(x,mu) = sin(x) * integral sin(y) mu(dy), g(x) = -kappa*sin(x).

The spatial approximation uses cyclic upwind Markov generators on Nx distinct
nodes. The primal reference is uniform on nodes and actions at every time,
including the terminal reference. Dual values and derivatives are the P1-time
Galerkin representation; the primal has one mass vector per interval. The
terminal-plus-derivative Riesz equation and generator terms use matching
midpoint assembly. This is a finite variational problem, not an exact
continuous-time or continuum diffusion computation.

Projection and best response use the same implicit-Euler scheme. Cyclic
tridiagonal solves include both wraparound corners and their transposes.
The projected flow's exploitability is a same-grid numerical diagnostic.
For kappa=0 an analytic circular heat flow is saved as one equilibrium
reference; the game need not select that distribution. No whole-line or
reflecting-grid benchmark is loaded.
"""
from __future__ import annotations

import os

# Set before importing numerical libraries; users can explicitly override.
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
import scipy.linalg as sla
from scipy.special import logsumexp


class CyclicImplicit:
    """Factor I-delta*Q or its transpose, Q with row rates up/lo.

    A noncyclic tridiagonal LU plus a rank-two Woodbury correction gives O(N)
    solves. The corner entries are retained even for nonconstant feedbacks.
    """

    def __init__(self, up, lo, delta, transpose=False):
        up, lo = np.asarray(up), np.asarray(lo)
        self.size = len(up)
        diag = 1.0 + delta * (up + lo)
        upper, lower = -delta * up[:-1], -delta * lo[1:]
        corner_0n, corner_n0 = -delta * lo[0], -delta * up[-1]
        if transpose:
            upper, lower = lower.copy(), upper.copy()
            corner_0n, corner_n0 = corner_n0, corner_0n
        self.lu = sla.lapack.dgttrf(lower.copy(), diag.copy(), upper.copy())
        if self.lu[-1] != 0:
            raise RuntimeError(f"tridiagonal factorization failed: {self.lu[-1]}")
        update = np.zeros((self.size, 2))
        update[0, 0], update[-1, 1] = corner_0n, corner_n0
        self.z = self._base_solve(update)
        self.small = np.eye(2) + self.z[[-1, 0], :]

    def _base_solve(self, rhs):
        arr = np.asarray(rhs, dtype=float)
        vector = arr.ndim == 1
        matrix = arr[:, None] if vector else arr
        result, info = sla.lapack.dgttrs(*self.lu[:-1], matrix.copy())
        if info != 0:
            raise RuntimeError(f"tridiagonal solve failed: {info}")
        return result[:, 0] if vector else result

    def solve(self, rhs):
        y = self._base_solve(rhs)
        coeff = np.linalg.solve(self.small, y[[-1, 0], ...])
        return y - self.z @ coeff


class PeriodicGame:
    def __init__(self, nx=64, nt=40, metric="S", order=3, kappa=1.0):
        if nx < 4 or nt < 1 or order < 0:
            raise ValueError("require Nx>=4, Nt>=1 and m>=0")
        self.nx, self.nt, self.metric, self.order = nx, nt, metric, order
        self.kappa, self.h, self.dt = float(kappa), 2 * np.pi / nx, 1.0 / nt
        self.x = np.arange(nx) * self.h
        self.tm = (np.arange(nt) + 0.5) * self.dt
        self.phi = np.sin(self.x)
        self.gterm = -self.kappa * self.phi
        self.rho = np.zeros(nx)
        self.rho[0] = 1.0
        self.acts = np.array([-1.0, 1.0])
        self.up = 1 / self.h ** 2 + np.maximum(self.acts, 0) / self.h
        self.lo = 1 / self.h ** 2 + np.maximum(-self.acts, 0) / self.h
        self.qbar = float(2 / self.h ** 2 + 1 / self.h)
        freq = 2 * np.pi * np.arange(nx) / nx
        self.lap_eigenvalues = -4 * np.sin(freq / 2) ** 2 / self.h ** 2
        self.gram_eigenvalues = (np.ones(nx) if metric == "E" else
                                self.h * (1 - self.lap_eigenvalues) ** order)
        self.gram_rfft = self.gram_eigenvalues[:nx // 2 + 1]
        self.cH = float(np.sqrt(np.mean(1 / self.gram_eigenvalues)))
        qsymbol = (self.up[:, None] * np.expm1(1j * freq) +
                   self.lo[:, None] * np.expm1(-1j * freq))
        self.ellH = float(np.sqrt(np.max(np.mean(
            np.abs(qsymbol) ** 2 / self.gram_eigenvalues, axis=1))))
        self.LC = float(np.max(np.abs(self.phi)) ** 2)
        self.LA = float(np.sqrt(2 * self.cH ** 2 + 4 * self.ellH ** 2))

    def RH(self, d):
        return np.fft.irfft(np.fft.rfft(d, axis=-1) / self.gram_rfft,
                           n=self.nx, axis=-1)

    def H_sq(self, u):
        # Parseval with numpy's unnormalized forward FFT.
        transformed = np.fft.fft(u, axis=-1)
        return np.sum(np.abs(transformed) ** 2 * self.gram_eigenvalues,
                      axis=-1) / self.nx

    def Hstar_sq(self, d):
        return np.sum(d * self.RH(d), axis=-1)

    def apply_Q(self, values):
        right = np.roll(values, -1, axis=-1) - values
        left = np.roll(values, 1, axis=-1) - values
        return right[..., None] * self.up + left[..., None] * self.lo

    def apply_QT(self, q):
        return np.sum(np.roll(q, 1, axis=-2) * self.up +
                      np.roll(q, -1, axis=-2) * self.lo -
                      q * (self.up + self.lo), axis=-1)

    def defects(self, nu, q):
        n = self.apply_QT(q)
        cumulative = self.dt * (np.cumsum(n, axis=0) - 0.5 * n)
        return q.sum(-1) - self.rho - cumulative, nu - self.rho - self.dt * n.sum(0)

    def riesz_coordinates(self, d, dT):
        tail = self.dt * (np.cumsum(d[::-1], axis=0)[::-1] - 0.5 * d)
        mapped = self.RH(np.concatenate((dT[None], dT[None] + tail, d)))
        # v'= -R_H*d. Return (v(T), v(midpoints), v'(midpoints)).
        return mapped[0], mapped[1:self.nt + 1], -mapped[self.nt + 1:]

    def dual_step(self, center, d, dT, step):
        return tuple(u - step * v for u, v in
                     zip(center, self.riesz_coordinates(d, dT)))

    def slacks(self, nu, q, dual):
        terminal, values, derivative = dual
        moment = q.sum(-1) @ self.phi
        return (self.gterm - terminal,
                moment[:, None, None] * self.phi[None, :, None] +
                derivative[..., None] + self.apply_Q(values))

    def residuals(self, nu, q):
        d, dT = self.defects(nu, q)
        own = np.sqrt(max(0.0, float(self.Hstar_sq(dT) +
                                   self.dt * np.sum(self.Hstar_sq(d)))))
        tv = np.sqrt(np.abs(dT).sum() ** 2 +
                     self.dt * np.sum(np.abs(d).sum(-1) ** 2))
        return float(own), float(tv)

    def ynorm(self, dual):
        return float(np.sqrt(max(0.0, float(self.H_sq(dual[0]) +
                                           self.dt * np.sum(self.H_sq(dual[2]))))))

    def relaxed_rates(self, policy):
        return policy @ self.up, policy @ self.lo

    def circular_heat_mass(self, time_value):
        # At positive time, Fourier heat series on the continuous circle.
        modes = np.arange(1, max(20, int(np.ceil(np.sqrt(40 / time_value)))))
        density = (1 + 2 * (np.exp(-modes ** 2 * time_value)[:, None] *
                            np.cos(modes[:, None] * self.x)).sum(0)) / (2 * np.pi)
        mass = np.maximum(density, 0) * self.h
        return mass / mass.sum()


def feedback(q):
    marginal = q.sum(-1, keepdims=True)
    policy = np.full_like(q, 0.5)
    np.divide(q, marginal, out=policy, where=marginal > 0)
    return policy


def implicit_matrix(up, lo, delta):
    n = len(up)
    mat = np.diag(1 + delta * (up + lo))
    index = np.arange(n)
    mat[index, (index + 1) % n] = -delta * up
    mat[index, (index - 1) % n] = -delta * lo
    return mat


def independent_checks(game):
    """Independent dense checks cover corners, transpose, adjoint and FFT mass."""
    rng = np.random.default_rng(820)
    n, nt, dt = game.nx, game.nt, game.dt
    errors = {}
    dense_Q = []
    for action in range(2):
        q = (np.eye(n) - implicit_matrix(np.full(n, game.up[action]),
                                         np.full(n, game.lo[action]), 1.0))
        dense_Q.append(q)
        off = q.copy()
        np.fill_diagonal(off, 0)
        assert off.min() >= 0
        assert np.max(np.abs(q.sum(1))) < 2e-11
    test = rng.normal(size=(nt, n))
    masses = rng.uniform(size=(nt, n, 2))
    errors["generator_dense_error"] = float(max(
        np.max(np.abs(game.apply_Q(test)[..., a] - test @ dense_Q[a].T))
        for a in range(2)))
    errors["generator_duality_error"] = float(abs(
        np.sum(game.apply_Q(test) * masses) - np.sum(test * game.apply_QT(masses))))
    policy = rng.dirichlet([1, 1], size=n)
    up, lo = game.relaxed_rates(policy)
    delta = 0.017
    matrix = implicit_matrix(up, lo, delta)
    rhs = rng.normal(size=n)
    solved = CyclicImplicit(up, lo, delta).solve(rhs)
    solved_T = CyclicImplicit(up, lo, delta, transpose=True).solve(rhs)
    errors["cyclic_solve_error"] = float(np.max(np.abs(matrix @ solved - rhs)))
    errors["cyclic_transpose_error"] = float(np.max(np.abs(matrix.T @ solved_T - rhs)))
    positive = CyclicImplicit(up, lo, delta, transpose=True).solve(game.rho)
    errors["cyclic_mass_error"] = float(abs(positive.sum() - 1))
    assert positive.min() > -2e-13
    # Dense Fourier Gram, not a call to RH's implementation.
    freq = np.arange(n)
    fourier = np.exp(-2j * np.pi * freq[:, None] * freq[None, :] / n) / np.sqrt(n)
    inverse_gram = np.real(fourier.conj().T @
                           (fourier / game.gram_eigenvalues[:, None]))
    vector = rng.normal(size=n)
    fft_solution = game.RH(vector)
    reference_solution = inverse_gram @ vector
    errors["riesz_inverse_relative_error"] = float(
        np.linalg.norm(fft_solution - reference_solution) / np.linalg.norm(reference_solution))
    # Create arbitrary P1 test data via terminal and derivative coordinates.
    terminal = rng.normal(size=n)
    derivative = rng.normal(size=(nt, n))
    values = terminal - dt * (np.cumsum(derivative[::-1], axis=0)[::-1] -
                              0.5 * derivative)
    phi0 = terminal - dt * derivative.sum(0)
    nu = rng.dirichlet(np.ones(n))
    primal = rng.dirichlet(np.ones(n * 2), size=nt).reshape(nt, n, 2)
    d, dT = game.defects(nu, primal)
    weak = (nu @ terminal - game.rho @ phi0 -
            dt * np.sum(primal * (derivative[..., None] + game.apply_Q(values))))
    from_defects = dT @ terminal - dt * np.sum(d * derivative)
    errors["midpoint_adjoint_error"] = float(abs(weak - from_defects))
    vT, vv, vd = game.riesz_coordinates(d, dT)
    representation_error = vv - (vT - dt * (np.cumsum(vd[::-1], axis=0)[::-1] - 0.5 * vd))
    errors["time_riesz_representation_error"] = float(np.max(np.abs(representation_error)))
    for key, value in errors.items():
        assert value < 1e-8, (key, value)
    return errors


def project_and_exploit(game, avg_q, nsub):
    """Same-scheme forward projection and backward policy/HJB evaluations."""
    ns = max(1, int(np.ceil(nsub / game.nt)))
    delta = game.dt / ns
    policies = feedback(avg_q)
    mu = game.rho.copy()
    mhat = np.zeros(game.nt * ns)
    min_mass, mass_error = 0.0, 0.0
    caches = []
    for interval in range(game.nt):
        up, lo = game.relaxed_rates(policies[interval])
        forward = CyclicImplicit(up, lo, delta, transpose=True)
        backward = CyclicImplicit(up, lo, delta)
        caches.append(backward)
        for sub in range(ns):
            mu = forward.solve(mu)
            min_mass = min(min_mass, float(mu.min()))
            mass_error = max(mass_error, float(abs(mu.sum() - 1)))
            mhat[interval * ns + sub] = mu @ game.phi
    nu_hat = mu.copy()
    cost = delta * np.sum(mhat ** 2) + game.gterm @ nu_hat
    policy_value, best_value = game.gterm.copy(), game.gterm.copy()
    chosen = np.ones(game.nx, dtype=int)
    max_howard, max_bellman = 0, 0.0
    for interval in reversed(range(game.nt)):
        for sub in reversed(range(ns)):
            frozen = game.phi * mhat[interval * ns + sub]
            policy_value = caches[interval].solve(policy_value + delta * frozen)
            rhs = best_value + delta * frozen
            for iteration in range(100):
                up, lo = game.up[chosen], game.lo[chosen]
                candidate = CyclicImplicit(up, lo, delta).solve(rhs)
                qvalues = game.apply_Q(candidate)
                diff = qvalues[:, 1] - qvalues[:, 0]
                new_chosen = np.where(diff < -1e-12, 1,
                                      np.where(diff > 1e-12, 0, chosen))
                if np.array_equal(chosen, new_chosen):
                    break
                chosen = new_chosen
            else:
                raise RuntimeError("Howard iteration did not converge")
            best_value = candidate
            max_howard = max(max_howard, iteration + 1)
            bellman = candidate - delta * np.min(qvalues, axis=1) - rhs
            max_bellman = max(max_bellman, float(np.max(np.abs(bellman))))
    adjoint_error = float(abs(cost - policy_value[0]))
    exploitability = float(policy_value[0] - best_value[0])
    if adjoint_error > 2e-8 or max_bellman > 2e-8 or exploitability < -2e-8:
        raise RuntimeError("forward/HJB consistency check failed")
    return dict(exploitability=exploitability, cost=float(cost),
                cost_via_adjoint=float(policy_value[0]),
                best_response_value=float(best_value[0]), mhat=mhat, nu_hat=nu_hat,
                delta=delta, nsub_actual=game.nt * ns,
                policy_adjoint_error=adjoint_error, bellman_error=max_bellman,
                max_howard_iterations=max_howard, projection_mass_error=mass_error,
                projection_min_mass=min_mass)


def normalized_log_tilt(log_probability, slack, step, axes):
    tilted = log_probability - step * slack
    return tilted - logsumexp(tilted, axis=axes, keepdims=True)


def run(args):
    game = PeriodicGame(args.Nx, args.Nt, args.metric, args.m, args.kappa)
    safe_step = 0.99 / (game.LC + game.LA)
    step = safe_step if args.step == "auto" else float(args.step)
    if not np.isfinite(step) or step <= 0 or args.K < 1:
        raise ValueError("positive finite step and positive K required")
    theta = step * (game.LC + game.LA)
    here = Path(__file__).resolve().parent
    outdir = Path(args.outdir) if args.outdir else here / "periodic_grid_histories"
    outdir.mkdir(parents=True, exist_ok=True)
    metric_name = "E" if args.metric == "E" else f"S{args.m}"
    tag = args.tag or f"periodic_k{args.kappa:g}_{metric_name}_Nx{args.Nx}_Nt{args.Nt}_step{args.step}"
    paths = [outdir / f"{tag}_{suffix}" for suffix in ("history.csv", "summary.json", "result.npz")]
    if any(p.exists() for p in paths):
        raise FileExistsError(f"refusing to overwrite existing outputs for {tag}")

    def log(message):
        if not args.quiet:
            print(message, flush=True)

    checks = independent_checks(game)
    log(f"[{tag}] Nx={game.nx} Nt={game.nt} h={game.h:.5g} cH={game.cH:.5g} "
        f"ellH={game.ellH:.5g} LA={game.LA:.5g} eta={step:.5g} theta={theta:.5g}")
    log_nu = np.full(game.nx, -np.log(game.nx))
    log_q = np.full((game.nt, game.nx, 2), -np.log(2 * game.nx))
    nu, q = np.exp(log_nu), np.exp(log_q)
    dual = (np.zeros(game.nx), np.zeros((game.nt, game.nx)),
            np.zeros((game.nt, game.nx)))
    avg_nu, avg_q = np.zeros_like(nu), np.zeros_like(q)
    initial_residual = game.residuals(nu, q)
    checkpoints = set(np.geomspace(1, args.K, max(2, args.ncheck)).astype(int)) | {args.K}
    rows, t_iter, t_check = [], 0.0, 0.0
    stable, dual_identity_error = True, 0.0
    max_normalization_error = 0.0
    for k in range(1, args.K + 1):
        tic = time.perf_counter()
        d, dT = game.defects(nu, q)
        terminal_slack, running_slack = game.slacks(nu, q, dual)
        log_nu_half = normalized_log_tilt(log_nu, terminal_slack, step, (0,))
        log_q_half = normalized_log_tilt(log_q, running_slack, step, (1, 2))
        nu_half, q_half = np.exp(log_nu_half), np.exp(log_q_half)
        dual_half = game.dual_step(dual, d, dT, step)
        d_half, dT_half = game.defects(nu_half, q_half)
        terminal_slack, running_slack = game.slacks(nu_half, q_half, dual_half)
        log_nu = normalized_log_tilt(log_nu, terminal_slack, step, (0,))
        log_q = normalized_log_tilt(log_q, running_slack, step, (1, 2))
        nu, q = np.exp(log_nu), np.exp(log_q)
        dual = game.dual_step(dual, d_half, dT_half, step)
        avg_nu += (nu_half - avg_nu) / k
        avg_q += (q_half - avg_q) / k
        t_iter += time.perf_counter() - tic
        if k not in checkpoints:
            continue
        tic = time.perf_counter()
        own, tv = game.residuals(avg_nu, avg_q)
        last, _ = game.residuals(nu, q)
        moment = avg_q.sum(-1) @ game.phi
        mabs, dual_norm = float(np.max(np.abs(moment))), game.ynorm(dual)
        d_avg, dT_avg = game.defects(avg_nu, avg_q)
        expected = game.riesz_coordinates(d_avg, dT_avg)
        dual_identity_error = max(dual_identity_error, *(float(np.max(np.abs(u + k * step * v)))
                                                       for u, v in zip(dual, expected)))
        max_normalization_error = max(max_normalization_error, abs(avg_nu.sum() - 1),
                                      float(np.max(np.abs(avg_q.sum((1, 2)) - 1))))
        projected = project_and_exploit(game, avg_q, args.nsub) if k >= 50 else None
        exploitability = projected["exploitability"] if projected else np.nan
        reference_error = mabs if args.kappa == 0 else np.nan
        t_check += time.perf_counter() - tic
        rows.append([k, own, last, tv, mabs, dual_norm, exploitability, reference_error, t_iter])
        log(f"K={k:7d} res={own:.4e} TVdef={tv:.4e} max|m|={mabs:.4e} "
            f"expl={exploitability:.4e} |psi|={dual_norm:.4e} {1e3*t_iter/k:.3f}ms/it")
        if not np.isfinite(own + dual_norm) or own > 1e6:
            stable = False
            break
    tic = time.perf_counter()
    projected = project_and_exploit(game, avg_q, args.nsub)
    t_check += time.perf_counter() - tic
    rows = np.asarray(rows)
    rows[-1, 6] = projected["exploitability"]
    m_avg = avg_q.sum(-1) @ game.phi
    reference_mass = game.circular_heat_mass(1.0)
    summary = dict(tag=tag, model="periodic_sine_controlled_diffusion", kappa=args.kappa,
                   metric=metric_name, m=args.m if args.metric == "S" else None,
                   Nx=game.nx, Nt=game.nt, T=1.0, circumference=2 * np.pi, h=game.h,
                   lam=step, tau=step, cH=game.cH, ellH=game.ellH, LC=game.LC, LA=game.LA,
                   theta=theta, lam_max=safe_step, step_condition_verified=bool(theta <= 1),
                   qbar=game.qbar, K=k, stable=stable, res_avg=float(rows[-1, 1]),
                   res_last=float(rows[-1, 2]), res_avg_tv=float(rows[-1, 3]),
                   max_abs_moment=float(np.max(np.abs(m_avg))),
                   moment_l2=float(np.sqrt(game.dt * np.sum(m_avg ** 2))),
                   dual_norm=game.ynorm(dual), exploitability=projected["exploitability"],
                   cost=projected["cost"], cost_via_adjoint=projected["cost_via_adjoint"],
                   err_m_reference=(float(np.max(np.abs(m_avg))) if args.kappa == 0 else None),
                   reference="analytic circular heat flow; one equilibrium for kappa=0 only",
                   terminal_tv_to_heat_reference=(float(0.5 * np.sum(np.abs(avg_nu-reference_mass)))
                                                   if args.kappa == 0 else None),
                   sec_per_iter=t_iter / k, sec_iter_total=t_iter, sec_check_total=t_check,
                   res0_own=initial_residual[0], res0_tv=initial_residual[1],
                   delta_sub=projected["delta"], nsub_actual=projected["nsub_actual"],
                   initialization="uniform spatial/action/terminal reference",
                   reference_min_terminal_mass=1 / game.nx,
                   reference_min_running_mass=1 / (2 * game.nx),
                   uniform_entropy_bound=float(np.log(game.nx) + np.log(2*game.nx)),
                   dual_drift_max_coefficient_error=dual_identity_error,
                   max_normalization_error=max_normalization_error,
                   independent_checks=checks,
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   command=[sys.executable, *sys.argv], python_version=platform.python_version(),
                   numpy_version=np.__version__, scipy_version=scipy.__version__,
                   threads={key:os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                                               "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
    for key in ("best_response_value", "policy_adjoint_error", "bellman_error",
                "max_howard_iterations", "projection_mass_error", "projection_min_mass"):
        summary[key] = projected[key]
    if max_normalization_error > 1e-10 or not stable:
        raise RuntimeError("normalization or stability failure")
    np.savetxt(paths[0], rows, delimiter=",", comments="",
               header="K,residual_avg,residual_last,residual_avg_tv,max_abs_moment,dual_norm,exploitability,err_m_reference,seconds")
    paths[1].write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(paths[2], xg=game.x, yg=game.x, tm=game.tm, m_avg=m_avg,
                        avg_nu=avg_nu, avg_q=avg_q, nu_hat=projected["nu_hat"],
                        mhat=projected["mhat"], heat_reference_mass=reference_mass,
                        tsub=(np.arange(projected["nsub_actual"]) + 1) * projected["delta"],
                        psiT=dual[0], psi_slices=dual[1][[0, game.nt//4, game.nt//2, 3*game.nt//4]],
                        history=rows, params=np.array([step, step, args.kappa, game.nx, game.nt,
                                                       game.cH, game.ellH, game.LA, theta]))
    log(f"Saved {tag}: {t_iter:.2f}s iteration + {t_check:.2f}s diagnostics; "
        f"adjoint error={summary['policy_adjoint_error']:.2e}, Bellman error={summary['bellman_error']:.2e}")
    return summary


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--Nx", type=int, default=64)
    ap.add_argument("--Nt", type=int, default=40)
    ap.add_argument("--metric", choices=["E", "S"], default="S")
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--K", type=int, default=20000)
    ap.add_argument("--step", default="auto")
    ap.add_argument("--nsub", type=int, default=800,
                    help="requested total implicit-Euler substeps (rounded up to a multiple of Nt)")
    ap.add_argument("--ncheck", type=int, default=18)
    ap.add_argument("--outdir")
    ap.add_argument("--tag")
    ap.add_argument("--quiet", action="store_true")
    return ap


if __name__ == "__main__":
    run(parser().parse_args())
