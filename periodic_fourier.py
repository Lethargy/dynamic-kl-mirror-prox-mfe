"""Direct Fourier--Galerkin KL Mirror-Prox for a controlled circle diffusion.

On R/(2*pi Z), T=1 and X_0=0, the generator is L^a=a*d_x+d_xx,
a in {-1,+1}. The costs are f(x,mu)=sin(x)*int sin(y) mu(dy) and
g(x)=-kappa*sin(x). A positive uniform reference is used for terminal
and joint state--action masses. There is no jump-chain generator.

The real Fourier tests are 1/sqrt(2*pi), cos(nx)/sqrt(pi), sin(nx)/sqrt(pi).
Their H^3 Gram matrix is diag((1+n^2)^3), with each positive frequency
occurring twice. The differential generator acts analytically on these
tests. The time Riesz correction is evaluated by cumulative sums of
coefficient defects, with the anchored terminal-plus-derivative metric.

Time integrals use matching midpoint assembly: primal masses are stored
per interval and dual coefficients have a continuous P1 representation.
Thus the reported residual is the exact dual norm of this finite-test,
midpoint-assembled problem. It is not a certificate for the full diffusion
constraint. This script neither projects the masses to an admissible flow
nor reports stochastic-control exploitability.
"""
from __future__ import annotations

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np


def fourier_basis(points, modes):
    """Return basis values and their first two exact spatial derivatives."""
    points = np.asarray(points, dtype=float)
    frequencies = np.arange(1, modes + 1, dtype=float)
    angles = points[:, None] * frequencies
    cosine, sine = np.cos(angles) / np.sqrt(np.pi), np.sin(angles) / np.sqrt(np.pi)
    shape = (len(points), 2 * modes + 1)
    values, first, second = (np.zeros(shape) for _ in range(3))
    values[:, 0] = 1 / np.sqrt(2 * np.pi)
    values[:, 1::2], values[:, 2::2] = cosine, sine
    first[:, 1::2], first[:, 2::2] = -sine * frequencies, cosine * frequencies
    second[:, 1::2] = -cosine * frequencies ** 2
    second[:, 2::2] = -sine * frequencies ** 2
    return values, first, second


class FourierGame:
    def __init__(self, modes=8, nx=128, nt=50, kappa=1.0):
        if modes < 1 or nx < 2 * modes + 1 or nt < 1:
            raise ValueError("require modes>=1, Nx>=2*modes+1, Nt>=1")
        if not np.isfinite(kappa):
            raise ValueError("kappa must be finite")
        self.modes, self.nx, self.nt = modes, nx, nt
        self.kappa, self.dt = float(kappa), 1.0 / nt
        self.x = 2 * np.pi * np.arange(nx) / nx
        self.tm = (np.arange(nt) + 0.5) * self.dt
        self.actions = np.array([-1.0, 1.0])
        self.B, self.D1, self.D2 = fourier_basis(self.x, modes)
        self.frequencies = np.r_[0, np.repeat(np.arange(1, modes + 1), 2)]
        self.gram = (1 + self.frequencies ** 2).astype(float) ** 3
        self.gram_inverse = 1 / self.gram
        self.rho_moments = fourier_basis(np.array([0.0]), modes)[0][0]
        self.LB = self.D2[:, None, :] + self.actions[None, :, None] * self.D1[:, None, :]
        self.LB_flat = self.LB.reshape(2 * nx, -1)
        self.sin_x = np.sin(self.x)
        self.terminal_cost = -self.kappa * self.sin_x
        # Cauchy--Schwarz evaluation norms. Complete sine/cosine pairs make
        # these expressions constant in x, so the evaluated maxima are global.
        self.cH = float(np.sqrt(np.max(np.sum(self.B ** 2 / self.gram, axis=-1))))
        self.ellH = float(np.sqrt(np.max(np.sum(self.LB ** 2 / self.gram, axis=-1))))
        self.LC = 1.0  # |sin|<=1: a sharp-norm/total-variation Lipschitz bound.
        self.LA = float(np.sqrt(2 * self.cH ** 2 + 4 * self.ellH ** 2))

    def reference_logs(self):
        return (np.full(self.nx, -np.log(self.nx)),
                np.full((self.nt, self.nx, 2), -np.log(2 * self.nx)))

    def zero_dual(self):
        size = len(self.gram)
        return np.zeros(size), np.zeros((self.nt, size)), np.zeros((self.nt, size))

    def midpoint_values(self, terminal, derivative):
        """Evaluate the continuous P1 coefficients from their terminal anchor."""
        return terminal - self.dt * (np.cumsum(derivative[::-1], axis=0)[::-1]
                                     - 0.5 * derivative)

    def defects(self, nu, q):
        generator_moments = q.reshape(self.nt, -1) @ self.LB_flat
        cumulative = self.dt * (np.cumsum(generator_moments, axis=0)
                                - 0.5 * generator_moments)
        d = q.sum(-1) @ self.B - self.rho_moments - cumulative
        dT = nu @ self.B - self.rho_moments - self.dt * generator_moments.sum(0)
        return d, dT

    def riesz_coordinates(self, d, dT):
        terminal = dT * self.gram_inverse
        derivative = -d * self.gram_inverse
        return terminal, self.midpoint_values(terminal, derivative), derivative

    def dual_step(self, center, d, dT, step):
        return tuple(u - step * v for u, v in
                     zip(center, self.riesz_coordinates(d, dT)))

    def slacks(self, nu, q, dual):
        del nu  # The terminal cost here is independent of the terminal population.
        terminal, values, derivative = dual
        moment = q.sum(-1) @ self.sin_x
        G = self.terminal_cost - self.B @ terminal
        Gamma = ((moment[:, None] * self.sin_x[None, :]
                  + derivative @ self.B.T)[:, :, None]
                 + (values @ self.LB_flat.T).reshape(self.nt, self.nx, 2))
        return G, Gamma

    def residual_norm(self, nu, q):
        d, dT = self.defects(nu, q)
        square = np.dot(dT * self.gram_inverse, dT)
        square += self.dt * np.sum(d ** 2 * self.gram_inverse)
        return float(np.sqrt(max(0.0, square)))

    def dual_norm(self, dual):
        terminal, _, derivative = dual
        square = np.dot(terminal * self.gram, terminal)
        square += self.dt * np.sum(derivative ** 2 * self.gram)
        return float(np.sqrt(max(0.0, square)))


def tilt_logs(log_mass, slack, step, axes):
    """Keep log masses so numerical underflow does not remove support forever."""
    logits = log_mass - step * slack
    shifted = logits - np.max(logits, axis=axes, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axes, keepdims=True))


def self_tests(game):
    """Independent derivative, adjoint, positivity and constant-mode checks."""
    rng = np.random.default_rng(20261026)
    points = rng.uniform(0, 2 * np.pi, 13)
    B, D1, D2 = fourier_basis(points, game.modes)
    eps = 2e-5 / max(1, game.modes / 8)
    Bplus = fourier_basis(points + eps, game.modes)[0]
    Bminus = fourier_basis(points - eps, game.modes)[0]
    first_error = float(np.max(np.abs((Bplus - Bminus) / (2 * eps) - D1)))
    second_error = float(np.max(np.abs((Bplus - 2 * B + Bminus) / eps ** 2 - D2)))
    np.testing.assert_allclose((Bplus - Bminus) / (2 * eps), D1, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose((Bplus - 2 * B + Bminus) / eps ** 2, D2,
                               atol=5e-4, rtol=2e-5)
    np.testing.assert_allclose(2 * np.pi / game.nx * game.B.T @ game.B,
                               np.eye(len(game.gram)), atol=3e-14)
    nu = rng.random(game.nx)
    nu /= nu.sum()
    q = rng.random((game.nt, game.nx, 2))
    q /= q.sum((1, 2), keepdims=True)
    terminal = rng.normal(size=len(game.gram))
    derivative = rng.normal(size=(game.nt, len(game.gram)))
    values = game.midpoint_values(terminal, derivative)
    initial = terminal - game.dt * derivative.sum(0)
    d, dT = game.defects(nu, q)
    direct = (np.dot(nu @ game.B, terminal) - np.dot(game.rho_moments, initial)
              - game.dt * np.sum((q.sum(-1) @ game.B) * derivative)
              - game.dt * np.sum((q.reshape(game.nt, -1) @ game.LB_flat) * values))
    integrated = np.dot(dT, terminal) - game.dt * np.sum(d * derivative)
    np.testing.assert_allclose(direct, integrated, atol=3e-13, rtol=3e-13)
    riesz = game.riesz_coordinates(d, dT)
    represented = (np.dot(riesz[0] * game.gram, terminal)
                   + game.dt * np.sum(riesz[2] * game.gram * derivative))
    np.testing.assert_allclose(represented, direct, atol=3e-13, rtol=3e-13)
    np.testing.assert_allclose(d[:, 0], 0, atol=2e-15)
    np.testing.assert_allclose(dT[0], 0, atol=2e-15)
    np.testing.assert_array_equal(game.LB[..., 0], 0)
    tilted = np.exp(tilt_logs(np.log(q), rng.normal(size=q.shape), 0.3, (1, 2)))
    assert np.all(tilted > 0)
    np.testing.assert_allclose(tilted.sum((1, 2)), 1, atol=3e-15)
    reconstructed = game.midpoint_values(riesz[0], riesz[2])
    np.testing.assert_allclose(reconstructed, riesz[1], atol=2e-15)
    # Analytic global evaluation norms, independently of the quadrature nodes.
    n = np.arange(1, game.modes + 1, dtype=float)
    cH2 = 1 / (2 * np.pi) + np.sum(1 / (np.pi * (1 + n ** 2) ** 3))
    ellH2 = np.sum(n ** 2 / (np.pi * (1 + n ** 2) ** 2))
    np.testing.assert_allclose(game.cH ** 2, cH2, atol=3e-15)
    np.testing.assert_allclose(game.ellH ** 2, ellH2, atol=3e-15)
    return {"passed": True, "first_derivative_max_error": first_error,
            "second_derivative_max_error": second_error,
            "adjoint_absolute_error": float(abs(direct - integrated)),
            "riesz_absolute_error": float(abs(represented - direct)),
            "constant_mode_max_error": float(max(abs(dT[0]), np.max(abs(d[:, 0]))))}


def solve(game, iterations, step, output_dir, tag=None, tests=None):
    if iterations < 1 or not np.isfinite(step) or step <= 0:
        raise ValueError("require K>=1 and a positive finite step")
    theta = step * (game.LC + game.LA)
    if theta > 1 + 1e-13:
        raise ValueError(f"step violates sufficient bound: theta={theta:.8g}>1; "
                         f"use step<={1 / (game.LC + game.LA):.8g}")
    log_nu, log_q = game.reference_logs()
    nu, q = np.exp(log_nu), np.exp(log_q)
    dual = game.zero_dual()
    average_nu, average_q = np.zeros_like(nu), np.zeros_like(q)
    checkpoints = set(np.unique(np.geomspace(1, iterations, 32).astype(int)))
    checkpoints.add(iterations)
    initial_residual = game.residual_norm(nu, q)
    rows = []
    start = time.perf_counter()
    for k in range(1, iterations + 1):
        d, dT = game.defects(nu, q)
        G, Gamma = game.slacks(nu, q, dual)
        log_nu_half = tilt_logs(log_nu, G, step, (0,))
        log_q_half = tilt_logs(log_q, Gamma, step, (1, 2))
        nu_half, q_half = np.exp(log_nu_half), np.exp(log_q_half)
        dual_half = game.dual_step(dual, d, dT, step)
        d_half, dT_half = game.defects(nu_half, q_half)
        G_half, Gamma_half = game.slacks(nu_half, q_half, dual_half)
        # Both corrector tilts and the dual corrector retain the old center.
        log_nu = tilt_logs(log_nu, G_half, step, (0,))
        log_q = tilt_logs(log_q, Gamma_half, step, (1, 2))
        nu, q = np.exp(log_nu), np.exp(log_q)
        dual = game.dual_step(dual, d_half, dT_half, step)
        average_nu += (nu_half - average_nu) / k
        average_q += (q_half - average_q) / k
        if k in checkpoints:
            moment = average_q.sum(-1) @ game.sin_x
            terminal_moment = average_nu @ game.sin_x
            self_cost = game.dt * np.dot(moment, moment) - game.kappa * terminal_moment
            mass_error = max(abs(average_nu.sum() - 1),
                             np.max(np.abs(average_q.sum((1, 2)) - 1)))
            row = [k, game.residual_norm(average_nu, average_q),
                   game.residual_norm(nu, q), np.max(np.abs(moment)),
                   terminal_moment, game.dual_norm(dual), self_cost, mass_error]
            rows.append(row)
            print(f"K={k:6d} res_avg={row[1]:.6e} res_last={row[2]:.6e} "
                  f"max|m|={row[3]:.6e} terminal_sin={row[4]:.6e}", flush=True)
    elapsed = time.perf_counter() - start
    history = np.asarray(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if tag is None:
        tag = (f"kappa{game.kappa:g}_M{game.modes}_Nx{game.nx}_Nt{game.nt}"
               f"_K{iterations}_step{step:g}").replace(".", "p")
    if Path(tag).name != tag:
        raise ValueError("tag must be a file name without directory components")
    history_path, result_path = output_dir / f"{tag}_history.csv", output_dir / f"{tag}_result.npz"
    summary_path = output_dir / f"{tag}_summary.json"
    np.savetxt(history_path, history, delimiter=",", comments="",
               header="K,residual_avg,residual_last,max_abs_moment,terminal_sin_moment,"
                      "dual_norm,self_cost_unprojected,mass_error")
    np.savez_compressed(result_path, x=game.x, tm=game.tm, actions=game.actions,
                        average_nu=average_nu, average_q=average_q, last_nu=nu,
                        last_q=q, last_log_nu=log_nu, last_log_q=log_q,
                        dual_terminal=dual[0], dual_values=dual[1], dual_derivative=dual[2],
                        moment=average_q.sum(-1) @ game.sin_x, gram_diagonal=game.gram,
                        history=history, initial_residual=initial_residual)
    summary = {
        "model": "circle length 2*pi; dX=a dt+sqrt(2)dW, a=-1,+1, X0=0, T=1",
        "running_cost": "sin(x)*integral sin(y) mu_t(dy)",
        "terminal_cost": "-kappa*sin(x)",
        "method": "direct Fourier spatial generator; continuous P1 dual time coefficients; midpoint assembly",
        "limitations": "finite Fourier tests and fixed primal quadrature; no feasibility projection, exploitability, or continuum convergence certificate",
        "parameters": {"modes": game.modes, "basis_dimension": len(game.gram),
                       "Nx": game.nx, "Nt": game.nt, "K": iterations,
                       "lambda": step, "tau": step, "kappa": game.kappa},
        "constants": {"cH": game.cH, "ellH": game.ellH, "LC": game.LC,
                      "LA": game.LA, "theta": theta},
        "initial_residual": initial_residual, "final_residual_avg": float(history[-1, 1]),
        "final_residual_last": float(history[-1, 2]),
        "max_abs_moment": float(history[-1, 3]),
        "terminal_sin_moment": float(history[-1, 4]),
        "dual_norm": float(history[-1, 5]),
        "self_cost_unprojected": float(history[-1, 6]),
        "mass_error": float(history[-1, 7]),
        "minimum_average_terminal_mass": float(average_nu.min()),
        "minimum_average_joint_mass": float(average_q.min()),
        "maximum_constant_defect": float(max(np.max(np.abs(game.defects(average_nu, average_q)[0][:, 0])),
                                              abs(game.defects(average_nu, average_q)[1][0]))),
        "elapsed_seconds": elapsed, "self_tests": tests,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "files": {"history": history_path.name, "results": result_path.name},
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--Nx", type=int, default=128)
    parser.add_argument("--Nt", type=int, default=50)
    parser.add_argument("--K", type=int, default=20000)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "periodic_fourier_histories")
    parser.add_argument("--tag")
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    game = FourierGame(args.modes, args.Nx, args.Nt, args.kappa)
    tests = self_tests(game)
    print("self-tests:", json.dumps(tests), flush=True)
    print(f"cH={game.cH:.8g}, ellH={game.ellH:.8g}, LA={game.LA:.8g}, "
          f"theta={args.step * (game.LC + game.LA):.8g}", flush=True)
    if not args.self_test_only:
        solve(game, args.K, args.step, args.output_dir, args.tag, tests)


if __name__ == "__main__":
    main()
