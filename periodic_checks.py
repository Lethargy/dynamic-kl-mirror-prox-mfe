"""Independent numerical checks for the periodic controlled-diffusion solver.

Run directly; this does not create, replace, or alter any saved experiment.
"""
from itertools import product
import json

import numpy as np

from periodic_diffusion_grid import (PeriodicGame, CyclicImplicit,
                                     independent_checks, implicit_matrix,
                                     project_and_exploit)


def check_fourier_constants():
    game = PeriodicGame(24, 6, "S", 3)
    lap = np.zeros((game.nx, game.nx))
    idx = np.arange(game.nx)
    lap[idx, idx] = -2 / game.h ** 2
    lap[idx, (idx + 1) % game.nx] = 1 / game.h ** 2
    lap[idx, (idx - 1) % game.nx] = 1 / game.h ** 2
    gram = game.h * np.linalg.matrix_power(np.eye(game.nx) - lap, 3)
    inv = np.linalg.inv(gram)
    cH = np.sqrt(np.max(np.diag(inv)))
    ellH = 0.0
    for a in range(2):
        q = np.eye(game.nx) - implicit_matrix(np.full(game.nx, game.up[a]),
                                             np.full(game.nx, game.lo[a]), 1)
        ellH = max(ellH, np.sqrt(np.max(np.diag(q @ inv @ q.T))))
    error = max(abs(cH - game.cH), abs(ellH - game.ellH))
    assert error < 2e-10, error
    return error


def check_balanced_projection():
    game = PeriodicGame(32, 8, "S", 3, 0)
    q = np.full((game.nt, game.nx, 2), 1 / (2 * game.nx))
    result = project_and_exploit(game, q, 128)
    multiplier = 1 - result["delta"] * (1 + game.h / 2) * game.lap_eigenvalues
    exact_ie = np.fft.ifft(multiplier ** -result["nsub_actual"] *
                          np.fft.fft(game.rho)).real
    terminal_error = float(np.max(np.abs(result["nu_hat"] - exact_ie)))
    assert terminal_error < 2e-13, terminal_error
    assert np.max(np.abs(result["mhat"])) < 2e-13
    assert abs(result["exploitability"]) < 2e-13
    assert result["policy_adjoint_error"] < 2e-13
    return dict(terminal_fourier_error=terminal_error,
                maximum_sine_moment=float(np.max(np.abs(result["mhat"]))),
                exploitability=result["exploitability"])


def check_howard_exhaustively():
    """Enumerate all 64 stationary policies per implicit step on a tiny game."""
    game = PeriodicGame(6, 2, "S", 3, 1)
    rng = np.random.default_rng(94)
    q = rng.dirichlet(np.ones(12), size=2).reshape(2, 6, 2)
    result = project_and_exploit(game, q, 4)
    dt = result["delta"]
    matrices = []
    for policy in product((0, 1), repeat=6):
        choices = np.asarray(policy)
        matrices.append(implicit_matrix(game.up[choices], game.lo[choices], dt))
    value = game.gterm.copy()
    for moment in result["mhat"][::-1]:
        rhs = value + dt * game.phi * moment
        all_policy_values = np.array([np.linalg.solve(matrix, rhs) for matrix in matrices])
        value = np.min(all_policy_values, axis=0)
    error = float(abs(result["best_response_value"] - value[0]))
    assert error < 2e-13, error
    return error


def check_assembled_balanced_feasibility():
    """CN Fourier formula solves the midpoint variational constraints.

    This reference belongs to the assembled system; it is deliberately kept
    distinct from both the continuous-time chain and circular heat reference.
    Fourier roundoff can produce negative masses of order machine epsilon.
    """
    rows = []
    for nx, nt in ((64, 25), (128, 50), (256, 100)):
        game = PeriodicGame(nx, nt, "S", 3, 0)
        z = 0.5 * game.dt * (1 + game.h / 2) * game.lap_eigenvalues
        multiplier = (1 + z) / (1 - z)
        midpoint = np.array([np.fft.ifft(multiplier ** n / (1 - z)).real
                             for n in range(nt)])
        terminal = np.fft.ifft(multiplier ** nt).real
        q = np.repeat(0.5 * midpoint[..., None], 2, axis=2)
        residual, _ = game.residuals(terminal, q)
        minimum = float(min(q.min(), terminal.min()))
        assert residual < 2e-12 and minimum > -1e-14
        assert np.max(np.abs(midpoint.sum(1) - 1)) < 2e-13
        rows.append(dict(Nx=nx, Nt=nt, restricted_residual=residual,
                         minimum_mass=minimum,
                         maximum_sine_moment=float(np.max(np.abs(midpoint @ game.phi)))))
    return rows


def main():
    report = {"S3": independent_checks(PeriodicGame(32, 7, "S", 3)),
              "Euclidean": independent_checks(PeriodicGame(32, 7, "E", 0)),
              "fourier_constants_error": check_fourier_constants(),
              "balanced_policy": check_balanced_projection(),
              "howard_exhaustive_error": check_howard_exhaustively(),
              "assembled_balanced_feasibility": check_assembled_balanced_feasibility()}
    print(json.dumps(report, indent=2))
    print("All periodic numerical checks passed.")


if __name__ == "__main__":
    main()
