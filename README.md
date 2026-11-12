# Numerical source and results

This directory contains the scripts and saved outputs for the experiments in
`updated notations.tex`, together with the earlier experiments in `PD-KL-MP.tex`.

## Requirements

Use Python 3 with NumPy, SciPy, and Matplotlib. The saved CSV and NPZ files in
this directory are the data used for the figures currently included in the
paper.

## Periodic controlled diffusion

The updated notation draft uses the game on the circle `R/(2*pi Z)`:
`dX = a dt + sqrt(2) dW (mod 2*pi)`, `a in {-1,+1}`, `X_0=0`,
`f(x,mu)=sin(x)*integral sin(y) mu(dy)`, and `g(x)=-kappa*sin(x)`.
Both the terminal reference and each running state law are uniform on the
periodic grid; the action reference assigns probability one half to each
of the two actions. This initialization is strictly positive and need not
satisfy the initial-condition constraint.

`periodic_diffusion_grid.py` uses cyclic transition rates and an FFT solve for
the periodic Sobolev Riesz map. `run_periodic_sweep.py` runs the 12 cases:
`(Nx,Nt)=(64,25),(128,50),(256,100)`, `kappa=0,1`, and the Sobolev and
Euclidean metrics. Every run uses 20,000 proximal iterations. The Sobolev
step is 0.25; each Euclidean step is `0.99/(L_C+L_A)`. The saved outputs are
in `periodic_grid_histories/`; the manifest records exact commands and a
source hash. The collected results are `periodic_sweep_summary.csv` in that
folder. Reproduce into a fresh directory:

```sh
python run_periodic_sweep.py --K 20000 --jobs 3 --outdir periodic_grid_reproduced
python periodic_grid_plots.py --histdir periodic_grid_reproduced
python periodic_checks.py
```

The checks compare cyclic solves and FFT constants with independent dense
calculations, verify the weak pairing and midpoint adjoint identities, and
compare Howard iteration with exhaustive policy enumeration on a small grid.
They also verify an explicitly feasible balanced-policy comparison for each
assembled resolution. The `kappa=0` runs preserve zero sine moment and zero
exploitability to roundoff. The equilibrium diagnostic uses the same periodic
chain for forward and backward implicit-Euler solves. The finite-chain and
periodic spatial hypotheses are satisfied, while exact continuous-time
certificates and continuum convergence still require temporal and spatial
error estimates.

`periodic_fourier.py` is a separate direct-generator implementation. It uses
normalized real Fourier tests and diagonal `H^3` Gram weights, with a fixed
128-node periodic quadrature grid and 50 time intervals. Saved runs for
`M=4,8,16` are in `periodic_fourier_histories/`, including cross-mode residual
checks. Reproduction commands are:

```sh
python periodic_fourier.py --modes 4 --Nx 128 --Nt 50 --K 20000 --step 0.25 --kappa 1
python periodic_fourier.py --modes 8 --Nx 128 --Nt 50 --K 20000 --step 0.25 --kappa 1
python periodic_fourier.py --modes 16 --Nx 128 --Nt 50 --K 20000 --step 0.25 --kappa 1
python periodic_fourier_plots.py
```

A fixed Fourier test space enforces only its moment equations. The saved
cross-mode checks show why its own small residual should not be interpreted
as full forward-equation feasibility. The periodic figures are
`periodic_grid_g.png`, `periodic_grid_0.png`, `periodic_grid_moment.png`, and
`periodic_fourier.png` in `../plots/`. The scripts below retain the earlier
whole-line and reflecting-grid experiments for reproduction of the older draft.

## Finite-state congestion game

`finite_state_rates.py` runs primal--dual KL Mirror-Prox on the five-state,
three-action congestion game with `H = R^E` and the analytic time-Riesz formula.
The dual iterate, its time derivative, and its terminal value are propagated
separately, so no finite difference is used for the dual time derivative.
Time integrals are evaluated by the midpoint rule on 100 intervals.

The script computes the weak feasibility residual, the re-simulation
projection using matrix exponentials, and the exploitability of the projected
average using an RK4 solution of the finite-state HJB system. It writes
`finite_state_history.csv` and `finite_state_result.npz`.

Run:

```sh
python finite_state_rates.py
```

## Archived whole-line Hermite--Galerkin experiment

`diffusion_exact_time.py G_WEIGHT TAG KMAX STEP` runs the two-action diffusion
with the nonlocal `tanh` coupling and terminal cost
`-G_WEIGHT * tanh(y)`. It uses a 16-function Hermite test space and 80
Gauss--Hermite primal nodes. The dual time derivative is represented exactly
in the finite-coordinate formulation; time integrals use 50 midpoint
intervals and the spatial integrals use Gauss--Hermite quadrature.

The script also computes an independent finite-difference fictitious-play
benchmark for the HJB--Fokker--Planck system and evaluates the exploitability
of the re-simulated projected average. The two paper runs are:

```sh
python diffusion_exact_time.py 1.0 diffusion_g 50000 0.25
python diffusion_exact_time.py 0.0 diffusion_0 50000 0.25
```

They write `<TAG>_history.csv` and `<TAG>_result.npz`. The `diffusion_g.png`
curves are included in the paper's Hermite--Galerkin subsection as an
alternative to the Markov-chain implementation. This is a fixed finite-test
illustration, not a numerical test of the continuum inexact-step error
bounds. The remaining Hermite figures are retained as supplementary outputs.
The independent finite-difference benchmark stored in `diffusion_g_result.npz`
is still used for the grid-moment comparison.

## Archived reflecting-grid experiment

`diffusion_grid.py` replaces the controlled diffusion on `[-6,6]` by the
Kushner--Dupuis Markov chain and compares the Euclidean dual metric with the
discrete Sobolev metric `H_h^3`. The paper uses
`Nx` in `{121, 241, 481}`, `Nt` in `{25, 50, 100}`, `K = 50000`, and the
fixed steps `lambda = tau = 0.25`. Per-run histories, summaries, and saved
states for the original sweep are preserved in `grid_histories/`. The paper's
corrected sweep is in `grid_histories_normalized/`: feedback is normalized even
in subnormal-mass tails, all final diagnostics are evaluated at `K = 50000`,
and the final averaged state-action measure is saved in each NPZ.

From this directory, reproduce the complete 42-case sweep into a fresh output
directory and collect its figures with:

```sh
python rerun_normalized_grid_sweep.py --K 50000 --jobs 8 --outdir grid_histories_reproduced
python diffusion_grid_plots.py --histdir grid_histories_reproduced
```

The driver refuses to overwrite an existing result and records the exact commands
and source hashes in `rerun_manifest.json`. To regenerate the paper's figures
without rerunning experiments, use `python diffusion_grid_plots.py --histdir
grid_histories_normalized`. The corrected consolidated table is
`grid_histories_normalized/grid_sweep_summary.csv`; the original table remains
`grid_sweep_summary.csv`.

These are reflecting-grid, midpoint-quadrature experiments with a sampled
Gaussian reference. They do not verify the periodic mesh theorem or its
uniform-reference hypothesis verbatim. Reported residuals and implicit-Euler
exploitabilities are numerical diagnostics, not continuum convergence
certificates. Further implementation details and the original sweep record
are in `grid_report.md`.

## Figures

`make_plots.py` regenerates the finite-state and Hermite figures in
`../plots/` from the saved outputs:

```sh
python make_plots.py
```

`diffusion_grid_plots.py` regenerates `diffusion_grid_g.png`,
`diffusion_grid_0.png`, and the supplementary moment comparison
`diffusion_grid_moment.png` from the saved grid sweep.

Every fixed grid run is a finite-state instance of the theory. Its
interpretation as an approximation of the diffusion is conditional on the
additional tightness and consistency needed to pass to the continuum limit.
