Primal–Dual KL Mirror-Prox for Dynamic Mean Field Games
Code and saved results for the numerical experiments in
> E. Bayraktar, I. Ekren, L. Vy and Z. Zhang,
> \*Primal–Dual Kullback–Leibler Mirror-Prox for Dynamic Mean Field Games\*,
> arXiv:XXXX.XXXXX (2026).
The method computes finite-horizon mean field equilibria in the
occupation-measure formulation of the game. Each iteration tilts the
occupation measures exponentially by the current HJB residuals and corrects
the dual variable with a linear Riesz map, so no HJB equation or
best-response problem is solved. Code for the static version of the method
is in kl-mirror-prox-mfe.
Contents
Paper	Experiment	Scripts	Saved results
§7.1	Finite-state congestion game	`finite\_state\_rates.py`, `make\_plots.py`	`finite\_state\_history.csv`, `finite\_state\_result.npz`
§7.2	Periodic controlled diffusion on Markov-chain grids	`periodic\_diffusion\_grid.py`, `run\_periodic\_sweep.py`, `periodic\_grid\_plots.py`, `periodic\_checks.py`	`periodic\_grid\_histories/`
§7.3	The same game with a Fourier test space	`periodic\_fourier.py`, `periodic\_fourier\_plots.py`	`periodic\_fourier\_histories/`
The figures in the paper are in `plots/`.
Installation
```sh
git clone https://github.com/Lethargy/pd-kl-mirror-prox-mfe.git
cd pd-kl-mirror-prox-mfe
python -m pip install -r requirements.txt
```
The code needs Python 3 with NumPy, SciPy and Matplotlib. It was tested with
Python X.Y, NumPy X.Y, SciPy X.Y and Matplotlib X.Y.
Regenerating the figures from the saved results
```sh
python make\_plots.py
python periodic\_grid\_plots.py --histdir periodic\_grid\_histories
python periodic\_fourier\_plots.py
```
Rerunning the experiments
§7.1 Finite-state congestion game
Five states, three actions (left, stay, right), horizon T = 1, and the dual
space `H = R^E` with the Euclidean inner product. The dual iterate, its time
derivative and its terminal value are stored separately, so no finite
difference in time is needed; time integrals use the midpoint rule on 100
intervals. The script records the forward-equation residual of the averaged
iterate, the size of the feasibility correction (the forward equation is
re-solved under the extracted feedback with matrix exponentials), and the
exploitability (Stampacchia gap) of the corrected average, computed from an
RK4 solution of the finite-state HJB system.
```sh
python finite\_state\_rates.py
```
This writes `finite\_state\_history.csv` and `finite\_state\_result.npz`.
Figures: `plots/finite\_state\_rates.png`, `plots/finite\_state\_equilibrium.png`.
§7.2 Periodic controlled diffusion
Players move on the circle R/2πZ by dX = a dt + √2 dW with a ∈ {−1, +1} and
X₀ = 0. The running cost is f(x, μ) = sin(x) ∫ sin(y) μ(dy) and the terminal
cost is g(x) = −κ sin(x). The Markov chain uses periodic Kushner–Dupuis
rates, and the Sobolev Riesz map is applied by FFT. The reference measures
are uniform in state and action; they are strictly positive and need not
satisfy the initial condition.
`run\_periodic\_sweep.py` runs 12 cases: (Nx, Nt) ∈ {(64, 25), (128, 50),
(256, 100)}, κ ∈ {0, 1}, and either the discrete Sobolev metric `H\_h^3` or
the Euclidean metric for the dual variable. Every run uses 20,000 iterations,
with step 0.25 for the Sobolev metric and 0.99/(L_C + L_A) for the Euclidean
metric.
```sh
python run\_periodic\_sweep.py --K 20000 --jobs 3 --outdir periodic\_grid\_reproduced
python periodic\_grid\_plots.py --histdir periodic\_grid\_reproduced
python periodic\_checks.py
```
The driver does not overwrite existing results, and its manifest records the
exact commands and a hash of the source. The values in the table of §7.2 are
in `periodic\_grid\_histories/periodic\_sweep\_summary.csv`.
Figures: `plots/periodic\_grid\_g.png`, `plots/periodic\_grid\_0.png`,
`plots/periodic\_grid\_moment.png`.
`periodic\_checks.py` compares the periodic solves and FFT constants with dense
computations, checks the weak pairing and midpoint adjoint identities,
compares Howard iteration with exhaustive policy enumeration on a small grid,
and checks a feasible balanced-policy comparison at each resolution.
§7.3 Fourier test space
`periodic\_fourier.py` imposes the forward equation of the §7.2 game against M
real Fourier modes with diagonal `H^3` Gram weights, on a fixed 128-node
quadrature grid with 50 time intervals.
```sh
for M in 4 8 16; do
  python periodic\_fourier.py --modes $M --Nx 128 --Nt 50 --K 20000 --step 0.25 --kappa 1
done
python periodic\_fourier\_plots.py
```
Figure: `plots/periodic\_fourier.png`.
What the experiments do and do not show
Each grid run is a finite-state game of the kind covered by the paper's
theory, up to the midpoint discretization in time. Statements about the
diffusion itself, or about the exact continuous-time iteration, would need
temporal and spatial error estimates that these experiments do not provide.
A fixed Fourier test space enforces only its own moment equations. The
cross-mode checks in `periodic\_fourier\_histories/` show that a small
residual in that space does not imply full feasibility.
For κ = 0 the balanced policy is an exact equilibrium, and the runs keep the
sine moment and the exploitability at roundoff level.
Citation
```bibtex
@article{BayraktarEkrenVyZhang2026,
  title   = {Primal--Dual {K}ullback--{L}eibler Mirror-Prox for Dynamic Mean Field Games},
  author  = {Bayraktar, Erhan and Ekren, Ibrahim and Vy, Lu and Zhang, Ziqing},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```
License
MIT. See `LICENSE`.
