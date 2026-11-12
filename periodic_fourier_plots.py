"""Plot the saved direct Fourier-Galerkin periodic diffusion experiment.

Every curve comes from periodic_fourier_histories. Residuals are evaluated
against the saved finite Fourier test space with midpoint time assembly;
they are not full forward-equation residuals or exploitability certificates.
No result from a jump-chain or a whole-line diffusion is used as a benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bw_style as bw
from periodic_grid_plots import inverse_iteration_guide, positive_curve


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histdir", type=Path, default=HERE / "periodic_fourier_histories")
    parser.add_argument("--plotdir", type=Path, default=HERE.parent / "plots")
    parser.add_argument("--kappa", type=float, default=1.0)
    args = parser.parse_args()
    hist, plots = args.histdir.resolve(), args.plotdir.resolve()
    selected = []
    for path in sorted(hist.glob("*_summary.json")):
        record = json.loads(path.read_text())
        if np.isclose(float(record["parameters"]["kappa"]), args.kappa):
            selected.append(record)
    selected.sort(key=lambda r: (r["parameters"]["modes"], r["parameters"]["Nx"],
                                 r["parameters"]["Nt"]))
    if not selected:
        raise FileNotFoundError(f"No kappa={args.kappa:g} Fourier runs in {hist}")

    bw.setup(font=11)
    fig, axes = plt.subplots(1, 3, figsize=(12.1, 3.9))
    residual_records = []
    for i, record in enumerate(selected):
        params = record["parameters"]
        history = np.atleast_1d(np.genfromtxt(hist / record["files"]["history"],
                                             delimiter=",", names=True))
        count = len(history)
        properties = bw.style(i, count, frac=0.14, ms=4.0)
        properties["label"] = rf"$M={params['modes']}$ ({params['basis_dimension']} tests)"
        positive_curve(axes[0], history["K"], history["residual_avg"], **properties)
        axes[1].semilogx(history["K"], history["max_abs_moment"], **properties)
        residual_records.append((history["K"], history["residual_avg"]))
        with np.load(hist / record["files"]["results"]) as result:
            properties = bw.style(i, len(result["tm"]), frac=0.14, ms=4.0)
            properties["label"] = rf"$M={params['modes']}$"
            axes[2].plot(result["tm"], result["moment"], **properties)

    inverse_iteration_guide(axes[0], residual_records)
    axes[0].set(xlabel="Proximal iteration $K$", ylabel="Finite-test dual residual",
                title="Residual of the averaged iterate")
    axes[1].set(xlabel="Proximal iteration $K$",
                ylabel=r"$\max_t |\int\sin(x)\,d\bar\mu_K(t)|$",
                title="Population moment during iteration")
    axes[2].set(xlabel="Physical time $t$", ylabel=r"$\int\sin(x)\,d\bar\mu_K(t)$",
                title="Final averaged population moment")
    axes[0].legend(loc="best", fontsize=8.4)
    axes[2].legend(loc="best", fontsize=8.4)
    resolutions = {(r["parameters"]["Nx"], r["parameters"]["Nt"]) for r in selected}
    steps = {(r["parameters"]["lambda"], r["parameters"]["tau"]) for r in selected}
    subtitle = rf"Direct diffusion Fourier-Galerkin, $\varkappa={args.kappa:g}$"
    if len(resolutions) == 1:
        nx, nt = next(iter(resolutions))
        subtitle += rf", $N_x={nx},\ N_t={nt}$"
    if len(steps) == 1:
        lam, tau = next(iter(steps))
        subtitle += rf", $\lambda={lam:g},\ \tau={tau:g}$"
    fig.suptitle(subtitle, fontsize=12)
    fig.text(0.5, 0.01,
             "Direct differential generator; finite Fourier tests and midpoint time assembly. "
             "The dotted line is a slope guide.", ha="center", fontsize=8.8)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94), w_pad=1.7)
    plots.mkdir(parents=True, exist_ok=True)
    destination = plots / "periodic_fourier.png"
    bw.save(fig, destination)
    print("wrote", destination)


if __name__ == "__main__":
    main()
