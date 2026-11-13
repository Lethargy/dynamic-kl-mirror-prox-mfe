"""Plot saved periodic diffusion experiments without importing old benchmarks.

Run after periodic_diffusion_grid.py. All plotted diagnostics, grid sizes,
step sizes, and slope-guide anchors are read from periodic_grid_histories.
The two residual norms belong to different dual metrics and need not agree.
The K^{-1} lines are reference guides, not fitted rates or certificates for
the midpoint time quadrature. Roundoff-zero diagnostics use linear axes.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bw_style as bw


HERE = Path(__file__).resolve().parent


def read_runs(directory: Path) -> list[dict]:
    runs = []
    for path in sorted(directory.glob("*_summary.json")):
        run = json.loads(path.read_text())
        tag = run.get("tag", path.name.removesuffix("_summary.json"))
        history_path = directory / f"{tag}_history.csv"
        if not history_path.exists():
            raise FileNotFoundError(history_path)
        run["tag"] = tag
        run["_history"] = np.atleast_1d(
            np.genfromtxt(history_path, delimiter=",", names=True)
        )
        runs.append(run)
    if not runs:
        raise FileNotFoundError(f"No periodic experiment summaries in {directory}")
    return runs


def make_styles(runs: list[dict]):
    grids = sorted({(int(r["Nx"]), int(r["Nt"])) for r in runs})
    index = {grid: i for i, grid in enumerate(grids)}

    def style(run: dict, n: int, *, steps: bool = True) -> dict:
        grid = (int(run["Nx"]), int(run["Nt"]))
        i = index[grid]
        result = bw.style(i, n, frac=0.15, ms=4.0, lw=1.5)
        result["label"] = rf"$N_x={grid[0]},\ N_t={grid[1]}$"
        if steps:
            lam, tau = float(run["lam"]), float(run["tau"])
            if np.isclose(lam, tau):
                result["label"] += rf", $\eta={lam:.2g}$"
            else:
                result["label"] += rf", $\lambda={lam:.2g},\tau={tau:.2g}$"
        return result

    return style


def positive_curve(ax, x, y, **style):
    x, y = np.asarray(x), np.asarray(y)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if np.any(valid):
        ax.loglog(x[valid], y[valid], **style)


def inverse_iteration_guide(ax, records: list[tuple[np.ndarray, np.ndarray]]):
    """Anchor a slope -1 guide to the observed final third of the data."""
    usable = []
    for x, y in records:
        good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 1e-13)
        if np.count_nonzero(good) >= 4:
            usable.append((x[good], y[good]))
    if not usable:
        return
    x, y = usable[0]
    start = max(0, len(x) * 2 // 3)
    anchor = float(np.median(x[start:] * y[start:]))
    xmin = max(float(x[0]), float(x[-1]) / 12.0)
    xmax = float(x[-1])
    if xmax <= xmin or not np.isfinite(anchor) or anchor <= 0:
        return
    xx = np.geomspace(xmin, xmax, 24)
    ax.plot(xx, 0.55 * anchor / xx, ":", color="0.45", lw=1.0,
            label=r"$K^{-1}$ guide", zorder=1)


def diagnostic_figure(runs: list[dict], kappa: float, plots: Path, style):
    selected = [r for r in runs if np.isclose(float(r["kappa"]), kappa)
                and r["metric"] in ("S3", "E")]
    if not selected:
        return
    metrics = [metric for metric in ("S3", "E")
               if any(r["metric"] == metric for r in selected)]
    fig, axes = plt.subplots(len(metrics), 3, figsize=(12.0, 3.7 * len(metrics)),
                             squeeze=False)
    names = ("residual_avg", "max_abs_moment", "exploitability")
    titles = ("Averaged feasibility residual", "Maximum population moment",
              "Stampacchia gap")
    zero_case = np.isclose(kappa, 0.0)
    for row, metric in enumerate(metrics):
        group = sorted((r for r in selected if r["metric"] == metric),
                       key=lambda r: (r["Nx"], r["Nt"], r["lam"]))
        for col, field in enumerate(names):
            records = []
            for run in group:
                history = run["_history"]
                x, y = history["K"], history[field]
                properties = style(run, len(x))
                if col == 1 or (zero_case and col == 2):
                    valid = np.isfinite(y) & np.isfinite(x) & (x > 0)
                    axes[row, col].semilogx(x[valid], y[valid], **properties)
                else:
                    positive_curve(axes[row, col], x, y, **properties)
                records.append((x, y))
            if metric == "S3" and (col == 0 or (col == 2 and not zero_case)):
                inverse_iteration_guide(axes[row, col], records)
            axes[row, col].set_title(titles[col], fontsize=10.5)
            axes[row, col].set_xlabel("Proximal iteration $K$")
            if zero_case and col > 0:
                values = np.concatenate([ys[np.isfinite(ys)] for _, ys in records])
                peak = float(np.max(np.abs(values))) if values.size else 0.0
                if peak < 1e-10:
                    axes[row, col].text(0.04, 0.92,
                        rf"Symmetry check: max $|\cdot|={peak:.1e}$",
                        transform=axes[row, col].transAxes, fontsize=8.4, va="top",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
                    axes[row, col].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            if col == 0:
                label = (r"Sobolev $H_h^3$" if metric == "S3"
                         else r"Euclidean $\mathbb{R}^{E_h}$")
                axes[row, col].set_ylabel(label + "\nresidual in its own dual norm")
            elif col == 1:
                axes[row, col].set_ylabel(r"$\max_t |\int\sin(x)\,d\bar\mu_K(t)|$")
            else:
                axes[row, col].set_ylabel("Same-scheme cost difference")
        handles, labels = axes[row, 0].get_legend_handles_labels()
        axes[row, 0].legend(handles, labels, loc="best", fontsize=7.4,
                             framealpha=0.9, handlelength=2.3)
        max_theta = max(float(r["theta"]) for r in group)
        axes[row, 1].text(0.98, 0.04, rf"$\max\theta={max_theta:.3g}$",
                         transform=axes[row, 1].transAxes, ha="right", fontsize=8.5)
    fig.suptitle(rf"Periodic controlled diffusion: $\varkappa={kappa:g}$", fontsize=13)
    note = ("Zero moment and zero cost differences are symmetry checks; no rate is inferred."
            if zero_case else
            "Step sizes are shown for every grid. Dotted lines indicate a reference slope only.")
    fig.text(0.5, 0.01, note, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97), h_pad=2.0, w_pad=1.3)
    filename = "periodic_grid_0.png" if zero_case else "periodic_grid_g.png"
    destination = plots / filename
    bw.save(fig, destination)
    print("wrote", destination)


def moment_figure(runs: list[dict], hist: Path, plots: Path, style):
    selected = sorted((r for r in runs if np.isclose(float(r["kappa"]), 1.0)
                       and r["metric"] == "S3"),
                      key=lambda r: (r["Nx"], r["Nt"]))
    if not selected:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.25))
    for run in selected:
        with np.load(hist / f"{run['tag']}_result.npz") as result:
            times, moments = result["tm"], result["m_avg"]
            properties = style(run, len(times), steps=False)
            axes[0].plot(times, moments, **properties)
            key = "xg" if "xg" in result else "yg"
            grid = result[key]
            if "nu_hat" in result:
                masses = result["nu_hat"]
                if masses.ndim > 1:
                    masses = masses[-1]
                label = "Projected terminal density"
            else:
                masses = result["avg_nu"]
                label = "Averaged terminal density"
            order = np.argsort(grid)
            grid, density = grid[order], masses[order] / float(run["h"])
            # Add one periodic copy of the first node to close the circle.
            period = float(run["h"]) * int(run["Nx"])
            xx = np.r_[grid, grid[0] + period]
            yy = np.r_[density, density[0]]
            axes[1].plot(xx, yy, **style(run, len(xx), steps=False))
    axes[0].set(xlabel="Physical time $t$", ylabel=r"$\int\sin(x)\,d\bar\mu_K(t)$",
                title="Population moment across periodic grids")
    axes[1].set(xlabel="State $x$ on the circle", ylabel="Density (mass / grid width)",
                title=label)
    axes[0].legend(loc="best", fontsize=8.8)
    axes[1].legend(loc="best", fontsize=8.8)
    iterations = sorted({int(r["K"]) for r in selected})
    suffix = rf", $K={iterations[0]:,}$" if len(iterations) == 1 else ""
    fig.suptitle(r"Periodic game, $\varkappa=1$, Sobolev $H_h^3$" + suffix, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    destination = plots / "periodic_grid_moment.png"
    bw.save(fig, destination)
    print("wrote", destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histdir", type=Path, default=HERE / "periodic_grid_histories")
    parser.add_argument("--plotdir", type=Path, default=HERE.parent / "plots")
    args = parser.parse_args()
    hist, plots = args.histdir.resolve(), args.plotdir.resolve()
    plots.mkdir(parents=True, exist_ok=True)
    runs = read_runs(hist)
    bw.setup(font=11)
    style = make_styles(runs)
    columns = ["tag", "kappa", "metric", "Nx", "Nt", "h", "lam", "tau", "theta",
               "K", "stable", "res_avg", "max_abs_moment", "exploitability"]
    with (hist / "periodic_sweep_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for run in sorted(runs, key=lambda r: (-r["kappa"], r["metric"], r["Nx"], r["Nt"])):
            writer.writerow({column: run.get(column, "") for column in columns})
    diagnostic_figure(runs, 1.0, plots, style)
    diagnostic_figure(runs, 0.0, plots, style)
    moment_figure(runs, hist, plots, style)


if __name__ == "__main__":
    main()
