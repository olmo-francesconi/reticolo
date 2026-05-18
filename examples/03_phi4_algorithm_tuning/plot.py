"""Compare HMC vs Metropolis across two regimes:
    - easy     (κ ≪ κ_c, deep in the symmetric phase, fast dynamics)
    - critical (κ ≈ κ_c, slow dynamics, where HMC's advantage shows up)

Each plot has the two scenarios side-by-side. Outputs (in this folder):

  rho_semilog.png        ρ(t) on semi-log Y axis, per scenario.
  hmc_step_scan.png      τ_int(Σφ²) and acceptance vs step size ε = τ/n_md,
                         one curve per τ, per integrator, per scenario.
  pareto.png             cost-per-independent-sample vs acceptance for every
                         HMC run, with the empirical Pareto frontier per
                         integrator drawn explicitly so you can see whether
                         the picked "best" is robust or just the corner of a
                         flat plateau.
  metropolis_scan.png    τ_int(σ) and accept(σ) per scenario.
  summary_bar.png        best-per-algorithm independent samples / wall second,
                         per scenario.

τ_int is converted to wall seconds via the *algorithm-only* per-update cost
(observable measurement is timed separately and excluded).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_common"))
from autocorr import autocorrelation  # noqa: E402

RESULTS_ROOT = HERE / "results"

ALGO_ORDER = ["metropolis", "hmc_leapfrog", "hmc_omelyan2", "hmc_omelyan4"]
ALGO_SHORT = {
    "metropolis":   "Metropolis",
    "hmc_leapfrog": "Leapfrog",
    "hmc_omelyan2": "Omelyan2",
    "hmc_omelyan4": "Omelyan4",
}
ALGO_COLOR = {
    "metropolis":   "#4477AA",
    "hmc_leapfrog": "#EE6677",
    "hmc_omelyan2": "#228833",
    "hmc_omelyan4": "#CCBB44",
}


def load_summary(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k, v in list(r.items()):
                if v == "":
                    r[k] = None
                else:
                    try:
                        r[k] = float(v) if "." in v or "e" in v.lower() else int(v)
                    except (ValueError, TypeError):
                        pass
            rows.append(r)
    return rows


def file_for(scenario_dir: Path, row: dict) -> Path:
    if row["algo"] == "metropolis":
        return scenario_dir / f"metropolis_sigma{row['sigma']:06.3f}.h5"
    return scenario_dir / (
        f"{row['algo']}_tau{float(row['tau']):05.2f}_nmd{int(row['n_md']):03d}.h5"
    )


def best_per_algo(rows, *, by="tau_msq_sec", accept_min=0.5):
    out = {}
    for algo in ALGO_ORDER:
        cands = [
            r for r in rows
            if r["algo"] == algo
            and r.get("accept", 0.0) >= accept_min
            and r.get(by) is not None
            and isinstance(r.get(by), float)
            and np.isfinite(r.get(by, np.nan))
        ]
        if cands:
            out[algo] = min(cands, key=lambda r: r[by])
    return out


def discover_scenarios() -> list[tuple[str, Path, list[dict]]]:
    out = []
    for d in sorted(RESULTS_ROOT.iterdir()):
        summary = d / "summary.csv"
        if d.is_dir() and summary.exists():
            out.append((d.name, d, load_summary(summary)))
    return out


# -----------------------------------------------------------------------------
# Pareto frontier (per-algorithm lower envelope on cost-vs-accept)
# -----------------------------------------------------------------------------
def pareto_frontier(accs: np.ndarray, costs: np.ndarray) -> np.ndarray:
    """Return the indices of points on the lower envelope of (accept, cost),
    sorted by acceptance. A point is on the frontier if no other point has
    both higher accept AND lower cost."""
    order = np.argsort(accs)
    a_sorted = accs[order]
    c_sorted = costs[order]
    frontier = []
    min_cost = np.inf
    # Walk from highest accept down — keep a running min of cost.
    for i in range(len(a_sorted) - 1, -1, -1):
        if c_sorted[i] < min_cost:
            min_cost = c_sorted[i]
            frontier.append(order[i])
    return np.array(sorted(frontier, key=lambda k: accs[k]))


# -----------------------------------------------------------------------------
# Plot 1: ρ(t) semilog
# -----------------------------------------------------------------------------
def plot_rho_semilog(scenarios):
    n = len(scenarios)
    fig, axes = plt.subplots(n, 2, figsize=(13, 4.4 * n), squeeze=False, sharey=True)
    for row_i, (name, sdir, rows) in enumerate(scenarios):
        best = best_per_algo(rows)
        for col, (obs, title) in enumerate([("s", "S"),
                                             ("mean_sq", r"$\Sigma\phi^2$")]):
            ax = axes[row_i, col]
            for algo, r in best.items():
                with h5py.File(file_for(sdir, r), "r") as f:
                    x = f[f"/prod/obs/{obs}"][...]
                rho = autocorrelation(x)
                lag_s = np.arange(len(rho)) * r["algo_per_update"]
                tau_sec = r["tau_S_sec"] if obs == "s" else r["tau_msq_sec"]
                stop = min(len(rho),
                           max(250, int(np.ceil(8 * tau_sec / r["algo_per_update"]))))
                if r["algo"] == "metropolis":
                    tag = f"σ={r['sigma']:.2f}"
                else:
                    tag = f"τ={r['tau']:.1f}, n_md={int(r['n_md'])}"
                ax.semilogy(
                    lag_s[:stop], np.clip(rho[:stop], 1e-3, None),
                    color=ALGO_COLOR[algo], lw=1.2,
                    marker="o", markersize=2, markevery=max(1, stop // 40),
                    label=f"{ALGO_SHORT[algo]} ({tag})",
                )
            ax.axhline(np.e**-1, color="k", lw=0.5, ls=":", alpha=0.55,
                       label=r"$\rho = e^{-1}$")
            ax.set_xscale("log")
            ax.set_xlabel("lag [s of algorithm wall time]")
            if col == 0:
                ax.set_ylabel(r"$\rho(t)$")
            ax.set_ylim(1e-3, 1.1)
            ax.set_title(f"{name} — {title}", fontsize=11)
            ax.grid(True, which="both", alpha=0.3, lw=0.5)
            ax.legend(fontsize=7.5, framealpha=0.9, loc="upper right")
    fig.suptitle("Autocorrelation function — best run per algorithm, both scenarios")
    fig.tight_layout()
    fig.savefig(HERE / "rho_semilog.png", dpi=140)
    plt.close(fig)
    print("wrote rho_semilog.png")


# -----------------------------------------------------------------------------
# Plot 2: HMC step-size scan
# -----------------------------------------------------------------------------
def plot_hmc_step_scan(scenarios):
    hmcs = ["hmc_leapfrog", "hmc_omelyan2", "hmc_omelyan4"]
    n = len(scenarios)
    fig, axes = plt.subplots(2 * n, 3, figsize=(15.5, 8 * n), sharex="col")
    taus_all = sorted({r["tau"] for _, _, rows in scenarios for r in rows
                       if r["algo"] in hmcs and r["tau"] is not None})
    cmap = plt.get_cmap("viridis")
    tau_colors = {t: cmap(i / max(1, len(taus_all) - 1))
                  for i, t in enumerate(taus_all)}

    for row_i, (name, _, rows) in enumerate(scenarios):
        for col, algo in enumerate(hmcs):
            ax_top = axes[2 * row_i + 0, col]
            ax_bot = axes[2 * row_i + 1, col]
            for tau in taus_all:
                cands = [r for r in rows
                         if r["algo"] == algo and r["tau"] == tau
                         and r.get("tau_msq_sec") is not None
                         and isinstance(r["tau_msq_sec"], float)
                         and np.isfinite(r["tau_msq_sec"])
                         and r["accept"] > 0.05]
                if not cands:
                    continue
                cands.sort(key=lambda r: r["n_md"])
                eps = np.array([tau / r["n_md"] for r in cands])
                tau_ms = 1000.0 * np.array([r["tau_msq_sec"] for r in cands])
                tau_ms_err = 1000.0 * np.array([r["tau_msq_sec_err"] for r in cands])
                acc = np.array([r["accept"] for r in cands])
                color = tau_colors[tau]
                ax_top.errorbar(eps, tau_ms, yerr=tau_ms_err, fmt="o-",
                                color=color, lw=1.0, ms=4, capsize=2,
                                label=f"τ={tau:g}")
                ax_bot.plot(eps, acc, "o-", color=color, lw=1.0, ms=4)
            ax_top.set_xscale("log")
            ax_top.set_yscale("log")
            ax_top.grid(True, which="both", alpha=0.3, lw=0.5)
            ax_top.set_title(f"{name} — {ALGO_SHORT[algo]}")
            if col == 0:
                ax_top.set_ylabel(r"$\tau_{\rm int}(\Sigma\phi^2)$ [ms]")
            ax_top.legend(fontsize=7, ncols=2, framealpha=0.85)
            ax_bot.set_xscale("log")
            ax_bot.set_ylim(-0.05, 1.05)
            ax_bot.axhline(0.65, color="k", lw=0.5, ls=":", alpha=0.5)
            ax_bot.grid(True, which="both", alpha=0.3, lw=0.5)
            ax_bot.set_xlabel(r"step size $\epsilon = \tau / n_{md}$")
            if col == 0:
                ax_bot.set_ylabel("acceptance")
    fig.suptitle(r"HMC tuning curves per scenario: $\tau_{\rm int}(\Sigma\phi^2)$ "
                 "(top) and acceptance (bottom) vs step size")
    fig.tight_layout()
    fig.savefig(HERE / "hmc_step_scan.png", dpi=140)
    plt.close(fig)
    print("wrote hmc_step_scan.png")


# -----------------------------------------------------------------------------
# Plot 3: Pareto frontier on cost-vs-accept
# -----------------------------------------------------------------------------
def plot_pareto(scenarios):
    hmcs = ["hmc_leapfrog", "hmc_omelyan2", "hmc_omelyan4"]
    n = len(scenarios)
    fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 6.0), squeeze=False)
    for col, (name, _, rows) in enumerate(scenarios):
        ax = axes[0, col]
        for algo in hmcs:
            cands = [r for r in rows
                     if r["algo"] == algo
                     and r.get("tau_msq_sec") is not None
                     and isinstance(r["tau_msq_sec"], float)
                     and np.isfinite(r["tau_msq_sec"])
                     and r["accept"] > 0.05]
            if not cands:
                continue
            accs = np.array([r["accept"] for r in cands])
            costs = 1000.0 * np.array([2.0 * r["tau_msq_sec"] for r in cands])
            errs  = 1000.0 * np.array([2.0 * r["tau_msq_sec_err"] for r in cands])
            n_mds = np.array([r["n_md"] for r in cands])
            ax.scatter(accs, costs, c=ALGO_COLOR[algo], s=15 + 4 * n_mds,
                       alpha=0.50, edgecolors="k", linewidths=0.3,
                       label=ALGO_SHORT[algo])
            # Pareto frontier
            idx = pareto_frontier(accs, costs)
            ax.plot(accs[idx], costs[idx], color=ALGO_COLOR[algo], lw=1.6,
                    alpha=0.95, zorder=5)
        # Metropolis best as a band
        metr_best = best_per_algo(rows, by="tau_msq_sec").get("metropolis")
        if metr_best is not None:
            c = 1000.0 * 2.0 * metr_best["tau_msq_sec"]
            ce = 1000.0 * 2.0 * metr_best["tau_msq_sec_err"]
            ax.axhline(c, color=ALGO_COLOR["metropolis"], lw=1.4, ls="--",
                       label=f"Metropolis best (σ={metr_best['sigma']:.2f})")
            ax.axhspan(c - ce, c + ce, color=ALGO_COLOR["metropolis"], alpha=0.10)
        ax.set_yscale("log")
        ax.set_xlabel("acceptance rate")
        if col == 0:
            ax.set_ylabel(r"wall ms per independent $\Sigma\phi^2$ sample")
        ax.set_xlim(0, 1.0)
        ax.grid(True, which="both", alpha=0.3, lw=0.5)
        ax.legend(fontsize=8, loc="upper left",
                  title=f"{name}   (size ∝ n_md)", title_fontsize=8.5)
        ax.set_title(f"{name}: Pareto frontier per integrator")
    fig.suptitle("Cost per independent sample vs acceptance — the Pareto view")
    fig.tight_layout()
    fig.savefig(HERE / "pareto.png", dpi=140)
    plt.close(fig)
    print("wrote pareto.png")


# -----------------------------------------------------------------------------
# Plot 4: Metropolis scan
# -----------------------------------------------------------------------------
def plot_metropolis_scan(scenarios):
    n = len(scenarios)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4.5 * n), squeeze=False)
    for row_i, (name, _, rows) in enumerate(scenarios):
        metr = sorted([r for r in rows if r["algo"] == "metropolis"],
                      key=lambda r: r["sigma"])
        if not metr:
            continue
        sigma = np.array([r["sigma"] for r in metr])

        ax = axes[row_i, 0]
        for obs, label, marker in (
            ("S",   r"$\tau_{\rm int}$(S) [ms]",          "o"),
            ("msq", r"$\tau_{\rm int}(\Sigma\phi^2)$ [ms]", "s"),
        ):
            y  = 1000.0 * np.array([r[f"tau_{obs}_sec"] for r in metr])
            ey = 1000.0 * np.array([r[f"tau_{obs}_sec_err"] for r in metr])
            ax.errorbar(sigma, y, yerr=ey, marker=marker, ms=5, lw=0.8,
                        capsize=2, label=label)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"proposal width $\sigma$")
        ax.set_ylabel(r"$\tau_{\rm int}$ [ms]")
        ax.grid(True, which="both", alpha=0.3, lw=0.5)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.set_title(f"{name} — autocorrelation")

        ax = axes[row_i, 1]
        acc = np.array([r["accept"] for r in metr])
        ax.plot(sigma, acc, "o-", color="k", lw=0.8, ms=5)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.0)
        ax.set_xlabel(r"proposal width $\sigma$")
        ax.set_ylabel("acceptance")
        ax.axhline(0.5, color="r", lw=0.5, ls=":", alpha=0.5)
        ax.grid(True, which="both", alpha=0.3, lw=0.5)
        ax.set_title(f"{name} — acceptance")
    fig.suptitle("Metropolis tuning per scenario")
    fig.tight_layout()
    fig.savefig(HERE / "metropolis_scan.png", dpi=140)
    plt.close(fig)
    print("wrote metropolis_scan.png")


# -----------------------------------------------------------------------------
# Plot 5: summary bar per scenario
# -----------------------------------------------------------------------------
def plot_summary(scenarios):
    n = len(scenarios)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4.5 * n), squeeze=False)
    for row_i, (name, _, rows) in enumerate(scenarios):
        best = best_per_algo(rows)
        for col, obs in enumerate(["S", "msq"]):
            ax = axes[row_i, col]
            keys, vals, errs, labels = [], [], [], []
            for algo in ALGO_ORDER:
                if algo not in best:
                    continue
                r = best[algo]
                tau_sec = r[f"tau_{obs}_sec"]
                tau_err = r[f"tau_{obs}_sec_err"]
                ess = 1.0 / (2.0 * tau_sec)
                ess_err = ess * (tau_err / tau_sec)
                keys.append(algo)
                vals.append(ess)
                errs.append(ess_err)
                if r["algo"] == "metropolis":
                    lbl = f"{ALGO_SHORT[algo]}\nσ={r['sigma']:.2f}"
                else:
                    lbl = f"{ALGO_SHORT[algo]}\nτ={r['tau']:.1f}, n_md={int(r['n_md'])}"
                labels.append(lbl)
            bars = ax.bar(range(len(keys)), vals, yerr=errs,
                          color=[ALGO_COLOR[a] for a in keys],
                          capsize=4, edgecolor="k", lw=0.4)
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel("independent samples / s")
            obs_label = "S" if obs == "S" else r"$\Sigma\phi^2$"
            ax.set_title(f"{name} — {obs_label}")
            ax.grid(True, axis="y", alpha=0.3, lw=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v * 1.04,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Best per algorithm: independent samples per wall-second")
    fig.tight_layout()
    fig.savefig(HERE / "summary_bar.png", dpi=140)
    plt.close(fig)
    print("wrote summary_bar.png")


def main() -> None:
    scenarios = discover_scenarios()
    if not scenarios:
        sys.exit(f"no scenario summaries found in {RESULTS_ROOT}; run analyze.py first")
    print(f"found scenarios: {[s[0] for s in scenarios]}")

    plt.rcParams.update({
        "figure.dpi": 100,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })

    plot_rho_semilog(scenarios)
    plot_hmc_step_scan(scenarios)
    plot_pareto(scenarios)
    plot_metropolis_scan(scenarios)
    plot_summary(scenarios)


if __name__ == "__main__":
    main()
