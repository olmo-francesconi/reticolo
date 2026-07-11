#!/usr/bin/env python3
"""Reconstruct the density of states of the field amplitude Q = Σφ² from the
generalized-observable LLR run, and cross-check against a direct HMC histogram.

The LLR windows/adapts/exchanges on Q (not the action), so the converged
per-replica tilt is a_n = d ln ρ / dQ at each window centre E_n. Trapezoidal
integration of a_n reconstructs ln ρ(Q) over the whole ladder — including the
suppressed tails the plain HMC chain never visits.

Everything is plotted in per-site units ⟨φ²⟩ = Q/V (V read from /cfg@V):
  1. a_n adaptation history per replica (the orchestration converging).
  2. Converged tilt a_n vs ⟨φ²⟩; the zero crossing marks the DoS peak.
  3. ln ρ(⟨φ²⟩): LLR reconstruction vs direct histogram (aligned by a shift).
  4. ρ(⟨φ²⟩) peak-normalised — LLR reaching beyond the direct band.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

sys.path.insert(0, str(HERE.parent / "_common"))
from llr_reconstruct import load_a_history, reconstruct_log_rho  # noqa: E402


def main():
    llr_path = RESULTS / "amp_llr.h5"
    hmc_path = RESULTS / "hmc.h5"
    if not llr_path.exists():
        print(f"no {llr_path}; run run.sh first", file=sys.stderr)
        return 1

    with h5py.File(llr_path, "r") as f:
        volume = float(f["/cfg"].attrs["V"])
        ndim = int(f["/vars"].attrs["ndim"])
        extent = int(f["/vars"].attrs["size"])
    llr = load_a_history(llr_path)
    e_n = llr["E_n"]              # Σφ² units
    p2 = e_n / volume            # ⟨φ²⟩ per site
    a_hist = llr["a_hist"]
    n_nr = llr["n_nr"]
    a_final = np.median(a_hist[:, -max(1, a_hist.shape[1] // 5):], axis=1)
    ln_rho = reconstruct_log_rho(e_n, a_final)   # ln ρ(Q) = ∫ a dQ

    # Direct validation histogram: ⟨φ²⟩ from the unconstrained HMC chain.
    with h5py.File(hmc_path, "r") as f:
        m2 = np.asarray(f["/prod/obs/m2"][:], dtype=float)
    h, edges = np.histogram(m2, bins=45, density=True)
    cen = 0.5 * (edges[1:] + edges[:-1])
    m = h > 0
    cen, ln_p = cen[m], np.log(h[m])
    lo, hi = cen.min(), cen.max()

    # Align LLR to the direct histogram over their overlap in ⟨φ²⟩.
    sel = (p2 >= lo) & (p2 <= hi)
    shift = np.mean(np.interp(p2[sel], cen, ln_p)) - np.mean(ln_rho[sel])
    ln_rho_al = ln_rho + shift
    zc = float(np.interp(0.0, a_final[::-1], p2[::-1]))  # ⟨φ²⟩ where a=0

    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.spines.top": False,
                         "axes.spines.right": False})
    c_llr, c_dir = "#E45756", "#4C78A8"
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.4))
    fig.suptitle("Generalized-observable LLR: density of states of Q = Σφ²  "
                 f"(φ⁴ {ndim}D L={extent} · {len(e_n)} replicas · "
                 "orch::llr::Orchestrator)", fontsize=11, y=0.99)

    # 1) tilt adaptation history (colour = window centre)
    for n in range(len(e_n)):
        ax[0, 0].plot(a_hist[n], lw=0.9, alpha=0.8,
                      color=plt.cm.viridis(n / max(1, len(e_n) - 1)))
    ax[0, 0].axvline(n_nr - 0.5, color="0.4", ls="--", lw=1.0)
    ax[0, 0].text(n_nr - 0.4, ax[0, 0].get_ylim()[1] * 0.85, " NR → RM",
                  fontsize=8, color="0.35")
    ax[0, 0].set(title="Tilt adaptation per replica (colour = window)",
                 xlabel="iteration", ylabel="a")

    # 2) converged tilt a_n = d lnρ/dQ
    ax[0, 1].axhline(0, color="0.6", lw=0.8)
    ax[0, 1].plot(p2, a_final, "o-", color=c_llr, ms=3.5, lw=1.3)
    ax[0, 1].axvline(zc, color="0.5", ls=":", lw=1.0)
    ax[0, 1].text(zc, ax[0, 1].get_ylim()[1] * 0.88, f" a=0 at\n ⟨φ²⟩≈{zc:.3f}",
                  fontsize=8, color="0.35")
    ax[0, 1].set(title="Converged tilt  a_n = d lnρ/dQ", xlabel="⟨φ²⟩", ylabel="a")

    # 3) ln ρ vs direct histogram
    ax[1, 0].plot(p2, ln_rho_al, "o-", color=c_llr, ms=3.5, lw=1.5,
                  label="LLR  ∫a dQ")
    ax[1, 0].plot(cen, ln_p, "s", color=c_dir, ms=3.5, alpha=0.85,
                  label="direct histogram")
    ax[1, 0].axvspan(lo, hi, color=c_dir, alpha=0.06)
    ax[1, 0].set(title="log density of states  ln ρ(⟨φ²⟩)", xlabel="⟨φ²⟩",
                 ylabel="ln ρ (+const)")
    ax[1, 0].legend(frameon=False)

    # 4) ρ normalised
    rho = np.exp(ln_rho_al - ln_rho_al.max())
    p_dir = np.exp(ln_p - ln_p.max())
    ax[1, 1].plot(p2, rho, "o-", color=c_llr, ms=3.5, lw=1.5, label="LLR ρ (norm.)")
    ax[1, 1].plot(cen, p_dir, "s", color=c_dir, ms=3.5, alpha=0.85,
                  label="direct (norm.)")
    ax[1, 1].set(title="density of states  ρ(⟨φ²⟩)  (peak-normalised)",
                 xlabel="⟨φ²⟩", ylabel="ρ/ρ_max")
    ax[1, 1].legend(frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = HERE / "generalized_dos_llr.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"replicas={len(e_n)}  d(⟨φ²⟩)={p2[1]-p2[0]:.4f}  δ(Σφ²)={e_n[1]-e_n[0]:.2f}")
    print(f"a=0 (DoS peak) at ⟨φ²⟩ ≈ {zc:.3f}")
    print(f"ln ρ range: {ln_rho.max()-ln_rho.min():.1f} "
          f"(~{(ln_rho.max()-ln_rho.min())/np.log(10):.1f} decades)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
