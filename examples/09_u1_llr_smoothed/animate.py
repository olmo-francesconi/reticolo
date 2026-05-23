#!/usr/bin/env python3
"""Side-by-side animation of the DoS evolution for vanilla LLR and smoothed
LLR. Same per-frame layout as examples/05_u1_llr/animate_critical.py,
mirrored across the figure: left half = vanilla, right half = smoothed.

Inputs from run.sh:
    results/llr_vanilla.h5
    results/llr_smoothed.h5
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

sys.path.insert(0, str(HERE.parent / "_common"))
from llr_reconstruct import (  # noqa: E402
    load_a_history,
    piecewise_log_rho,
    reconstruct_log_rho,
)


def load_llr(path):
    data = load_a_history(path)
    n_rep, n_total = data["a_hist"].shape
    pre_hist = None
    with h5py.File(path, "r") as f:
        size = int(f["/vars"].attrs["size"])
        ndim = int(f["/vars"].attrs["ndim"])
        beta = float(f["/vars"].attrs["beta"])
        if f"/replica_{0:03d}/a_pre_shrink" in f:
            pre_hist = np.empty((n_rep, n_total), dtype=float)
            for n in range(n_rep):
                pre = np.asarray(f[f"/replica_{n:03d}/a_pre_shrink"][:], dtype=float)
                m = min(len(pre), n_total)
                pre_hist[n, :m] = pre[:m]
                if m < n_total:
                    pre_hist[n, m:] = pre[-1]
    n_plaq = (ndim * (ndim - 1) // 2) * (size ** ndim)
    data["scale"] = float(n_plaq)
    data["beta"] = beta
    data["size"] = size
    data["ndim"] = ndim
    data["a_pre_hist"] = pre_hist  # None for vanilla runs
    return data


def precompute_frames(data):
    """Per-iteration ln rho / rho reconstructions, indexed by frame k."""
    e_n = data["E_n"]
    a_hist = data["a_hist"]
    delta = data["delta"]
    scale = data["scale"]
    n_iter = a_hist.shape[1]
    pw_x = None
    pw_logs, log_centres_list, rhos = [], [], []
    for k in range(n_iter):
        a_k = a_hist[:, k]
        log_rho = reconstruct_log_rho(e_n, a_k)
        pw_x_now, pw_log = piecewise_log_rho(e_n, a_k, log_rho, delta)
        if pw_x is None:
            pw_x = pw_x_now
        offset = float(pw_log.max())
        pw_log -= offset
        log_rho -= offset
        rho_un = np.exp(pw_log)
        z = float(np.trapezoid(rho_un, pw_x / scale))
        rho = rho_un / z if z > 0 else rho_un
        pw_logs.append(pw_log)
        log_centres_list.append(log_rho)
        rhos.append(rho)
    return {"pw_x_p": pw_x / scale, "pw_logs": pw_logs,
            "log_centres": log_centres_list, "rhos": rhos}


def build_panel(fig, gs_slice, frames, data, label, shared_limits):
    """Build the 2x2 panel block for one algorithm into the given gridspec
    slice. Returns the artists needed by `update`."""
    e_n_p = data["E_n"] / data["scale"]
    a_hist = data["a_hist"]
    n_rep, n_iter = a_hist.shape
    n_nr = data["n_nr"]
    iters_x = np.arange(n_iter)

    axes = np.empty((2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            axes[i, j] = fig.add_subplot(gs_slice[i, j])

    ax = axes[0, 0]
    ln_pw, = ax.plot([], [], "-", color="C3", lw=1.0, label="LLR (piecewise exp)")
    pt_c, = ax.plot([], [], "x", color="C3", ms=5, mew=1.0,
                    label="LLR window centres")
    ax.set_xlim(e_n_p.min(), e_n_p.max())
    ax.set_ylim(shared_limits["y_log_min"], 1.0)
    ax.set_xlabel(r"$S / (6 V)$")
    ax.set_ylabel(r"$\ln \rho$  (peak = 0)")
    ax.set_title(label, fontsize=11)
    ax.legend(loc="lower center", fontsize=7)

    ax = axes[0, 1]
    ln_rho, = ax.plot([], [], "-", color="C3", lw=1.0)
    ax.set_xlim(e_n_p.min(), e_n_p.max())
    ax.set_ylim(0.0, shared_limits["y_rho_max"])
    ax.set_xlabel(r"$S / (6 V)$")
    ax.set_ylabel(r"$\rho$  (unit integral)")

    ax = axes[1, 0]
    cmap = plt.get_cmap("viridis", n_rep)
    trace_lines = [ax.plot([], [], lw=1.0, color=cmap(n))[0] for n in range(n_rep)]
    ax.axvline(n_nr - 0.5, color="0.4", lw=0.8, ls="--", label="NR → RM")
    ax.axhline(0.0, color="0.8", lw=0.8)
    cursor = ax.axvline(0, color="C3", lw=1.2, alpha=0.6)
    ax.set_xlim(-0.5, n_iter - 0.5)
    ax.set_ylim(shared_limits["a_min"], shared_limits["a_max"])
    ax.set_xlabel("iteration (NR then RM)")
    ax.set_ylabel(r"$a_n$")
    ax.legend(loc="best", fontsize=7)

    ax = axes[1, 1]
    a_pre_hist = data.get("a_pre_hist")
    if a_pre_hist is not None:
        ln_aE_pre, = ax.plot([], [], "o-", color="C2", ms=3, lw=0.7,
                             alpha=0.25,
                             label=r"$a_n$ pre-shrink (post-RM)")
    else:
        ln_aE_pre = None
    ln_aE, = ax.plot([], [], "o-", color="C2", ms=4, lw=0.7,
                     label=r"$a_n$ at frame $k$")
    ax.axhline(0.0, color="0.6", lw=0.8, ls="--", label=r"$a = 0$")
    ax.set_xlim(e_n_p.min(), e_n_p.max())
    ax.set_ylim(shared_limits["a_min"], shared_limits["a_max"])
    ax.set_xlabel(r"$S / (6 V)$")
    ax.set_ylabel(r"$a_n$")
    ax.legend(loc="best", fontsize=7)

    return {
        "e_n_p": e_n_p,
        "iters_x": iters_x,
        "a_hist": a_hist,
        "a_pre_hist": a_pre_hist,
        "frames": frames,
        "ln_pw": ln_pw, "pt_c": pt_c, "ln_rho": ln_rho,
        "trace_lines": trace_lines, "cursor": cursor,
        "ln_aE": ln_aE, "ln_aE_pre": ln_aE_pre,
        "n_rep": n_rep,
    }


def update_panel(panel, k):
    panel["ln_pw"].set_data(panel["frames"]["pw_x_p"], panel["frames"]["pw_logs"][k])
    panel["pt_c"].set_data(panel["e_n_p"], panel["frames"]["log_centres"][k])
    panel["ln_rho"].set_data(panel["frames"]["pw_x_p"], panel["frames"]["rhos"][k])
    for n in range(panel["n_rep"]):
        panel["trace_lines"][n].set_data(panel["iters_x"][:k + 1], panel["a_hist"][n, :k + 1])
    panel["cursor"].set_xdata([k, k])
    panel["ln_aE"].set_data(panel["e_n_p"], panel["a_hist"][:, k])
    if panel["ln_aE_pre"] is not None:
        panel["ln_aE_pre"].set_data(panel["e_n_p"], panel["a_pre_hist"][:, k])


def main():
    vpath = RESULTS / "llr_vanilla.h5"
    ppath = RESULTS / "llr_smoothed.h5"
    if not vpath.exists() or not ppath.exists():
        print(f"missing {vpath} or {ppath}; run run.sh first", file=sys.stderr)
        return 1

    van = load_llr(vpath)
    pol = load_llr(ppath)
    assert van["a_hist"].shape == pol["a_hist"].shape, \
        "vanilla and smoothed runs disagree on (n_rep, n_iter); rerun with matching args"
    assert np.allclose(van["E_n"], pol["E_n"]), "E_n grids differ"

    n_iter = van["a_hist"].shape[1]
    n_nr = van["n_nr"]
    size = van["size"]
    ndim = van["ndim"]
    beta = van["beta"]

    van_frames = precompute_frames(van)
    pol_frames = precompute_frames(pol)

    # Shared limits across the two panels so the eye compares directly.
    y_log_min = min(
        float(van_frames["pw_logs"][-1].min()),
        float(pol_frames["pw_logs"][-1].min()),
    ) - 1.5
    tail = max(1, n_iter // 10)
    y_rho_max = max(
        max(float(r.max()) for r in van_frames["rhos"][-tail:]),
        max(float(r.max()) for r in pol_frames["rhos"][-tail:]),
    ) * 1.2
    a_arrays = [van["a_hist"], pol["a_hist"]]
    for d in (van, pol):
        if d.get("a_pre_hist") is not None:
            a_arrays.append(d["a_pre_hist"])
    a_min = min(float(np.nanmin(arr)) for arr in a_arrays)
    a_max = max(float(np.nanmax(arr)) for arr in a_arrays)
    a_pad = 0.1 * (a_max - a_min) if a_max > a_min else 0.1
    shared = {"y_log_min": y_log_min, "y_rho_max": y_rho_max,
              "a_min": a_min - a_pad, "a_max": a_max + a_pad}

    fig = plt.figure(figsize=(18.0, 9.0), constrained_layout=True)
    outer = fig.add_gridspec(1, 2, wspace=0.08)
    left = outer[0, 0].subgridspec(2, 2)
    right = outer[0, 1].subgridspec(2, 2)

    van_panel = build_panel(fig, left, van_frames, van, "vanilla LLR", shared)
    pol_panel = build_panel(fig, right, pol_frames, pol, "smoothed LLR", shared)
    title = fig.suptitle("")

    def update(k):
        update_panel(van_panel, k)
        update_panel(pol_panel, k)
        if k < n_nr:
            phase, idx = "NR", k + 1
        else:
            phase, idx = "RM", k - n_nr + 1
        title.set_text(
            rf"$L={size}^{{{ndim}}}$, $\beta = {beta:.4f}$  —  "
            rf"{phase} iter {idx}  ({k + 1}/{n_iter})"
        )
        return ()

    anim = FuncAnimation(fig, update, frames=n_iter, interval=200, blit=False)
    out = HERE / "rho_evolution_side_by_side.gif"
    anim.save(out, writer=PillowWriter(fps=6))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
