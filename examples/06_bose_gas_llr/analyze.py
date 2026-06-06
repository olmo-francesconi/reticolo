#!/usr/bin/env python3
"""LLR-vs-HMC diagnostic plot for the 4D Bose gas at finite chemical
potential. Per-µ panel layout mirrors examples 04 and 05:

  Row 0:  ln ρ(s) — HMC histogram (markers) vs LLR piecewise-exp (line)
          vs odd-poly fit antiderivative (dashed).
  Row 1:  ρ(s)    — linear, unit integral, same overlay.
  Row 2:  a_n     — convergence per replica across NR + RM iterations.
  Row 3:  a_n vs s — converged LLR slope = d ln ρ_pq/dS_I + odd-poly fit.
  Row 4:  cumulative oscillating integral with accumulation point s* and
          fit/integration cut.

A second figure (phase_factor.pdf) collects <e^{iS_I}>_pq and the overlap
free-energy density −ln<e^{iS_I}>/V vs µ, LLR vs HMC. LLR errors are the
spread over independent-seed replica runs (LLR_SEEDS in run.sh).

S_I distribution is symmetric (ρ(S_I) = ρ(−S_I)) — we run the LLR only on
[0, S_I_max), so all per-µ panels use the intensive variable s = S_I / V on
the x-axis with a vertical line marking s = 0 (the natural peak by symmetry).
The HMC chain is folded via |S_I| so both halves contribute to the
histogram; the first bin (the s = 0 boundary) is half-width and uses the
appropriately scaled density estimator.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

sys.path.insert(0, str(HERE.parent / "_common"))
from llr_reconstruct import (  # noqa: E402
    load_a_history,
    reconstruct_log_rho,
)
from llr_polyfit import (  # noqa: E402
    fit_odd_poly,
    log_rho_coeffs,
    rm_tail,
    select_order,
)

try:
    from oscillating_integral import accumulation_scan, phase_factor  # noqa: E402
except ImportError:
    phase_factor = None
    print("python-flint not installed — skipping phase-factor reconstruction",
          file=sys.stderr)

S_CUT_SAFETY = 1.2


def load_hmc(path):
    # S_I (imaginary action) — bose gas is the only mode-B LLR app, so
    # the HMC twin records `/prod/obs/s_i` rather than `/prod/obs/s`.
    with h5py.File(path, "r") as f:
        return np.asarray(f["/prod/obs/s_i"][:], dtype=float)


def load_llr(path):
    # Wrap the shared loader to also stash mu / volume / size / ndim — the
    # plot code uses `volume` to render the intensive variable s = S_I / V.
    data = load_a_history(path)
    with h5py.File(path, "r") as f:
        size = int(f["/vars"].attrs["size"])
        ndim = int(f["/vars"].attrs["ndim"])
        mu   = float(f["/vars"].attrs["mu"])
    data["size"]   = size
    data["ndim"]   = ndim
    data["mu"]     = mu
    data["volume"] = size ** ndim
    return data


def piecewise_log_rho_symmetric(e_n, a_final, log_rho, delta, n_per_window=16):
    """Finely-sampled piecewise-exponential reconstruction on [−E_max, +E_max].

    LLR ran on [0, S_max] with windows centred at `E_n = δ/2, 3δ/2, ..., S_max
    − δ/2` (the half-step shift means no window sits on S_I = 0). By the
    symmetry ρ(S_I) = ρ(−S_I) the negative-side reconstruction has the same
    `log_rho[n]` value at `−E_n` and slope `−a_final[n]`. Every window is
    mirrored — no skip, no duplicates.
    """
    xs, ys = [], []
    half = 0.5 * delta
    for n in range(len(e_n) - 1, -1, -1):
        ctr   = -e_n[n]
        seg_x = np.linspace(ctr - half, ctr + half, n_per_window)
        seg_y = log_rho[n] + (-a_final[n]) * (seg_x - ctr)
        xs.append(seg_x)
        ys.append(seg_y)
    for n in range(len(e_n)):
        seg_x = np.linspace(e_n[n] - half, e_n[n] + half, n_per_window)
        seg_y = log_rho[n] + a_final[n] * (seg_x - e_n[n])
        xs.append(seg_x)
        ys.append(seg_y)
    return np.concatenate(xs), np.concatenate(ys)


def main():
    hmc_files = sorted(RESULTS.glob("hmc_mu*.h5"))
    if not hmc_files:
        print(f"no inputs in {RESULTS}; run run.sh first", file=sys.stderr)
        return 1

    pairs = []
    for h in hmc_files:
        suffix = h.stem.split("mu")[-1]
        # Seed replicas llr_mu<X>_s<seed>.h5; fall back to the unsuffixed
        # single-run layout for results predating LLR_SEEDS.
        lfiles = sorted(RESULTS.glob(f"llr_mu{suffix}_s*.h5"))
        if not lfiles and (RESULTS / f"llr_mu{suffix}.h5").exists():
            lfiles = [RESULTS / f"llr_mu{suffix}.h5"]
        if not lfiles:
            continue
        pairs.append((float(suffix), h, lfiles))
    pairs.sort(key=lambda p: p[0])

    n_cols = len(pairs)
    fig, axes = plt.subplots(
        5, n_cols, figsize=(4.5 * n_cols, 17.5), constrained_layout=True, squeeze=False
    )

    pf_rows = []  # (mu, pf_llr, pf_llr_err, pf_hmc, pf_hmc_err)

    for col, (mu, hmc_path, llr_paths) in enumerate(pairs):
        s_chain = load_hmc(hmc_path)
        llr     = load_llr(llr_paths[0])
        e_n     = llr["E_n"]
        delta   = llr["delta"]  # window width; grid step may be narrower
        spacing = float(e_n[1] - e_n[0]) if len(e_n) > 1 else delta
        volume  = llr["volume"]
        a_hist  = llr["a_hist"]
        hists   = [a_hist] + [load_a_history(lp)["a_hist"] for lp in llr_paths[1:]]
        n_seed  = len(hists)
        # Converged a_n per seed (median of the RM tail), shape (n_seed, n_rep).
        a_fin_seeds = np.stack(
            [np.median(h[:, -max(1, h.shape[1] // 5):], axis=1) for h in hists]
        )

        # Paper-style reconstruction, per independent seed replica:
        #   1. unweighted odd-poly fit of a(s) over the full window range,
        #   2. accumulation scan — the oscillating integral saturates well
        #      before s_max, and fitting data past that point only feeds
        #      uncertainty into the integral,
        #   3. refit on s <= s_cut = safety * s_star, integrate up to s_cut.
        # The seed spread is the statistical error on <e^{iS_I}>.
        # The complex action is S = S_R + i sinh(mu) S_I and /prod/obs/s_i
        # records the bare S_I, so the Fourier frequency is sinh(mu).
        omega = np.sinh(mu)
        s     = e_n / volume
        s_max = (e_n[-1] + 0.5 * spacing) / volume
        acc   = None
        if phase_factor is not None:
            # Accumulation cut per seed (the chi2-selected full-range fit
            # only locates s*; the production order is chosen below on the
            # integral itself).
            a_cs, s_stars = [], []
            for hist in hists:
                a_c = rm_tail(hist).mean(axis=1)
                (_, c_full, _), _ = select_order(s, a_c)
                s_star, _, _ = accumulation_scan(log_rho_coeffs(c_full), volume,
                                                 s_max, omega)
                a_cs.append(a_c)
                s_stars.append(s_star)
            s_star = float(np.median(s_stars))
            s_cut  = min(s_max, S_CUT_SAFETY * s_star)
            m = s <= s_cut

            # Order selection on the integral itself: scan odd orders, average
            # the phase factor over seeds, and keep the last order of the
            # monotone prefix — the truncation bias releases monotonically
            # with the order, and noise breaks the trend. chi2 alone plateaus
            # at the per-window noise floor while the integral is still
            # drifting, which biases <e^{iS_I}> low.
            orders = [o for o in (3, 5, 7, 9, 11, 13)
                      if (o + 1) // 2 <= int(m.sum()) - 2]
            pf_tab = np.array(
                [[phase_factor(log_rho_coeffs(fit_odd_poly(s[m], a_c[m], o)[0]),
                               volume, s_cut, omega)[0]
                  for a_c in a_cs] for o in orders])  # (n_order, n_seed)
            pf_mean_o = pf_tab.mean(axis=1)
            sel, sgn = 0, 0.0
            for i in range(1, len(orders)):
                step = float(np.sign(pf_mean_o[i] - pf_mean_o[i - 1]))
                sgn = step if sgn == 0.0 else sgn
                if step != sgn:
                    break
                sel = i
            order    = orders[sel]
            pf_seeds = pf_tab[sel]
            coeffs   = fit_odd_poly(s[m], a_cs[0][m], order)[0]
            _, acc_xs, acc_cum = accumulation_scan(log_rho_coeffs(coeffs),
                                                   volume, s_cut, omega)
            acc = (acc_xs, acc_cum, s_star, float(pf_seeds[0]))
            print(f"mu={mu}: s* = {s_star:.4f}, fit {m.sum()}/{len(s)} windows; "
                  "pf order scan "
                  + "  ".join(f"{o}:{p:.3e}" for o, p in zip(orders, pf_mean_o))
                  + f"  -> order {order}")
            pf_mean = float(np.mean(pf_seeds))
            pf_err  = float(np.std(pf_seeds, ddof=1)) if len(pf_seeds) > 1 else 0.0
            cos_chain  = np.cos(omega * s_chain)
            pf_hmc     = float(cos_chain.mean())
            pf_hmc_err = float(cos_chain.std(ddof=1) / np.sqrt(len(cos_chain)))
            pf_rows.append((mu, pf_mean, pf_err, pf_hmc, pf_hmc_err, volume,
                            llr["size"]))
            print(f"mu={mu}: <e^iS_I> LLR = {pf_mean:.4e} +- {pf_err:.1e} "
                  f"({len(pf_seeds)} seeds)   HMC = {pf_hmc:.4e} +- {pf_hmc_err:.1e}")
        else:
            s_cut = s_max
            (order, coeffs, _), table = select_order(s, rm_tail(a_hist).mean(axis=1))
            print(f"mu={mu}: fit order {order} (chi2/dof "
                  + ", ".join(f"{o}:{x:.3g}" for o, x in table) + ")")

        # Symmetric grid for plotting: LLR ran on [0, S_max] but ρ is
        # symmetric so we mirror to display the full bell curve. With the
        # half-step shift the innermost window centre is at S_I = ±δ/2; no
        # element on either side coincides at S_I = 0, so the mirror is
        # full (no skip).
        e_n_sym = np.concatenate([-e_n[::-1], e_n])

        # HMC histogram on the full symmetric grid — no folding; bin centres
        # at e_n_sym, bin width = the grid spacing throughout (including the
        # bin centred on zero, which then naturally absorbs samples on both
        # sides of 0).
        bin_edges = np.concatenate([[e_n_sym[0] - spacing / 2.0],
                                    e_n_sym + spacing / 2.0])
        n_total   = len(s_chain)
        raw_counts, _ = np.histogram(s_chain, bins=bin_edges)
        density_hmc   = raw_counts / (n_total * spacing)
        density_err   = np.sqrt(raw_counts) / (n_total * spacing)
        mask          = raw_counts > 0
        log_density_hmc       = np.full_like(density_hmc, np.nan, dtype=float)
        log_density_hmc[mask] = np.log(density_hmc[mask])
        log_density_hmc_err   = np.full_like(density_hmc, np.nan, dtype=float)
        log_density_hmc_err[mask] = 1.0 / np.sqrt(raw_counts[mask])

        # Align each seed's reconstruction to HMC at the innermost positive
        # window (S_I = +δ/2); no window sits exactly at S_I = 0 with the
        # half-step shift.
        inner = log_density_hmc[len(e_n)]  # in e_n_sym, +δ/2 sits here
        lr_shifted, pw_logs = [], []
        for af in a_fin_seeds:
            lr  = reconstruct_log_rho(e_n, af)
            lrs = lr + (inner - lr[0])
            pw_x, pw = piecewise_log_rho_symmetric(e_n, af, lrs, spacing)
            lr_shifted.append(lrs)
            pw_logs.append(pw)
        lr_shifted  = np.stack(lr_shifted)
        sym_shifted = np.concatenate([lr_shifted[:, ::-1], lr_shifted], axis=1)
        sym_mean = sym_shifted.mean(axis=0)
        sym_std  = (sym_shifted.std(axis=0, ddof=1) if n_seed > 1
                    else np.zeros_like(sym_mean))

        # Intensive x-axis: s = S_I / V.
        e_n_iv  = e_n_sym / volume
        pw_x_iv = pw_x / volume

        # Row 0: ln ρ overlay.
        ax = axes[0, col]
        ax.errorbar(e_n_iv[mask], log_density_hmc[mask], yerr=log_density_hmc_err[mask],
                    fmt="x", markersize=4, markeredgewidth=0.8,
                    elinewidth=0.8, capsize=2, color="C0", label="HMC histogram")
        for j, pw in enumerate(pw_logs):
            ax.plot(pw_x_iv, pw, "-", color="C3", linewidth=0.6, alpha=0.7,
                    label=f"LLR piecewise ({n_seed} seeds)" if j == 0 else None)
        ax.errorbar(e_n_iv, sym_mean, yerr=sym_std, fmt="x", color="C3",
                    markersize=4, markeredgewidth=0.8, elinewidth=0.8, capsize=2)
        s_plot  = np.linspace(-s_cut, s_cut, 400)
        p_curve = volume * np.polynomial.polynomial.polyval(s_plot, log_rho_coeffs(coeffs))
        p_curve += inner - volume * np.polynomial.polynomial.polyval(
            s[0], log_rho_coeffs(coeffs))
        ax.plot(s_plot, p_curve, "--", color="k", linewidth=0.9,
                label=f"poly fit (order {order + 1})")
        ax.axvline(0.0, color="0.7", linewidth=0.8, linestyle="--")
        ax.set_title(rf"$\mu = {mu:.2f}$")
        ax.set_xlabel(r"$S_I / V$")
        ax.set_ylabel(r"$\ln \rho(S_I)$ + const.")
        ax.legend(loc="best", fontsize=8)

        # Row 1: ρ(S_I) linear, unit integral over the plotted x-range, one
        # curve per seed.
        ax = axes[1, col]
        z_hmc            = float(np.trapezoid(density_hmc, e_n_iv))
        density_hmc_norm = density_hmc / z_hmc if z_hmc > 0 else density_hmc
        density_err_norm = density_err / z_hmc if z_hmc > 0 else density_err

        ax.errorbar(e_n_iv, density_hmc_norm, yerr=density_err_norm,
                    fmt="x", markersize=4, markeredgewidth=0.8,
                    elinewidth=0.8, capsize=2, color="C0", label="HMC histogram")
        for j, pw in enumerate(pw_logs):
            rho_un = np.exp(pw - pw.max())
            z_llr  = float(np.trapezoid(rho_un, pw_x_iv))
            ax.plot(pw_x_iv, rho_un / z_llr if z_llr > 0 else rho_un, "-",
                    color="C3", linewidth=0.6, alpha=0.7,
                    label=f"LLR piecewise ({n_seed} seeds)" if j == 0 else None)
        ax.axvline(0.0, color="0.7", linewidth=0.8, linestyle="--")
        ax.set_xlabel(r"$S_I / V$")
        ax.set_ylabel(r"$\rho(S_I)$  (unit integral)")
        ax.legend(loc="best", fontsize=8)

        # Row 2: a_n history per window — mean over seeds, error bar = the
        # standard deviation across seeds at that iteration.
        ax = axes[2, col]
        hist_stack = np.stack(hists)  # (n_seed, n_rep, n_iter)
        h_mean = hist_stack.mean(axis=0)
        h_std  = (hist_stack.std(axis=0, ddof=1) if n_seed > 1
                  else np.zeros_like(h_mean))
        n_rep, n_iter = h_mean.shape
        iters = np.arange(n_iter)
        cmap = plt.get_cmap("viridis", n_rep)
        for n in range(n_rep):
            show_label = (n_rep <= 8 or n % max(1, n_rep // 6) == 0)
            ax.errorbar(iters, h_mean[n], yerr=h_std[n], linewidth=1.0,
                        elinewidth=0.6, capsize=0, color=cmap(n),
                        label=f"$S_I/V$={e_n[n] / volume:.3f}" if show_label else None)
        ax.axvline(llr["n_nr"] - 0.5, color="0.4", linewidth=0.8, linestyle="--",
                   label="NR → RM")
        ax.axhline(0.0, color="0.8", linewidth=0.8)
        ax.set_xlabel("iteration (NR then RM)")
        ax.set_ylabel(r"$a_n$")
        ax.set_title(rf"$a_n$ convergence (mean $\pm$ sd, {n_seed} seeds)")
        ax.legend(loc="best", fontsize=7, ncol=2)

        # Row 3: converged a_n vs S_I/V. By ρ(−s) = ρ(s) the slope is
        # antisymmetric — we mirror the LLR-sampled positive half onto the
        # negative axis so the symmetry is visible.
        ax = axes[3, col]
        af_mean = a_fin_seeds.mean(axis=0)
        af_std  = (a_fin_seeds.std(axis=0, ddof=1) if n_seed > 1
                   else np.zeros_like(af_mean))
        ax.errorbar(e_n_iv, np.concatenate([-af_mean[::-1], af_mean]),
                    yerr=np.concatenate([af_std[::-1], af_std]),
                    fmt="o-", color="C2", markersize=4, linewidth=0.7,
                    elinewidth=0.8, capsize=2,
                    label=rf"$a_n$ (mean $\pm$ sd, {n_seed} seeds, mirrored)")
        ax.plot(s_plot, np.polynomial.polynomial.polyval(s_plot, coeffs), "--",
                color="k", linewidth=0.9, label=f"odd poly fit (order {order})")
        ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle="--",
                   label=r"$a = 0$ (natural peak)")
        ax.axvline(0.0, color="0.7", linewidth=0.8, linestyle="--")
        ax.set_xlabel(r"$S_I / V$")
        ax.set_ylabel(r"$a_n = d \ln \rho_{pq}/dS_I$")
        ax.legend(loc="best", fontsize=8)

        # Row 4: integral accumulation — cumulative <e^{iS_I}> from the cut
        # fit must plateau at the final value before the cut; s* is where
        # the certified envelope bound says the tail no longer contributes.
        ax = axes[4, col]
        if acc is not None:
            acc_xs, acc_cum, s_star, pf_v = acc
            ax.plot(acc_xs, np.abs(acc_cum), "-", color="C3", linewidth=0.9,
                    label="|cumulative integral|")
            ax.axhline(abs(pf_v), color="0.6", linewidth=0.8)
            ax.set_yscale("log")
            ax.axvline(s_star, color="C0", linewidth=0.8, linestyle="--",
                       label=r"$s_\star$ (accumulation point)")
            ax.axvline(s_cut, color="C2", linewidth=0.8, linestyle=":",
                       label=r"$s_{\rm cut}$ (fit/integration limit)")
            ax.set_xlabel(r"$S_I / V$ upper limit")
            ax.set_ylabel(r"|cumulative $\langle e^{iS_I}\rangle_{pq}$|")
            ax.legend(loc="best", fontsize=8)
        else:
            ax.set_axis_off()

    out = HERE / "rho_hmc_vs_llr.pdf"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    if pf_rows:
        mus, pf, pf_err, pf_h, pf_h_err, vols, sizes = map(np.array, zip(*pf_rows))
        fig2, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=(14.0, 4.0), constrained_layout=True
        )
        # One style per series, same plotting (= legend) order on every panel:
        # LLR, HMC, paper.
        llr_kw   = dict(fmt="o", markersize=4, capsize=3, color="C3",
                        label="LLR (poly fit + multiprec.)")
        hmc_kw   = dict(fmt="x", markersize=5, capsize=3, color="C0",
                        label=r"HMC $\langle\cos(\sinh\mu\, S_I)\rangle$")
        L = int(sizes[0])
        paper_kw = dict(fmt="s", markersize=3, capsize=3, color="k", mfc="none",
                        label=f"PRD 101, 014504 ($ {L}^4 $)")

        # Reference curve from PRD 101, 014504 (2020), Tab. 2 — full-precision
        # LLR at lambda = m = 1, if the run's volume is tabulated there.
        paper = HERE / "paper_data_raw" / "1910.11026_deltaF_4to10_tab2.csv"
        pf_paper = None
        if paper.exists() and np.all(sizes == L):
            rows = [ln.strip() for ln in paper.read_text().splitlines()
                    if ln.strip() and not ln.startswith("#")]
            names = [n.strip() for n in rows[0].split(",")]
            body = np.array([[np.nan if v.strip() == "--" else float(v)
                              for v in r.split(",")] for r in rows[1:]])
            tab = {n: body[:, i] for i, n in enumerate(names)}
            cname = f"dF_{L}x{L}"
            if cname in tab:
                okp = np.isfinite(tab[cname])
                mu_p = tab["mu"][okp]
                dfp  = tab[cname][okp] * 1e-3
                dfe  = tab[f"err_{L}x{L}"][okp] * 1e-3
                pf_paper = np.exp(-vols * np.interp(mus, mu_p, dfp))
                pf_paper_err = vols * np.interp(mus, mu_p, dfe) * pf_paper

        ax1.errorbar(mus, pf, yerr=pf_err, **llr_kw)
        ax1.errorbar(mus, pf_h, yerr=pf_h_err, **hmc_kw)
        if pf_paper is not None:
            ax1.errorbar(mu_p, np.exp(-vols[0] * dfp),
                         yerr=vols[0] * dfe * np.exp(-vols[0] * dfp), **paper_kw)
        ax1.set_yscale("log")
        ax1.set_xlabel(r"$\mu$")
        ax1.set_ylabel(r"$\langle e^{i\sinh\mu\, S_I}\rangle_{pq}$")

        # Overlap free-energy density ΔF = −ln<e^{i sinh(mu) S_I}>/V (points
        # with a non-positive central value carry no log — skipped).
        for vals, errs, kw in ((pf, pf_err, llr_kw), (pf_h, pf_h_err, hmc_kw)):
            ok = vals > 0
            ax2.errorbar(mus[ok], -np.log(vals[ok]) / vols[ok],
                         yerr=errs[ok] / (vals[ok] * vols[ok]), **kw)
        if pf_paper is not None:
            ax2.errorbar(mu_p, dfp, yerr=dfe, **paper_kw)
        ax2.set_xlabel(r"$\mu$")
        ax2.set_ylabel(r"$\Delta F = -\ln\langle e^{i\sinh\mu\, S_I}\rangle_{pq}\,/\,V$")

        # Pull plot: deviation from the LLR reference in units of the LLR
        # seed-spread standard deviation.
        okd = pf_err > 0
        ax3.axhspan(-1.0, 1.0, color="C3", alpha=0.15,
                    label=r"LLR $\pm 1\sigma$")
        ax3.axhline(0.0, color="C3", linewidth=0.8)
        ax3.errorbar(mus[okd], (pf_h[okd] - pf[okd]) / pf_err[okd],
                     yerr=pf_h_err[okd] / pf_err[okd], **hmc_kw)
        if pf_paper is not None:
            ax3.errorbar(mus[okd], (pf_paper[okd] - pf[okd]) / pf_err[okd],
                         yerr=pf_paper_err[okd] / pf_err[okd], **paper_kw)
        ax3.set_ylim(-12.0, 12.0)  # HMC pulls past mu_c are off-scale noise
        ax3.set_xlabel(r"$\mu$")
        ax3.set_ylabel(r"$(x - \mathrm{LLR})\,/\,\sigma_{\mathrm{LLR}}$")
        for ax in (ax1, ax2, ax3):
            ax.legend(loc="best", fontsize=8)
        for ax in (ax1, ax2, ax3):
            ax.set_xlim(0.0, 2.05)
        out2 = HERE / "phase_factor.pdf"
        fig2.savefig(out2, dpi=150)
        print(f"wrote {out2}")

if __name__ == "__main__":
    main()
