"""Certified multiprecision evaluation of the LLR oscillating integral.

<e^{iS_I}>_pq = integral rho(S_I) e^{iS_I} dS_I / integral rho(S_I) dS_I
with rho = exp(V * P(s)), s = S_I/V, P the even polynomial from
llr_polyfit. The numerator cancels over more orders of magnitude as V and
mu grow, so both integrals run through FLINT's acb_calc_integrate (Petras
algorithm, certified ball arithmetic) at a working precision that
auto-escalates until the numerator ball reaches the requested relative
accuracy. Needs `pip install python-flint`.
"""
from __future__ import annotations

from flint import acb, ctx


def _exp_vp(x, coeffs, volume):
    p = acb(0)
    for c in reversed(coeffs):
        p = p * x + acb(c)
    return (acb(volume) * p).exp()


def phase_factor(p_coeffs, volume, s_max, omega=1.0, rel_dps=8, start_dps=30,
                 max_dps=4000):
    """<e^{i omega S}> for symmetric rho with ln rho(sV) = V*P(s).

    omega is the oscillation frequency in the extensive action (e.g.
    sinh(mu) when the complex action is S_R + i sinh(mu) S_I). By symmetry
    the sin part vanishes:
        num = int_0^{s_max} exp(V P(s)) cos(omega V s) ds
        den = int_0^{s_max} exp(V P(s)) ds
    (the V ds Jacobian cancels in the ratio). Returns (value, certified
    relative error, dps used).
    """
    volume = float(volume)
    coeffs = [float(c) for c in p_coeffs]
    freq = float(omega) * volume

    def num_f(x, _):
        return _exp_vp(x, coeffs, volume) * (acb(freq) * x).cos()

    def den_f(x, _):
        return _exp_vp(x, coeffs, volume)

    need_bits = rel_dps * 3.33
    dps_saved, dps = ctx.dps, start_dps
    try:
        while True:
            ctx.dps = dps
            num = acb.integral(num_f, 0, s_max).real
            den = acb.integral(den_f, 0, s_max).real
            if (min(num.rel_accuracy_bits(), den.rel_accuracy_bits()) >= need_bits
                    or dps >= max_dps):
                break
            dps *= 2
        ratio = num / den
        rel = float(ratio.rad() / abs(ratio.mid())) if ratio.mid() != 0 else float("inf")
        return float(ratio.mid()), rel, dps
    finally:
        ctx.dps = dps_saved


def accumulation_scan(p_coeffs, volume, s_max, omega=1.0, n_seg=64, tol=1e-3,
                      rel_dps=8, start_dps=30, max_dps=4000):
    """Locate where the oscillating integral stops accumulating.

    Splits [0, s_max] into n_seg segments, integrates numerator and
    denominator per segment (certified balls, shared escalating precision)
    and prefix-sums them. The oscillating tail past x is rigorously bounded
    by its positive envelope |int_x^{s_max} e^{VP} cos| <= int_x^{s_max}
    e^{VP}; s_star is the first segment edge where that bound drops below
    tol * |num_total|. Returns (s_star, xs, cum_pf) with cum_pf[k] =
    num([0, xs[k]]) / den([0, s_max]) — the manual piece-by-piece
    accumulation, automated.
    """
    volume = float(volume)
    coeffs = [float(c) for c in p_coeffs]
    freq = float(omega) * volume

    def num_f(x, _):
        return _exp_vp(x, coeffs, volume) * (acb(freq) * x).cos()

    def den_f(x, _):
        return _exp_vp(x, coeffs, volume)

    edges = [s_max * k / n_seg for k in range(n_seg + 1)]
    need_bits = rel_dps * 3.33
    dps_saved, dps = ctx.dps, start_dps
    try:
        while True:
            ctx.dps = dps
            num_segs = [acb.integral(num_f, edges[k], edges[k + 1]).real
                        for k in range(n_seg)]
            den_segs = [acb.integral(den_f, edges[k], edges[k + 1]).real
                        for k in range(n_seg)]
            num_total = sum(num_segs[1:], num_segs[0])
            den_total = sum(den_segs[1:], den_segs[0])
            if (min(num_total.rel_accuracy_bits(),
                    den_total.rel_accuracy_bits()) >= need_bits
                    or dps >= max_dps):
                break
            dps *= 2

        threshold = float(abs(num_total.mid())) * tol
        s_star = s_max
        env_tail = 0.0
        for k in range(n_seg, 0, -1):
            env_tail += float(abs(den_segs[k - 1].mid()))
            if env_tail > threshold:
                s_star = edges[k]
                break

        cum, run = [], None
        for seg in num_segs:
            run = seg if run is None else run + seg
            cum.append(float((run / den_total).mid()))
        return s_star, edges[1:], cum
    finally:
        ctx.dps = dps_saved
