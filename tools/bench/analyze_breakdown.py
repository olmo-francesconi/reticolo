#!/usr/bin/env python3
"""Per-component HMC breakdown: throughput, scaling, and trajectory composition.

Reads <dir>/breakdown.csv (base,threads,shape,action,pass,ms,gbps,calls; plus
machine/stream rows for the STREAM ceiling). For each action writes:
  breakdown_<action>_throughput.png  GB/s vs threads, panel/pass, curve/size + ceiling
  breakdown_<action>_speedup.png     t(1)/t(p) vs threads, panel/pass, curve/size + ideal
  breakdown_composition.png          ms/traj = ms·calls stacked by pass, per size (all actions)
and prints a per-action table (ms/call, GB/s, ms/traj, %share) at max threads.
Matplotlib optional — the table still prints without it.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

PASSES = ["refresh", "snapshot", "kinetic", "s_full", "kick", "drift"]


def load(path):
    comp = defaultdict(dict)  # (action, pass, base) -> {threads: {ms, gbps, calls}}
    ceil = defaultdict(dict)  # base -> {threads: gbps}
    with open(path) as f:
        for r in csv.DictReader(f):
            p = int(r["threads"])
            if r["action"] == "machine":
                ceil[r["base"]][p] = float(r["gbps"])
                continue
            comp[(r["action"], r["pass"], r["base"])][p] = {
                "ms": float(r["ms"]), "gbps": float(r["gbps"]), "calls": int(r["calls"])
            }
    return comp, ceil


def actions_of(comp):
    return sorted({a for (a, _, _) in comp})


def bases_of(comp, action):
    return sorted({b for (a, _, b) in comp if a == action}, key=int)


def ceil_curve(ceil):
    # STREAM ceiling is machine-level (per thread count); average over bases.
    agg = defaultdict(list)
    for base in ceil:
        for p, g in ceil[base].items():
            agg[p].append(g)
    return {p: sum(v) / len(v) for p, v in agg.items()}


def table(comp, action):
    print(f"\n== {action} — per-pass at max threads (ms/call, GB/s, ms/traj, %share) ==")
    bases = bases_of(comp, action)
    for base in bases:
        rows = []
        for pas in PASSES:
            series = comp.get((action, pas, base), {})
            if not series:
                continue
            pmax = max(series)
            d = series[pmax]
            rows.append((pas, d["ms"], d["gbps"], d["ms"] * d["calls"]))
        total = sum(r[3] for r in rows) or 1.0
        print(f"  L={base}  (p={max(comp[(action, PASSES[0], base)]) if rows else '?'}):")
        for pas, ms, gbps, mst in rows:
            print(f"    {pas:9s} {ms:9.4f} ms  {gbps:8.2f} GB/s  {mst:9.4f} ms/traj  {100*mst/total:5.1f}%")
        print(f"    {'TOTAL':9s} {'':9s}     {'':8s}       {total:9.4f} ms/traj")


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("\n(matplotlib not found — skipping plots)")
        return None


def plot_grid(comp, ceil, action, metric, out, plt):
    bases = bases_of(comp, action)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    cc = ceil_curve(ceil)
    for i, pas in enumerate(PASSES):
        ax = axes[i // 3][i % 3]
        pmax = 1
        for base in bases:
            series = comp.get((action, pas, base), {})
            if not series:
                continue
            xs = sorted(series)
            pmax = max(pmax, max(xs))
            if metric == "gbps":
                ys = [series[p]["gbps"] for p in xs]
            else:  # speedup from time = ms; t(1)/t(p)
                t1 = series[xs[0]]["ms"]
                ys = [t1 / series[p]["ms"] for p in xs]
            ax.plot(xs, ys, "o-", ms=4, label=f"L={base}")
        xs_all = sorted({p for b in bases for p in comp.get((action, pas, b), {})}) or [1, pmax]
        if metric == "gbps":
            cx = sorted(cc)
            if cx:
                ax.plot(cx, [cc[p] for p in cx], "k--", lw=1, label="STREAM ceiling")
            ax.set_ylabel("GB/s")
        else:
            ax.plot(xs_all, xs_all, "k--", lw=1, label="ideal")
            ax.set_ylabel("speedup t(1)/t(p)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs_all)
        ax.set_xticklabels([str(p) for p in xs_all])
        ax.set_title(f"{action} — {pas}")
        ax.set_xlabel("threads")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"{action} — {'effective throughput' if metric=='gbps' else 'per-pass scaling'}",
                 fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


def plot_composition(comp, out, plt):
    actions = actions_of(comp)
    fig, axes = plt.subplots(1, len(actions), figsize=(7 * len(actions), 5), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for ai, action in enumerate(actions):
        ax = axes[0][ai]
        bases = bases_of(comp, action)
        bottoms = [0.0] * len(bases)
        for pi, pas in enumerate(PASSES):
            vals = []
            for base in bases:
                series = comp.get((action, pas, base), {})
                if series:
                    d = series[max(series)]
                    vals.append(d["ms"] * d["calls"])
                else:
                    vals.append(0.0)
            ax.bar(range(len(bases)), vals, bottom=bottoms, label=pas, color=cmap(pi))
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax.set_xticks(range(len(bases)))
        ax.set_xticklabels([f"L={b}" for b in bases])
        ax.set_ylabel("ms / trajectory (at max threads)")
        ax.set_title(f"{action} — trajectory composition")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    comp, ceil = load(d / "breakdown.csv")
    for action in actions_of(comp):
        table(comp, action)
    plt = _mpl()
    if plt is None:
        return
    for action in actions_of(comp):
        slug = action.lower().replace("<", "_").replace(">", "").replace(" ", "")
        plot_grid(comp, ceil, action, "gbps", d / f"breakdown_{slug}_throughput.png", plt)
        plot_grid(comp, ceil, action, "speedup", d / f"breakdown_{slug}_speedup.png", plt)
    plot_composition(comp, d / "breakdown_composition.png", plt)


if __name__ == "__main__":
    main()
