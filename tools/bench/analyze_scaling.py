#!/usr/bin/env python3
"""Turn scaling_{strong,weak}.csv into per-size speedup/efficiency curves.

strong: speedup(p)    = t(1)/t(p),   ideal = p   (efficiency = speedup/p)
weak:   efficiency(p) = t(1)/t(p),   ideal = 1   (work/thread held constant)

Each CSV carries a `base` column (strong: cube edge L; weak: per-thread base), so
a panel plots one curve per size. Writes strong_scaling.png + weak_scaling.png
(grid: action rows × op cols) alongside the CSVs and prints a text summary.
Matplotlib optional — without it the table still prints.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

OPS = [("force_ms", "compute_force"), ("sfull_ms", "s_full"), ("traj_ms", "trajectory")]


def load(path):
    # rows[(action, op_key, base)] = {threads: ms}
    rows = defaultdict(dict)
    if not path.exists():
        return rows
    with open(path) as f:
        for r in csv.DictReader(f):
            p = int(r["threads"])
            for key, _ in OPS:
                v = float(r[key])
                if v > 0:
                    rows[(r["action"], key, r["base"])][p] = v
    return rows


def ratios(series):
    ps = sorted(series)
    if not ps or series[ps[0]] <= 0:
        return []
    t1 = series[ps[0]]
    return [(p, t1 / series[p]) for p in ps]


def actions_of(rows):
    return sorted({a for (a, _, _) in rows})


def bases_of(rows, action, key):
    return sorted({b for (a, k, b) in rows if a == action and k == key}, key=int)


def table(name, rows):
    ideal = "speedup t1/tp, %=eff" if name == "strong" else "eff t1/tp, ideal 1.0"
    print(f"\n== {name} scaling ({ideal}) ==")
    for action in actions_of(rows):
        for key, opname in OPS:
            for base in bases_of(rows, action, key):
                rr = ratios(rows[(action, key, base)])
                if not rr:
                    continue
                cells = "  ".join(
                    f"p={p}:{r:5.2f}x" + (f"({100*r/p:3.0f}%)" if name == "strong" else "")
                    for p, r in rr
                )
                tag = f"L={base}" if name == "strong" else f"base={base}"
                print(f"  {action:12s} {opname:13s} {tag:8s} {cells}")


def plot_mode(name, rows, out):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not found — skipping plots)")
        return
    actions = actions_of(rows)
    if not actions:
        return
    nrow, ncol = len(actions), len(OPS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False)
    ylabel = "speedup" if name == "strong" else "efficiency"
    for ai, action in enumerate(actions):
        for ci, (key, opname) in enumerate(OPS):
            ax = axes[ai][ci]
            pmax = 1
            for base in bases_of(rows, action, key):
                rr = ratios(rows[(action, key, base)])
                if not rr:
                    continue
                xs, ys = zip(*rr)
                pmax = max(pmax, max(xs))
                lbl = f"L={base}" if name == "strong" else f"base={base}⁴"
                ax.plot(xs, ys, "o-", ms=4, label=lbl)
            ideal_x = sorted(
                {p for (a, k, b) in rows if a == action and k == key for p in rows[(a, k, b)]}
            ) or [1, pmax]
            if name == "strong":
                ax.plot(ideal_x, ideal_x, "k--", lw=1, label="ideal")
                ax.set_ylim(bottom=0)
            else:
                ax.axhline(1.0, color="k", ls="--", lw=1, label="ideal")
                ax.set_ylim(0, 1.15)
            ax.set_xscale("log", base=2)
            ax.set_xticks(ideal_x)
            ax.set_xticklabels([str(p) for p in ideal_x])
            ax.set_title(f"{action} — {opname}")
            ax.set_xlabel("threads")
            ax.set_ylabel(ylabel)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle(f"{name} scaling", fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


def schedule(path):
    """Print the thread count the policy actually picks per (action, size) as the
    OMP ceiling rises — i.e. where threading kicks in and how it ramps. Reads the
    optional mb/nthr columns; silently skips a CSV written before they existed."""
    if not path.exists():
        return
    info = {}  # (action, base) -> {"mb": float, "nthr": {p: n}}
    with open(path) as f:
        for r in csv.DictReader(f):
            if not r.get("nthr"):
                return
            d = info.setdefault((r["action"], r["base"]), {"mb": float(r["mb"]), "nthr": {}})
            d["nthr"][int(r["threads"])] = int(r["nthr"])
    if not info:
        return
    print("\n== thread-spawn schedule (nthr the policy picks at each OMP ceiling) ==")
    for action, base in sorted(info, key=lambda k: (k[0], int(k[1]))):
        d = info[(action, base)]
        cells = "  ".join(f"p{p}:{d['nthr'][p]}" for p in sorted(d["nthr"]))
        print(f"  {action:12s} L={base:<3s} {d['mb']:8.2f} MB   {cells}")


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    for name, fname in (("strong", "scaling_strong.csv"), ("weak", "scaling_weak.csv")):
        rows = load(d / fname)
        table(name, rows)
        plot_mode(name, rows, d / f"{name}_scaling.png")
    schedule(d / "scaling_strong.csv")


if __name__ == "__main__":
    main()
