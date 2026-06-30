#!/usr/bin/env python3
"""Analyse a Nsight profiling sweep (nsys CSVs + throughput.jsonl).

Run on the Mac after downloading the Kaggle `profile/` artifacts:

    python tools/profile/analyze.py <profile_dir>

Reads throughput.jsonl + the per-config nsys kernel/mem CSVs (+ ncu CSVs if the
GPU allowed perf counters) and writes plots + a printed summary to
<profile_dir>/plots/. The raw .nsys-rep / .ncu-rep open directly in the free
Nsight Systems / Nsight Compute GUIs for the interactive timeline / roofline.
"""
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PEAK_GBPS = 732.0  # Tesla P100 HBM2

# Map a demangled kernel name to a coarse category for the breakdown.
CATEGORIES = [
    ("force", re.compile(r"stencil_kernel|su_plaq_force")),
    ("action", re.compile(r"reduce_fwd_site|su_plaq_energy")),
    ("reduce", re.compile(r"reduce_(sum|sumsq|partial|final)|block_reduce")),
    ("drift/kick", re.compile(r"axpy|su_expi_lmul")),  # su(N) drift is a group exp, not axpy
    ("sample", re.compile(r"fill_normals|su_sample_algebra")),
    ("accept", re.compile(r"mh_accept|resolve|bump_counter")),
]


def categorize(name):
    for cat, rx in CATEGORIES:
        if rx.search(name):
            return cat
    return "other"


def load_throughput(d):
    rows = []
    p = d / "throughput.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line.startswith("{"):
                rows.append(json.loads(line))
    return rows


def _find_col(header, *cands):
    for i, h in enumerate(header):
        hl = h.strip().lower()
        for c in cands:
            if c in hl:
                return i
    return None


def load_kernel_csv(path):
    """Return {category: total_ns} from an nsys cuda_gpu_kern_sum CSV."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    # nsys CSVs sometimes carry a preamble; the header row has a 'Name' column.
    hdr_i = next((i for i, r in enumerate(rows) if any("name" == c.strip().lower() for c in r)), None)
    if hdr_i is None:
        return {}
    header = rows[hdr_i]
    name_i = _find_col(header, "name")
    time_i = _find_col(header, "total time")
    if name_i is None or time_i is None:
        return {}
    out = {}
    for r in rows[hdr_i + 1 :]:
        if len(r) <= max(name_i, time_i) or not r[name_i].strip():
            continue
        try:
            ns = float(r[time_i].replace(",", ""))
        except ValueError:
            continue
        out[categorize(r[name_i])] = out.get(categorize(r[name_i]), 0.0) + ns
    return out


def main(argv):
    if len(argv) < 2:
        print("usage: analyze.py <profile_dir>", file=sys.stderr)
        return 2
    d = Path(argv[1])
    plots = d / "plots"
    plots.mkdir(exist_ok=True)

    tools = d / "tools.txt"
    if tools.exists():
        print(tools.read_text())

    thru = load_throughput(d)
    hmc = [r for r in thru if r.get("mode") == "hmc"]
    force = [r for r in thru if r.get("mode") == "force"]

    # ---- throughput scaling ----
    print("\n=== HMC throughput (ms/traj) ===")
    print(f"{'action':>6} {'L':>4} {'V':>10} {'dof':>12} {'ms/traj':>10} {'traj/s':>10}")
    for r in sorted(hmc, key=lambda x: (x["action"], x["V"])):
        print(f"{r['action']:>6} {r['L']:>4} {r['V']:>10} {int(r['dof']):>12} "
              f"{r['ms_per_traj']:>10.4f} {r['traj_per_s']:>10.2f}")

    if hmc:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for act in sorted({r["action"] for r in hmc}):
            rs = sorted([r for r in hmc if r["action"] == act], key=lambda x: x["V"])
            vs = [r["V"] for r in rs]
            a1.loglog(vs, [r["ms_per_traj"] for r in rs], "o-", label=act)
            a2.loglog(vs, [r["dof"] * r["traj_per_s"] for r in rs], "o-", label=act)
        a1.set(xlabel="V (sites)", ylabel="ms / trajectory", title="HMC cost vs volume")
        a2.set(xlabel="V (sites)", ylabel="dof · traj/s", title="throughput (dof-evals/s)")
        for a in (a1, a2):
            a.grid(True, which="both", alpha=0.3)
            a.legend()
        fig.tight_layout()
        fig.savefig(plots / "throughput.png", dpi=130)
        print(f"\nwrote {plots/'throughput.png'}")

    # ---- effective force-kernel bandwidth (streaming model) ----
    if force:
        print("\n=== force kernel — effective bandwidth (streaming model) ===")
        print(f"{'action':>6} {'L':>4} {'us/force':>10} {'eff GB/s':>10} {'% peak':>8}")
        for r in sorted(force, key=lambda x: (x["action"], x["V"])):
            print(f"{r['action']:>6} {r['L']:>4} {r['us_per_force']:>10.3f} "
                  f"{r['eff_GBps']:>10.1f} {r['pct_peak']:>8.1f}")

    # ---- kernel breakdown from the in-binary CUDA-event timings (JSON) ----
    # Per-trajectory call counts: force ×n_md, axpy ×2·n_md (kick+drift),
    # s_full ×2 (H before/after), sample ×1. (nsys/ncu are unavailable on the
    # managed GPU host, so this event-based split is the portable substitute.)
    if any("us_force" in r for r in hmc):
        print("\n=== kernel time breakdown (CUDA events, weighted per trajectory) ===")
        hdr = f"{'action':>6} {'L':>4} | {'force':>7} {'axpy':>7} {'s_full':>7} {'sample':>7} | {'acct ms':>8} {'wall ms':>8}"
        print(hdr)
        fig2, ax2 = plt.subplots(figsize=(max(8, 0.8 * len(hmc)), 5))
        keys2, segs = [], {"force": [], "axpy": [], "s_full": [], "sample": []}
        for r in sorted(hmc, key=lambda x: (x["action"], x["V"])):
            nmd = r.get("n_md", 10)
            comp = {
                "force": nmd * r.get("us_force", 0.0),
                "axpy": 2 * nmd * r.get("us_axpy", 0.0),
                "s_full": 2 * r.get("us_sfull", 0.0),
                "sample": r.get("us_sample", 0.0),
            }
            acct = sum(comp.values()) / 1e3  # µs → ms (accounted)
            tot = sum(comp.values()) or 1.0
            print(f"{r['action']:>6} {r['L']:>4} | "
                  + " ".join(f"{100*comp[c]/tot:6.1f}%" for c in segs)
                  + f" | {acct:8.3f} {r['ms_per_traj']:8.3f}")
            keys2.append(f"{r['action']}\nL{r['L']}")
            for c in segs:
                segs[c].append(100 * comp[c] / tot)
        bottoms = [0.0] * len(keys2)
        for c in segs:
            ax2.bar(keys2, segs[c], bottom=bottoms, label=c)
            bottoms = [x + y for x, y in zip(bottoms, segs[c])]
        ax2.set(ylabel="% of accounted kernel time", title="per-config kernel breakdown (CUDA events)")
        ax2.legend(ncol=4, fontsize=8)
        fig2.tight_layout()
        fig2.savefig(plots / "kernel_breakdown_events.png", dpi=130)
        print(f"\nwrote {plots/'kernel_breakdown_events.png'}")
        print("('acct ms' = sum of weighted isolated atom times; 'wall ms' = graph "
              "trajectory. acct < wall is expected — the graph adds per-launch + "
              "memcpy(old<-field) overhead the isolated atoms exclude.)")

    # ---- kernel breakdown (nsys, if available) ----
    breakdowns = {}
    for p in sorted(d.glob("kern_*.csv")):
        m = re.match(r"kern_(\w+?)_L(\d+)\.csv", p.name)
        if not m:
            continue
        breakdowns[(m.group(1), int(m.group(2)))] = load_kernel_csv(p)

    if breakdowns:
        print("\n=== kernel time breakdown (% of kernel time, from nsys) ===")
        cats = ["force", "action", "reduce", "drift/kick", "sample", "accept", "other"]
        for key in sorted(breakdowns):
            b = breakdowns[key]
            tot = sum(b.values()) or 1.0
            parts = "  ".join(f"{c}:{100*b.get(c,0)/tot:4.1f}%" for c in cats if b.get(c, 0) > 0)
            print(f"{key[0]:>6} L={key[1]:<3}  {parts}")

        # stacked bars, one column per config
        keys = sorted(breakdowns)
        fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(keys)), 5))
        bottoms = [0.0] * len(keys)
        for c in cats:
            vals = []
            for k in keys:
                b = breakdowns[k]
                tot = sum(b.values()) or 1.0
                vals.append(100 * b.get(c, 0) / tot)
            if any(v > 0 for v in vals):
                ax.bar([f"{a}\nL{l}" for a, l in keys], vals, bottom=bottoms, label=c)
                bottoms = [x + y for x, y in zip(bottoms, vals)]
        ax.set(ylabel="% of kernel time", title="per-config kernel breakdown (nsys)")
        ax.legend(ncol=4, fontsize=8)
        fig.tight_layout()
        fig.savefig(plots / "kernel_breakdown.png", dpi=130)
        print(f"\nwrote {plots/'kernel_breakdown.png'}")

    # ---- ncu real DRAM throughput, if it ran ----
    ncu = sorted(d.glob("ncu_*.csv"))
    if ncu:
        print("\n=== ncu measured metrics (force kernel) ===")
        for p in ncu:
            if p.stat().st_size == 0:
                continue
            print(f"-- {p.name} --")
            with open(p, newline="") as f:
                for row in csv.reader(f):
                    line = ",".join(row)
                    if re.search(r"dram__throughput|sm__throughput|occupancy|Duration", line, re.I):
                        print("   " + line)
    else:
        print("\n(ncu produced no CSV — perf counters likely locked on the GPU host; "
              "rely on nsys timings + the streaming-model bandwidth above)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
