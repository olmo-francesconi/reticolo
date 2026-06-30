#!/usr/bin/env python3
"""Cross-GPU comparison of profile_cuda_hmc bench sweeps.

    python tools/profile/compare.py tools/modal/output/<session> [--baseline T4]
    python tools/profile/compare.py <run_dir1> <run_dir2> ...

Each run dir holds throughput.jsonl + meta.json (meta['gpu'] = the nvidia-smi csv
"name, compute_cap, driver"). Rows are joined on (action, L); the tool prints

  * HMC throughput (traj/s) per GPU,
  * speedup vs the baseline GPU,
  * force-kernel effective bandwidth (eff_GBps) per GPU,
  * the largest lattice each GPU sustained before OOM (the "pushed volume"),

and writes plots to <session>/plots/. `pct_peak` is ignored (the binary hardcodes
P100's 732 GB/s); absolute `eff_GBps` is used for cross-GPU bandwidth.
"""
import argparse
import json
from pathlib import Path

GPU_ORDER = ["T4", "L4", "A10", "L40S", "A100", "H100", "H200", "B200"]


def gpu_label(meta_path):
    g = json.loads(Path(meta_path).read_text()).get("gpu", "")
    name = g.split(",")[0].strip()
    # Longest match first so "A100" wins over "A10" (and "L40S" over "L4").
    for k in sorted(GPU_ORDER, key=len, reverse=True):
        if k in name:
            return k
    return name.replace("NVIDIA ", "").replace("Tesla ", "").strip() or "?"


def load_rows(jsonl_path):
    return [json.loads(ln) for ln in Path(jsonl_path).read_text().splitlines()
            if ln.strip().startswith("{")]


def discover_runs(paths):
    runs = []
    for p in map(Path, paths):
        if (p / "throughput.jsonl").exists():
            runs.append(p)
        elif p.is_dir():
            runs += [m.parent for m in sorted(p.rglob("throughput.jsonl"))]
    return list(dict.fromkeys(runs))


def order_gpus(labels):
    seen = list(dict.fromkeys(labels))
    known = [g for g in GPU_ORDER if g in seen]
    return known + sorted(set(seen) - set(known))


def build_matrix(runs):
    """-> (gpus, hmc, bw, vmap, maxL, hrow). hmc/bw: {(action,L): {gpu: value}};
    vmap: {(action,L): V}; maxL: {(action,gpu): largest L}; hrow: {(action,L):
    {gpu: full hmc row}} (carries dof + us_* step breakdown)."""
    hmc, bw, vmap, maxL, hrow, labels = {}, {}, {}, {}, {}, []
    for d in runs:
        gpu = gpu_label(d / "meta.json")
        labels.append(gpu)
        for r in load_rows(d / "throughput.jsonl"):
            if r.get("status") == "oom" or r.get("action") is None or r.get("L") is None:
                continue
            key = (r["action"], int(r["L"]))
            if "V" in r:
                vmap[key] = int(r["V"])
            if r.get("mode") == "hmc" and "traj_per_s" in r:
                hmc.setdefault(key, {})[gpu] = r["traj_per_s"]
                hrow.setdefault(key, {})[gpu] = r
            elif r.get("mode") == "force" and "eff_GBps" in r:
                bw.setdefault(key, {})[gpu] = r["eff_GBps"]
            mk = (r["action"], gpu)
            maxL[mk] = max(maxL.get(mk, 0), int(r["L"]))
    return order_gpus(labels), hmc, bw, vmap, maxL, hrow


def _print_grid(headers, rows):
    cols = list(zip(*([headers] + rows))) if rows else [[h] for h in headers]
    widths = [max(len(str(c)) for c in col) for col in cols]
    line = lambda r: "  ".join(str(c).ljust(w) for c, w in zip(r, widths))
    print(line(headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(line(r))


def cell(v, fmt="{:,.1f}"):
    return fmt.format(v) if isinstance(v, (int, float)) else "—"


def print_value_table(title, mat, gpus):
    print(f"\n=== {title} ===")
    rows = [[a, str(L)] + [cell(mat[(a, L)].get(g)) for g in gpus]
            for (a, L) in sorted(mat)]
    _print_grid(["action", "L"] + gpus, rows)


def print_speedup_table(hmc, gpus, baseline):
    if baseline not in gpus:
        print(f"\n(no baseline '{baseline}' in data — skipping speedup table)")
        return
    print(f"\n=== HMC speedup vs {baseline} (traj/s ratio) ===")
    rows = []
    for (a, L) in sorted(hmc):
        cells = hmc[(a, L)]
        base = cells.get(baseline)
        vals = [cell(cells[g] / base, "{:.2f}x") if base and g in cells else "—"
                for g in gpus]
        rows.append([a, str(L)] + vals)
    _print_grid(["action", "L"] + gpus, rows)


def print_frontier(maxL, gpus, actions):
    print("\n=== max lattice L sustained (OOM frontier) ===")
    rows = [[a] + [str(maxL.get((a, g), "—")) for g in gpus] for a in actions]
    _print_grid(["action"] + gpus, rows)


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("\n(matplotlib not available — skipping plots)")
        return None


# Per-trajectory call counts for the HMC step kernels: force runs n_md times,
# the drift/kick axpy 2*n_md (momentum + position), s_full twice (begin/end
# Hamiltonian), sample once. us_* are per-call microseconds.
def _step_ms(r):
    nmd = r.get("n_md", 1)
    return {
        "force":  r.get("us_force", 0.0) * nmd / 1e3,
        "drift":  r.get("us_axpy", 0.0) * 2 * nmd / 1e3,
        "s_full": r.get("us_sfull", 0.0) * 2 / 1e3,
        "sample": r.get("us_sample", 0.0) / 1e3,
    }


def plot_step_breakdown(hrow, gpus, actions, outdir):
    plt = _plt()
    if plt is None:
        return
    comps = ["force", "drift", "s_full", "sample"]
    fig, axes = plt.subplots(1, len(actions), figsize=(6 * len(actions), 5), squeeze=False)
    for ax, action in zip(axes[0], actions):
        shared = [L for (a, L) in hrow if a == action
                  and all(g in hrow[(a, L)] for g in gpus)]
        if not shared:
            ax.set_title(f"{action}: no L common to all GPUs")
            continue
        L = max(shared)
        bottoms = [0.0] * len(gpus)
        for c in comps:
            vals = [_step_ms(hrow[(action, L)][g])[c] for g in gpus]
            ax.bar(gpus, vals, bottom=bottoms, label=c)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax.set(title=f"{action}  L={L}", ylabel="ms / trajectory")
        ax.legend()
    fig.suptitle("HMC per-trajectory step breakdown (force / drift / s_full / sample)")
    fig.tight_layout()
    fig.savefig(outdir / "step_breakdown.png", dpi=120)
    plt.close(fig)
    print(f"  wrote {outdir / 'step_breakdown.png'}")


def plot_dof_throughput(hrow, vmap, gpus, actions, outdir):
    plt = _plt()
    if plt is None:
        return
    fig, axes = plt.subplots(1, len(actions), figsize=(6 * len(actions), 5), squeeze=False)
    for ax, action in zip(axes[0], actions):
        for g in gpus:
            pts = sorted((vmap.get((a, L), L), row[g]["dof"] * row[g]["traj_per_s"])
                         for (a, L), row in hrow.items() if a == action and g in row)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=g)
        ax.set(title=action, xlabel="V (sites)", ylabel="dof updated / s  (dof x traj/s)",
               xscale="log", yscale="log")
        ax.grid(True, which="both", alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    fig.suptitle("Throughput in dof/s vs lattice volume — flat plateau = linear (time ∝ V) scaling")
    fig.tight_layout()
    fig.savefig(outdir / "dof_throughput.png", dpi=120)
    plt.close(fig)
    print(f"  wrote {outdir / 'dof_throughput.png'}")


def make_plots(hmc, bw, vmap, gpus, actions, outdir, baseline):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available — skipping plots)")
        return
    outdir.mkdir(parents=True, exist_ok=True)

    def series(mat, action, gpu):
        pts = sorted((vmap.get((action, L), L), v[gpu])
                     for (a, L), v in mat.items() if a == action and gpu in v)
        return [p[0] for p in pts], [p[1] for p in pts]

    for mat, ylabel, fname in [(hmc, "traj/s", "throughput_by_gpu.png"),
                               (bw, "eff GB/s", "bandwidth_by_gpu.png")]:
        if not mat:
            continue
        fig, axes = plt.subplots(1, len(actions), figsize=(6 * len(actions), 5), squeeze=False)
        for ax, action in zip(axes[0], actions):
            for gpu in gpus:
                xs, ys = series(mat, action, gpu)
                if xs:
                    ax.plot(xs, ys, marker="o", label=gpu)
            ax.set(title=action, xlabel="V (sites)", ylabel=ylabel, xscale="log", yscale="log")
            ax.grid(True, which="both", alpha=0.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=120)
        plt.close(fig)
        print(f"  wrote {outdir / fname}")

    # speedup bars at the largest L each action shares with the baseline
    if hmc and baseline in gpus:
        fig, axes = plt.subplots(1, len(actions), figsize=(6 * len(actions), 5), squeeze=False)
        for ax, action in zip(axes[0], actions):
            shared = [L for (a, L) in hmc if a == action
                      and baseline in hmc[(a, L)]
                      and all(g in hmc[(a, L)] for g in gpus)]
            if not shared:
                ax.set_title(f"{action}: no L common to all GPUs")
                continue
            L = max(shared)
            base = hmc[(action, L)][baseline]
            ax.bar(gpus, [hmc[(action, L)][g] / base for g in gpus])
            ax.set(title=f"{action} L={L}", ylabel=f"speedup vs {baseline}")
            ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "speedup.png", dpi=120)
        plt.close(fig)
        print(f"  wrote {outdir / 'speedup.png'}")


def main():
    ap = argparse.ArgumentParser(description="cross-GPU profile_cuda_hmc comparison")
    ap.add_argument("paths", nargs="+", help="a session dir or explicit run dirs")
    ap.add_argument("--baseline", default="T4", help="GPU label for speedup ratios")
    ap.add_argument("--out", help="plot dir (default: <first path>/plots)")
    a = ap.parse_args()

    runs = discover_runs(a.paths)
    if not runs:
        ap.error("no run dirs with throughput.jsonl found under " + " ".join(a.paths))
    gpus, hmc, bw, vmap, maxL, hrow = build_matrix(runs)
    actions = sorted({a_ for (a_, _) in {**hmc, **bw}})
    print(f"GPUs: {gpus}   runs: {len(runs)}   actions: {actions}")

    print_value_table("HMC throughput (traj/s)", hmc, gpus)
    print_speedup_table(hmc, gpus, a.baseline)
    print_value_table("force-kernel bandwidth (eff GB/s)", bw, gpus)
    print_frontier(maxL, gpus, actions)

    outdir = Path(a.out) if a.out else Path(a.paths[0]) / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    print("\nplots:")
    make_plots(hmc, bw, vmap, gpus, actions, outdir, a.baseline)
    plot_step_breakdown(hrow, gpus, actions, outdir)
    plot_dof_throughput(hrow, vmap, gpus, actions, outdir)


if __name__ == "__main__":
    main()
