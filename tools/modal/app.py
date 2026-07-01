# /// script
# requires-python = ">=3.12"
# dependencies = ["modal>=1.0"]
# ///
"""Modal GPU runner for the reticolo CUDA backend — a self-contained uv script.

Dependencies are declared inline (PEP 723) so uv runs it with no setup and no
wrappers. Pinned to the reticolo-cuda Modal environment. One-time auth:
`uvx modal token new`.

    uv run tools/modal/app.py build --gpu H100 --cpu 16
    uv run tools/modal/app.py run --app phi4_cuda_hmc --args "--L 16 --n_prod 500 --out phi4.h5"
    uv run tools/modal/app.py run --app profile_cuda_hmc --args "--action=su3 --size=16"
    uv run tools/modal/app.py bench --gpus T4,L4,A100,H100 --name multigpu  # cross-GPU sweep
    uv run tools/modal/app.py pull            # list runs on the Volume
    uv run tools/modal/app.py pull <run_id>   # fetch a run's artifacts
    uv run tools/modal/app.py pull --session <session>   # fetch a whole bench session

Commands:
  build  cmake --preset / --build --preset / ctest --preset linux-nvcc — the gate;
         builds the whole suite incl. bench_cuda_hmc, profile_cuda_hmc, cuda_nightly.
  run    (re)build ONE target, run it in a temp dir, export its output to the run
         folder — also how you run the bench / profile / nightly binaries.
  pull   download out/<run_id>/ from the Volume into tools/modal/output/.

One image (toolchain) + one persistent Volume (build cache). Each invocation gets
a unique id YYYY-MM-DD-<HHMMSS>-<name> and its own dir on the Volume under
out/<run_id>/ — console.log + meta.json always, plus (for `run`) the app's files
under data/. Builds are incremental: the toolchain is a cached image layer, and
the CMake tree (build/linux-nvcc) + ccache live on the Volume.
"""
import argparse
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MODAL_ENVIRONMENT", "reticolo-cuda")

import modal

# Repo root = two levels above tools/modal/app.py — used only locally for the
# source mount. In the container __file__ is /root/app.py (shallow), where this
# value is unused (the image is already built), so fall back instead of indexing.
_self = Path(__file__).resolve()
REPO = _self.parents[2] if len(_self.parents) > 2 else _self.parent
OUTPUT = _self.parent / "output"
VOL = "reticolo-cuda-cache"
BUILD = "build/linux-nvcc"   # the linux-nvcc preset's binaryDir (relative to repo root)

# Multi-GPU benchmark grids. profile_cuda_hmc climbs each list until a config OOMs
# (cudaMalloc throws), then stops that action's climb. phi4 (f64) ~40 B/site; su3
# Wilson (f64) link fields are ~72x heavier per site, so its grid tops out far
# lower. Grids overshoot the small GPUs on purpose — the OOM cutoff is the signal.
BENCH_GRID = {
    "phi4": [16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192],
    "su3":  [8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72],
}
BENCH_ACTIONS = ["phi4", "su3"]
# The bench workload is f64 (phi4/su3), so default to GPUs with real 1:2 FP64
# (A100/H100/H200/B200). T4 (1:32) and L4 (1:64) throttle FP64 and are misleading
# here; they stay in the arch map only for non-f64 use.
GPU_ARCH = {"T4": "75", "L4": "89", "A100": "80", "H100": "90", "H200": "90", "B200": "100"}
BENCH_FP64_GPUS = "A100,H100,H200,B200"

# LLR scaling stress matrix (phi4_llr_cuda). Per volume L (4D, kappa=0.20 lambda=1.0):
# n_md tuned for acceptance, and (mean_S, sigma_S) measured from a thermalised HMC.
# The S-window is <S> +/- 2 sigma; the Gaussian width is fixed at delta = sigma(V);
# n_rep is set by --spacing = 4 sigma / (N-1), so only the replica count varies.
STRESS_VOLS = {8: (20, -211.0, 56.0), 12: (30, -1106.0, 132.0), 16: (50, -3706.0, 234.0)}
STRESS_NREPS = [8, 16, 32, 64, 128]

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])                                   # CUDA image ships a bad default
    .apt_install("git", "ninja-build", "libhdf5-dev", "ccache")
    .pip_install("cmake>=3.25")
    .add_local_dir(                                    # working tree, fresh each run
        str(REPO), "/root/reticolo", copy=False,
        ignore=["**/.git", "**/build", "tools/modal/output", "**/*.h5", "**/.DS_Store"],
    )
)

app = modal.App("reticolo-cuda")
cache = modal.Volume.from_name(VOL, create_if_missing=True)


def _container_setup(run_id, arch, jobs, meta):
    """Shared container prep: symlink the build tree onto the Volume, set the
    ccache/JOBS/arch env, stamp meta.json with the nvidia-smi GPU id. Returns the
    run's output dir. Runs inside the container."""
    import json
    import subprocess

    out = f"/cache/out/{run_id}"
    os.makedirs(out, exist_ok=True)
    # Per-arch build tree on the Volume: each concrete CUDA arch gets its own
    # CMake cache + objects, so switching GPUs (T4 sm_75 → A100 sm_80) can never
    # link a stale wrong-arch cubin ("named symbol not found" / SIGILL). "native"
    # (unknown GPU) falls back to a shared dir. Point the preset's binaryDir
    # (build/) at it so the tree + ccache persist across runs; add_local_dir
    # ignores **/build, so it is free.
    buildroot = f"/cache/build-{arch}" if arch and arch != "native" else "/cache/build"
    os.makedirs(buildroot, exist_ok=True)
    subprocess.run(f"ln -sfn {buildroot} /root/reticolo/build", shell=True, check=True)
    # add_local_dir normalizes mounted-source mtimes, so ninja compares the fresh
    # sources against the newer cached objects on the Volume and skips the rebuild
    # ("ninja: no work to do" → stale binary). Touch the tree so ninja re-evaluates;
    # ccache is content-hashed, so unchanged TUs stay cache hits and only real edits
    # actually recompile.
    subprocess.run(
        "find /root/reticolo/include /root/reticolo/src /root/reticolo/apps "
        "/root/reticolo/tests -type f -exec touch {} +", shell=True, check=True)
    os.environ.update({
        "PATH": "/usr/local/cuda/bin:/usr/lib/ccache:" + os.environ["PATH"],
        "CCACHE_DIR": "/cache/ccache",
        "CMAKE_CXX_COMPILER_LAUNCHER": "ccache",
        "CMAKE_CUDA_COMPILER_LAUNCHER": "ccache",
        "OUT_DIR": out,
        "JOBS": str(jobs),                             # build -j
    })
    if arch and arch != "native":                      # else fall back to the preset's native
        os.environ["RETICOLO_CUDA_ARCH"] = arch

    gpu = subprocess.run(
        "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader",
        shell=True, capture_output=True, text=True).stdout.strip()
    print(f"run_id={run_id}  gpu={gpu}  cores={os.cpu_count()}  jobs={jobs}", flush=True)
    with open(f"{out}/meta.json", "w") as f:
        json.dump({**(meta or {}), "run_id": run_id, "gpu": gpu}, f, indent=2)
    cache.commit()                                     # land meta before the (long) work
    return out


@app.function(image=IMAGE, gpu="T4", volumes={"/cache": cache}, timeout=3600)
def _exec(script: str, run_id: str, arch: str = "native",
          jobs: int = 8, meta: dict | None = None):
    import subprocess
    out = _container_setup(run_id, arch, jobs, meta)
    # stdbuf -oL sets LD_PRELOAD=libstdbuf, inherited by children, so long binaries
    # stream live instead of buffering until exit. Tee into the run dir so a failed
    # run still leaves a log.
    subprocess.run(["stdbuf", "-oL", "-eL", "bash", "-c",
                    f'set -o pipefail; ({script}) 2>&1 | tee "{out}/console.log"'],
                   cwd="/root/reticolo", check=True)
    cache.commit()                                     # persist build tree + ccache + artifacts


@app.function(image=IMAGE, gpu="T4", volumes={"/cache": cache}, timeout=7200)
def _stress_exec(run_id: str, vols: dict, nreps: list,
                 arch: str = "native", jobs: int = 8, meta: dict | None = None):
    """LLR scaling stress: build phi4_llr_cuda once, sweep (volume, n_rep). Per
    cell the app runs to completion; wall time comes from the app's own elapsed
    stamp (last HHH:MM:SS.mmm line). delta = sigma(V) is held fixed, n_rep set via
    --spacing, so only the replica count varies. One JSON line per cell → stress.jsonl."""
    import json
    import re
    import subprocess
    import time

    out = _container_setup(run_id, arch, jobs, meta)
    logf = open(f"{out}/console.log", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    arch_flag = f"-DCMAKE_CUDA_ARCHITECTURES={arch}" if arch and arch != "native" else ""
    subprocess.run(f"cmake --preset linux-nvcc {arch_flag}", shell=True,
                   cwd="/root/reticolo", check=True)
    subprocess.run(f"cmake --build --preset linux-nvcc --target phi4_llr_cuda -j {jobs}",
                   shell=True, cwd="/root/reticolo", check=True)
    binpath = subprocess.run(
        f'find {BUILD} -name phi4_llr_cuda -type f -perm -u+x | head -1',
        shell=True, cwd="/root/reticolo", capture_output=True, text=True).stdout.strip()
    if not binpath:
        raise RuntimeError("phi4_llr_cuda produced no binary")

    sched = dict(n_nr=1, n_therm_nr=30, n_meas_nr=100, n_rm=4, n_therm_rm=30, n_meas_rm=100)
    traj_per_rep = (sched["n_nr"] * (sched["n_therm_nr"] + sched["n_meas_nr"])
                    + sched["n_rm"] * (sched["n_therm_rm"] + sched["n_meas_rm"]))
    ts = re.compile(r"(\d{3}):(\d\d):(\d\d\.\d+)")  # log stamp HHH:MM:SS.mmm

    def app_elapsed(text):
        last = None
        for m in ts.finditer(text):
            last = m
        if not last:
            return None
        h, mn, s = last.groups()
        return (int(h) * 3600) + (int(mn) * 60) + float(s)

    jsonl = open(f"{out}/stress.jsonl", "w")
    for L, (n_md, mean_s, sigma) in sorted(vols.items()):
        emin, emax = mean_s - (2 * sigma), mean_s + (2 * sigma)
        for N in nreps:
            spacing = (emax - emin) / (N - 1)
            args = ([f"--size={L}", "--ndim=4", "--kappa=0.20", "--lambda=1.0",
                     "--tau=1.0", f"--n_md={n_md}", f"--E_min={emin}", f"--E_max={emax}",
                     f"--delta={sigma}", f"--spacing={spacing}"]
                    + [f"--{k}={v}" for k, v in sched.items()]
                    + ["--seed=20260701", "--workspace=/tmp/stress", "--out=s.h5"])
            t0 = time.time()
            r = subprocess.run([f"/root/reticolo/{binpath}", *args],
                               cwd="/root/reticolo", capture_output=True, text=True)
            wall = time.time() - t0
            ae = app_elapsed(r.stdout)
            traj = N * traj_per_rep
            rec = {"L": L, "sites": L ** 4, "N": N, "n_md": n_md, "delta": sigma,
                   "spacing": round(spacing, 4), "rc": r.returncode,
                   "wall_s": round(wall, 3), "app_s": round(ae, 3) if ae else None,
                   "traj": traj,
                   "traj_per_s": round(traj / ae, 1) if ae else None,
                   "mupd_per_s": round(traj * (L ** 4) * n_md / ae / 1e6, 1) if ae else None}
            jsonl.write(json.dumps(rec) + "\n")
            jsonl.flush()
            log(f"  L={L}^4 N={N:>3}: rc={r.returncode} app={rec['app_s']}s "
                f"traj/s={rec['traj_per_s']} Mupd/s={rec['mupd_per_s']}")
            if r.returncode != 0:
                log(f"    FAIL stderr: {r.stderr[-300:]}")
            cache.commit()
    jsonl.close()
    logf.close()
    cache.commit()


@app.function(image=IMAGE, gpu="T4", volumes={"/cache": cache}, timeout=7200)
def _bench_exec(run_id: str, actions: list, grids: dict, n_md: int, iters: int,
                arch: str = "native", jobs: int = 8, meta: dict | None = None):
    """Per-GPU throughput sweep: build profile_cuda_hmc for this GPU's arch, then
    climb each action's size grid. Each (action,L) runs an hmc pass (traj/s) and,
    on success, a --force-only pass (eff_GBps); both JSON lines append to
    throughput.jsonl. A cudaMalloc OOM exits the child nonzero → record a marker
    and stop that action's climb."""
    import json
    import subprocess

    out = _container_setup(run_id, arch, jobs, meta)
    logf = open(f"{out}/console.log", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    # Reconfigure the shared tree for this GPU's arch (sequential runs → no
    # contention), then build the one target.
    arch_flag = f"-DCMAKE_CUDA_ARCHITECTURES={arch}" if arch and arch != "native" else ""
    subprocess.run(f"cmake --preset linux-nvcc {arch_flag}", shell=True,
                   cwd="/root/reticolo", check=True)
    subprocess.run(f"cmake --build --preset linux-nvcc --target profile_cuda_hmc -j {jobs}",
                   shell=True, cwd="/root/reticolo", check=True)
    binpath = subprocess.run(
        f'find {BUILD} -name profile_cuda_hmc -type f -perm -u+x | head -1',
        shell=True, cwd="/root/reticolo", capture_output=True, text=True).stdout.strip()
    if not binpath:
        raise RuntimeError("profile_cuda_hmc produced no binary")

    def run_one(extra):
        r = subprocess.run([f"/root/reticolo/{binpath}", *extra],
                           cwd="/root/reticolo", capture_output=True, text=True)
        rows = [ln for ln in r.stdout.splitlines() if ln.strip().startswith("{")]
        return r.returncode, (rows[-1] if rows else ""), r.stderr

    jsonl = open(f"{out}/throughput.jsonl", "w")
    errf = open(f"{out}/sweep.err", "w")
    for action in actions:
        for L in grids[action]:
            base = [f"--action={action}", f"--size={L}", f"--iters={iters}"]
            rc, line, err = run_one(base + [f"--n_md={n_md}"])
            errf.write(err)
            if rc != 0 or not line:
                jsonl.write(json.dumps({"action": action, "L": L,
                                        "status": "oom", "rc": rc}) + "\n")
                jsonl.flush()
                log(f"  {action} L={L}: OOM/fail (rc={rc}) — stopping {action} climb")
                break
            jsonl.write(line + "\n")
            log(f"  {action} L={L}: {line}")
            frc, fline, ferr = run_one(base + ["--force-only"])
            errf.write(ferr)
            if frc == 0 and fline:
                jsonl.write(fline + "\n")
            jsonl.flush()
            cache.commit()                             # persist partial results
    jsonl.close()
    errf.close()
    logf.close()
    cache.commit()


# --- local orchestration (runs under `uv run`, not in the container) -----------

def _build_script():
    return (
        'set -e\n'
        'cmake --preset linux-nvcc ${RETICOLO_CUDA_ARCH:+-DCMAKE_CUDA_ARCHITECTURES=$RETICOLO_CUDA_ARCH}\n'
        'cmake --build --preset linux-nvcc ${JOBS:+-j $JOBS}\n'
        'ctest --preset linux-nvcc --output-on-failure\n'
    )


def _run_script(target, run_args):
    return (
        'set -e\n'
        # Reconfigure for this GPU's arch — the shared Volume build tree may hold a
        # different CMAKE_CUDA_ARCHITECTURES from the last run, and building without
        # reconfiguring would launch a wrong-SM cubin ("named symbol not found").
        'cmake --preset linux-nvcc ${RETICOLO_CUDA_ARCH:+-DCMAKE_CUDA_ARCHITECTURES=$RETICOLO_CUDA_ARCH}\n'
        f'cmake --build --preset linux-nvcc --target {target} ${{JOBS:+-j $JOBS}}\n'
        f'bin="$PWD/$(find "{BUILD}" -name {target} -type f -perm -u+x | head -1)"\n'
        f'[ -x "$bin" ] || {{ echo "target {target} produced no binary"; exit 1; }}\n'
        'work="$(mktemp -d)"; cd "$work"\n'
        f'echo "+ $bin {run_args}"\n'
        f'"$bin" {run_args}\n'
        'mkdir -p "$OUT_DIR/data"; cp -a "$work"/. "$OUT_DIR/data/" 2>/dev/null || true\n'
        'echo "exported $(ls -A "$work" 2>/dev/null | wc -l) item(s) -> $OUT_DIR/data"\n'
    )


def _dispatch(label, script, a):
    slug = re.sub(r"[^A-Za-z0-9]+", "-", (a.name or label)).strip("-").lower()
    now = datetime.now()
    run_id = f"{now:%Y-%m-%d}-{now:%H%M%S}-{slug}"
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    # Resolve a concrete arch for the requested GPU so the build lands in a
    # per-arch tree (build-<arch>) and CMAKE_CUDA_ARCHITECTURES is pinned — no
    # reliance on native detection against a shared, possibly-stale CMake cache.
    arch = a.arch if a.arch != "native" else GPU_ARCH.get(a.gpu, "native")
    meta = {"label": label, "name": a.name, "gpu": a.gpu, "arch": arch, "cpu": a.cpu,
            "git_sha": sha, "started": now.isoformat(), "script": script}
    opts = {"gpu": a.gpu, "cpu": a.cpu, **({"memory": a.mem} if a.mem else {})}

    print(f"run_id = {run_id}  (gpu={a.gpu}, arch={arch}, cpu={a.cpu}, sha={sha})")
    with modal.enable_output(), app.run():
        _exec.with_options(**opts).remote(script, run_id=run_id, arch=arch,
                                          jobs=max(1, int(a.cpu)), meta=meta)
    print(f"artifacts: out/{run_id}\n  fetch: uv run tools/modal/app.py pull {run_id}")


def _bench(a):
    now = datetime.now()
    session = a.name or f"{now:%Y-%m-%d}-{now:%H%M%S}-bench"
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"bench session: {session}  gpus={gpus}  sha={sha}")
    for gpu in gpus:
        gslug = re.sub(r"[^A-Za-z0-9]+", "-", gpu).strip("-").lower()
        run_id = f"{session}/{gslug}"                  # nested → pull --session fetches all
        arch = a.arch if a.arch != "native" else GPU_ARCH.get(gpu, "native")
        meta = {"label": "bench", "session": session, "gpu_request": gpu, "arch": arch,
                "cpu": a.cpu, "git_sha": sha, "actions": BENCH_ACTIONS, "grids": BENCH_GRID,
                "n_md": a.n_md, "iters": a.iters, "started": now.isoformat()}
        opts = {"gpu": gpu, "cpu": a.cpu, "timeout": 7200,
                **({"memory": a.mem} if a.mem else {})}
        print(f"\n=== bench gpu={gpu} arch={arch} run_id={run_id} ===")
        try:
            with modal.enable_output(), app.run():
                _bench_exec.with_options(**opts).remote(
                    run_id=run_id, actions=BENCH_ACTIONS, grids=BENCH_GRID,
                    n_md=a.n_md, iters=a.iters, arch=arch, jobs=max(1, int(a.cpu)), meta=meta)
        except Exception as e:                         # one GPU failing must not abort the rest
            print(f"!! gpu={gpu} failed: {e}\n   continuing with remaining GPUs")
    print(f"\nbench session {session} done")
    print(f"  pull:    uv run tools/modal/app.py pull --session {session}")
    print(f"  compare: python tools/profile/compare.py tools/modal/output/{session}")


def _stress(a):
    now = datetime.now()
    run_id = f"{now:%Y-%m-%d}-{now:%H%M%S}-{re.sub(r'[^A-Za-z0-9]+', '-', (a.name or 'stress')).strip('-').lower()}"
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    arch = a.arch if a.arch != "native" else GPU_ARCH.get(a.gpu, "native")
    meta = {"label": "stress", "name": a.name, "gpu": a.gpu, "arch": arch, "cpu": a.cpu,
            "git_sha": sha, "vols": {str(k): v for k, v in STRESS_VOLS.items()},
            "nreps": STRESS_NREPS, "started": now.isoformat()}
    opts = {"gpu": a.gpu, "cpu": a.cpu, "timeout": 7200, **({"memory": a.mem} if a.mem else {})}
    print(f"stress matrix on {a.gpu} (arch={arch}): "
          f"L={list(STRESS_VOLS)} × N={STRESS_NREPS}  run_id={run_id}")
    with modal.enable_output(), app.run():
        _stress_exec.with_options(**opts).remote(
            run_id=run_id, vols=STRESS_VOLS, nreps=STRESS_NREPS,
            arch=arch, jobs=max(1, int(a.cpu)), meta=meta)
    print(f"artifacts: out/{run_id}\n  fetch: uv run tools/modal/app.py pull {run_id}")


def _modal(*cmd):
    return subprocess.run([shutil.which("modal") or "modal", *cmd], check=False)


def _pull(run_id=None, session=None):
    if session:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        # `modal volume get` places the remote dir (basename) under the local target,
        # so passing OUTPUT lands the tree at OUTPUT/<session>/<gpu>/.
        print(f"fetching session out/{session} -> {OUTPUT / session}")
        _modal("volume", "get", "--force", VOL, f"out/{session}", str(OUTPUT))
        return
    if not run_id:
        print(f"runs on volume {VOL}:")
        _modal("volume", "ls", VOL, "out")
        print("fetch one with: uv run tools/modal/app.py pull <run_id>")
        return
    dst = OUTPUT / run_id
    dst.mkdir(parents=True, exist_ok=True)
    print(f"fetching out/{run_id} -> {dst}")
    _modal("volume", "get", "--force", VOL, f"out/{run_id}", str(dst))


def main():
    p = argparse.ArgumentParser(prog="app.py", description="reticolo CUDA backend on Modal GPUs")
    subs = p.add_subparsers(dest="cmd", required=True)

    def gpu_opts(sp):
        sp.add_argument("--gpu", default="T4", help="Modal GPU (T4|L4|A100|H100|B200|...)")
        sp.add_argument("--cpu", type=float, default=8.0, help="cores requested + build -j")
        sp.add_argument("--arch", default="native", help="override the preset's native CUDA arch")
        sp.add_argument("--name", default="", help="run-id label")
        sp.add_argument("--mem", type=int, default=0, help="memory request (MB)")

    gpu_opts(subs.add_parser("build", help="configure + build + ctest via the linux-nvcc preset"))
    r = subs.add_parser("run", help="build one target, run it, export its output")
    r.add_argument("--app", required=True, help="cuda target to build & run")
    r.add_argument("--args", default="", help="arguments passed to the binary")
    gpu_opts(r)

    b = subs.add_parser("bench", help="multi-GPU HMC throughput sweep (sequential)")
    b.add_argument("--gpus", default=BENCH_FP64_GPUS,
                   help="comma list, run sequentially (default: FP64-capable GPUs)")
    b.add_argument("--cpu", type=float, default=8.0, help="cores requested + build -j")
    b.add_argument("--arch", default="native", help="override the per-GPU arch map")
    b.add_argument("--name", default="", help="session id")
    b.add_argument("--mem", type=int, default=0, help="memory request (MB)")
    b.add_argument("--n_md", type=int, default=10, help="HMC MD steps per trajectory")
    b.add_argument("--iters", type=int, default=30, help="timed reps per config")

    gpu_opts(subs.add_parser("stress", help="LLR (volume × n_rep) scaling matrix on one GPU"))

    pl = subs.add_parser("pull", help="download a run's artifacts (no id → list runs)")
    pl.add_argument("run_id", nargs="?")
    pl.add_argument("--session", help="fetch a whole bench session (out/<session>/*)")

    a = p.parse_args()
    if a.cmd == "pull":
        _pull(a.run_id, a.session)
    elif a.cmd == "build":
        _dispatch("build", _build_script(), a)
    elif a.cmd == "run":
        _dispatch(a.name or f"run-{a.app}", _run_script(a.app, a.args), a)
    elif a.cmd == "bench":
        _bench(a)
    elif a.cmd == "stress":
        _stress(a)


if __name__ == "__main__":
    main()
