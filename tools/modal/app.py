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
    uv run tools/modal/app.py pull            # list runs on the Volume
    uv run tools/modal/app.py pull <run_id>   # fetch a run's artifacts

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


@app.function(image=IMAGE, gpu="T4", volumes={"/cache": cache}, timeout=3600)
def _exec(script: str, run_id: str, arch: str = "native",
          jobs: int = 8, meta: dict | None = None):
    import json
    import subprocess

    out = f"/cache/out/{run_id}"
    os.makedirs(out, exist_ok=True)
    os.makedirs("/cache/build", exist_ok=True)
    # Point the preset's binaryDir (build/) at the Volume, so the CMake tree +
    # ccache persist across runs and the gate/run commands share one warm build.
    # add_local_dir ignores **/build, so /root/reticolo/build is free.
    subprocess.run("ln -sfn /cache/build /root/reticolo/build", shell=True, check=True)
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
    cache.commit()                                     # land meta before the (long) build

    # stdbuf -oL sets LD_PRELOAD=libstdbuf, inherited by children, so long binaries
    # stream live instead of buffering until exit. Tee into the run dir so a failed
    # run still leaves a log.
    subprocess.run(["stdbuf", "-oL", "-eL", "bash", "-c",
                    f'set -o pipefail; ({script}) 2>&1 | tee "{out}/console.log"'],
                   cwd="/root/reticolo", check=True)
    cache.commit()                                     # persist build tree + ccache + artifacts


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
        f'[ -f "{BUILD}/CMakeCache.txt" ] || {{ echo "no configured build — run \'build\' first"; exit 1; }}\n'
        f'cmake --build --preset linux-nvcc --target {target} ${{JOBS:+-j $JOBS}}\n'
        f'bin="$(find "{BUILD}" -name {target} -type f -perm -u+x | head -1)"\n'
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
    meta = {"label": label, "name": a.name, "gpu": a.gpu, "arch": a.arch, "cpu": a.cpu,
            "git_sha": sha, "started": now.isoformat(), "script": script}
    opts = {"gpu": a.gpu, "cpu": a.cpu, **({"memory": a.mem} if a.mem else {})}

    print(f"run_id = {run_id}  (gpu={a.gpu}, cpu={a.cpu}, sha={sha})")
    with modal.enable_output(), app.run():
        _exec.with_options(**opts).remote(script, run_id=run_id, arch=a.arch,
                                          jobs=max(1, int(a.cpu)), meta=meta)
    print(f"artifacts: out/{run_id}\n  fetch: uv run tools/modal/app.py pull {run_id}")


def _modal(*cmd):
    return subprocess.run([shutil.which("modal") or "modal", *cmd], check=False)


def _pull(run_id):
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
    pl = subs.add_parser("pull", help="download a run's artifacts (no id → list runs)")
    pl.add_argument("run_id", nargs="?")

    a = p.parse_args()
    if a.cmd == "pull":
        _pull(a.run_id)
    elif a.cmd == "build":
        _dispatch("build", _build_script(), a)
    elif a.cmd == "run":
        _dispatch(a.name or f"run-{a.app}", _run_script(a.app, a.args), a)


if __name__ == "__main__":
    main()
