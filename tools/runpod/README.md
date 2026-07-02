# RunPod profiling workflow

SSH-driven build / test / run / **profile** of the CUDA backend on a rented
RunPod GPU. This exists because **`nsys` GPU-kernel profiling works on RunPod but
not on Modal** — Modal's virtualized GPU can't give nsys the GPU↔CPU clock sync
it needs (`ConvertGpuTicksToSyncNs` fatal error), so no `.nsys-rep` is produced.
On a RunPod box nsys runs cleanly.

You edit locally; these scripts rsync the tree to the pod, build there, and pull
results back. Nothing is committed to run.

## What works vs not (in a pod container)

| | status | notes |
|---|---|---|
| build + ctest | ✅ | full suite, `linux-nvcc` preset |
| run an app | ✅ | output pulled back |
| **`nsys` profile** | ✅ | needs `--cuda-graph-trace=node` (trajectory is a captured graph) |
| `ncu` occupancy/roofline | ❌ | `ERR_NVGPUCTRPERM` — GPU perf counters locked in the container; can't toggle the driver module param from inside a pod. Get occupancy from grid arithmetic instead. |

## Why a V100

Cheapest GPU with **real 1:2 FP64** (the CUDA backend runs in `double` by default;
consumer cards are 1:32 and skew the f64 profile). And its **80 SMs ≈ the A100's
108**, so kernel-occupancy questions ("does this kernel fill the device?")
*transfer* — a GTX-class card (8 SMs) can't tell you that. `sm_70` → `RUNPOD_ARCH=70`.

## Pod setup

1. Launch a pod with:
   - a **CUDA-devel or PyTorch-devel** template — it must have `nvcc` (a
     runtime-only image has the CUDA runtime but no compiler; `setup.sh` will
     error out early if `nvcc` is missing).
   - a **V100** (or A100/H100 → set `RUNPOD_ARCH` accordingly).
   - ~**20 GB** disk (build tree + HDF5 + the CUDA toolkit if the image lacks it).
   - **SSH** enabled (grab the `ssh root@<ip> -p <port> -i <key>` line RunPod gives).
2. Local config — copy the template and fill in the connection:
   ```sh
   cp tools/runpod/env.local.example tools/runpod/env.local
   $EDITOR tools/runpod/env.local     # RUNPOD_HOST / RUNPOD_PORT / RUNPOD_KEY / RUNPOD_ARCH
   ```
   (`env.local` is gitignored.)
3. Bootstrap the pod once:
   ```sh
   ./tools/runpod/setup.sh
   ```

## Usage

```sh
./tools/runpod/build.sh                       # sync + configure + build + ctest
./tools/runpod/build.sh phi4_hmc_cuda         # build one target, skip ctest
./tools/runpod/run.sh   phi4_hmc_cuda --size=8 --n_prod=200 --out=x.h5
./tools/runpod/profile.sh phi4_hmc_cuda --size=16 --n_therm=20 --n_prod=200 --out=p.h5
```

`run` output lands in `tools/runpod/output/<target>/`; `profile` drops
`prof.nsys-rep` (open in the **Nsight Systems GUI**) + kernel/api/memory CSVs in
`tools/runpod/output/prof/`. Keep profile runs SHORT — graph-node tracing grows
the trace fast; a few hundred trajectories is plenty for the kernel breakdown.

## Gotchas baked into the scripts (why they do what they do)

- **`nvcc` off PATH.** Images keep the toolkit at `/usr/local/cuda-X.Y/bin`, and
  the `/usr/local/cuda` symlink can get repointed to a config-only stub by apt.
  The scripts pick the newest `cuda-*/bin` that actually has `nvcc`.
- **GCC 11 lacks `<format>`.** Ubuntu 22.04's default compiler can't build the
  C++20 code. `setup.sh` installs **g++-13** (via the toolchain PPA, added by
  hand because pip's Python shadows the system `apt_pkg` and breaks
  `add-apt-repository`), and the build uses it as both CXX and CUDA host compiler
  with `-allow-unsupported-compiler`.
- **`nsys` package.** Installed as `cuda-nsight-systems-<ver>` matching the
  toolkit; `--cuda-graph-trace=node` is required because `cuda::Hmc` replays each
  trajectory as a captured CUDA graph.
