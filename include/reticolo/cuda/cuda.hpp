#pragma once

// Public umbrella for the CUDA backend (reticolo::cuda). A GPU app includes the
// core umbrella plus this one:
//
//     #include <reticolo/reticolo.hpp>
//     #include <reticolo/cuda/cuda.hpp>
//
// and gets the whole device stack: DeviceField (the host-side handle), the
// generic DeviceAction<HostAction,Field> + Hmc<DAct,Integ,Field>, on-device
// reductions, and the device-functor adapters for every supported action +
// gauge group. The per-phase validation harnesses under cuda/probes/ are NOT
// part of the public API and are intentionally excluded — only the selftest TU
// includes those.
//
// This header is nvcc-only: it pulls in .cuh kernel headers and <cuda_runtime.h>
// transitively. It must never be reached from a pure-host TU; that is the same
// rule that keeps CUDA headers out of reticolo::core.

// NOLINTBEGIN(misc-include-cleaner): re-exports are the whole point of the umbrella.
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_action.cuh>
#include <reticolo/cuda/device_buffer.hpp>
#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/device_topology.hpp>
#include <reticolo/cuda/graph.hpp>
#include <reticolo/cuda/hmc.cuh>
#include <reticolo/cuda/integ_ops.hpp>
#include <reticolo/cuda/macros.hpp>
#include <reticolo/cuda/pinned.hpp>
#include <reticolo/cuda/reduce.hpp>
#include <reticolo/cuda/stream.hpp>
// Device-functor adapters: one per supported host action.
#include <reticolo/cuda/actions/bose_gas.hpp>
#include <reticolo/cuda/actions/compact_u1.hpp>
#include <reticolo/cuda/actions/phi4.hpp>
#include <reticolo/cuda/actions/phi6.hpp>
#include <reticolo/cuda/actions/sine_gordon.hpp>
#include <reticolo/cuda/actions/wilson.hpp>
#include <reticolo/cuda/actions/xy.hpp>
// Gauge group device traits (re-implement per-link math RETICOLO_HD).
#include <reticolo/cuda/gauge/group_device.hpp>
#include <reticolo/cuda/gauge/su2_device.cuh>
#include <reticolo/cuda/gauge/su3_device.cuh>
// NOLINTEND(misc-include-cleaner)
