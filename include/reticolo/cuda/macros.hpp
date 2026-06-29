#pragma once

// The HD annotation macros now live in core (a shared concern: core action
// headers annotate their per-site formula functions RETICOLO_HD so one formula
// is compiled for both the CPU backend and the device kernels). This header
// remains as the CUDA-side spelling and for back-compat with cuda/ includes.
#include <reticolo/core/hd.hpp>
