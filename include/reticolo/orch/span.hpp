#pragma once

// Parameter-span orchestrator: run many independent HMC chains concurrently,
// one per point of a parameter grid. Aggregates the worker + driver.

#include <reticolo/orch/span/driver.hpp>
#include <reticolo/orch/span/worker.hpp>
