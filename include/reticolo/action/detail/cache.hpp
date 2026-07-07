#pragma once

#include <limits>

// Shared last-value caches for the action family bases. HMC snapshots the cache at
// h0 and rolls it back on a rejected trajectory (see HasSFullCache / HasSImagCache
// in algorithm/hmc.hpp); each base writes the member inside its own s_full / s_imag
// and inherits the accessors. A tiny mixin so the four family bases (Site/Bond/
// Complex/Gauge) don't each hand-roll the identical accessor+member triple.

namespace reticolo::action::detail {

struct SFullCache {
    [[nodiscard]] double last_s_full() const noexcept { return last_s_full_; }
    void restore_last_s_full(double v) const noexcept { last_s_full_ = v; }
    mutable double last_s_full_ = std::numeric_limits<double>::quiet_NaN();
};

// Complex-LLR twin: the imaginary part S_I is the LLR constraint observable.
struct SImagCache {
    [[nodiscard]] double last_s_imag() const noexcept { return last_s_imag_; }
    void restore_last_s_imag(double v) const noexcept { last_s_imag_ = v; }
    mutable double last_s_imag_ = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace reticolo::action::detail
