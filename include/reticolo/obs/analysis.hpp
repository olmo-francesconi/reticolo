#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>

// Ensemble-level reductions over time series of per-config measurements.
// These take std::span<double const> and do not touch the lattice — they
// belong on the analysis side, not in the per-config observer set.

namespace reticolo::obs::analysis {

// Sample mean.
[[nodiscard]] inline double mean(std::span<double const> xs) noexcept {
    if (xs.empty()) {
        return 0.0;
    }
    double s = 0.0;
    for (double const x : xs) {
        s += x;
    }
    return s / static_cast<double>(xs.size());
}

// Magnetic susceptibility:
//  chi = N * (<m^2> - <|m|>^2)
// where the input span holds per-config |m| values and N is the lattice volume.
// Returns 0 for an empty span.
[[nodiscard]] inline double susceptibility(std::span<double const> abs_m, double n_sites) {
    if (n_sites <= 0.0) {
        throw std::invalid_argument{"obs::analysis::susceptibility: n_sites must be positive"};
    }
    if (abs_m.empty()) {
        return 0.0;
    }
    double sum_m  = 0.0;
    double sum_m2 = 0.0;
    for (double const m : abs_m) {
        sum_m += m;
        sum_m2 += m * m;
    }
    auto const inv_n  = 1.0 / static_cast<double>(abs_m.size());
    auto const avg_m  = sum_m * inv_n;
    auto const avg_m2 = sum_m2 * inv_n;
    return n_sites * (avg_m2 - (avg_m * avg_m));
}

// Binder cumulant:
//  U = 1 - <m^4> / (3 <m^2>^2)
// where the two input spans hold per-config m^2 and m^4 values; the spans must
// be the same length (paired measurements). Returns 0 for empty spans.
[[nodiscard]] inline double binder(std::span<double const> m2_series,
                                   std::span<double const> m4_series) {
    if (m2_series.size() != m4_series.size()) {
        throw std::invalid_argument{"obs::analysis::binder: m2 and m4 spans must have equal size"};
    }
    if (m2_series.empty()) {
        return 0.0;
    }
    double sum_m2 = 0.0;
    double sum_m4 = 0.0;
    for (std::size_t i = 0; i < m2_series.size(); ++i) {
        sum_m2 += m2_series[i];
        sum_m4 += m4_series[i];
    }
    auto const inv_n  = 1.0 / static_cast<double>(m2_series.size());
    auto const avg_m2 = sum_m2 * inv_n;
    auto const avg_m4 = sum_m4 * inv_n;
    if (avg_m2 == 0.0) {
        return 0.0;
    }
    return 1.0 - (avg_m4 / (3.0 * avg_m2 * avg_m2));
}

}  // namespace reticolo::obs::analysis
