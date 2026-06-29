#pragma once

#include <reticolo/cuda/macros.hpp>

// Phase-1 dummy device action functors — the Phi4-shaped pair that exercises
// the scalar device protocol end-to-end (force-vs-FD gate, src/cuda/stencil_probe.cu).
//
// They ARE the protocol a real device action follows: a POD holding only
// couplings, RETICOLO_HD throughout (so the host compiler digests them when an
// app instantiates cuda::Hmc), streaming neighbours one at a time into a
// private accumulator — never receiving a materialized neighbour array (that
// would spill registers on the heavy gauge actions). `accumulate(int mu, T nbr)`
// is the single-neighbour arity; the *skeleton's access policy*, not the
// functor, decides how many times it is called: reduce_fwd streams the d
// forward neighbours (each bond once), stencil streams all 2d.
//
// The two functors share the (kappa, lambda) couplings: Phi4Energy yields the
// per-site action contribution, Phi4Force the MD force F = -dS/dphi. The real
// action::Phi4 formula (phi4.hpp) is wired to share this in Phase 2; here it is
// duplicated deliberately so Phase 1 never touches the verified CPU hot path.

namespace reticolo::cuda::test {

struct Phi4Energy {
    double kappa = 0.0;
    double lambda = 0.0;
    using element = double;

    RETICOLO_HD void init(double self) {
        phi_ = self;
        fwd_ = 0.0;
    }
    RETICOLO_HD void accumulate(int /*mu*/, double nbr) { fwd_ += nbr; }
    [[nodiscard]] RETICOLO_HD double finalize() const {
        double const phi2 = phi_ * phi_;
        double const dev = phi2 - 1.0;
        return (-2.0 * kappa * phi_ * fwd_) + phi2 + (lambda * dev * dev);
    }

private:
    double phi_ = 0.0;
    double fwd_ = 0.0;
};

struct Phi4Force {
    double kappa = 0.0;
    double lambda = 0.0;
    using element = double;

    RETICOLO_HD void init(double self) {
        phi_ = self;
        nbrs_ = 0.0;
    }
    RETICOLO_HD void accumulate(int /*mu*/, double nbr) { nbrs_ += nbr; }
    [[nodiscard]] RETICOLO_HD double finalize() const {
        return (2.0 * kappa * nbrs_) - (2.0 * phi_) -
               (4.0 * lambda * phi_ * (phi_ * phi_ - 1.0));
    }

private:
    double phi_ = 0.0;
    double nbrs_ = 0.0;
};

}  // namespace reticolo::cuda::test
