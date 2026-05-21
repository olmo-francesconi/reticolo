// bench_volume_scaling — sweep `s_full` and `compute_force` throughput across
// (ndim, L) for a small set of representative actions: phi4 (scalar),
// compact_u1 (hand-tuned U(1) link kernels), wilson_su2, wilson_su3
// (matrix-link gauge). Each kernel runs under a twin budget: stop after
// `budget_dofs` cumulative dof updates or `budget_seconds` wall time,
// whichever hits first. Output is one CSV row per (ndim, L, action, kernel)
// to stdout or `--out`.
//
// CSV columns: ndim, L, nsites, dofs, action, kernel, n_calls, wall_s,
//              dof_per_s.

#include <reticolo/reticolo.hpp>

#include "_bench/hot_init.hpp"
#include "_bench/timing.hpp"

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using reticolo::bench::Budget;
using reticolo::bench::BudgetedResult;
using reticolo::bench::consume;
using reticolo::bench::hot_init;
using reticolo::bench::time_per_call_budgeted;

std::vector<std::string> split_csv(std::string const& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char const c : s) {
        if (c == ',') {
            if (!cur.empty()) {
                out.push_back(cur);
            }
            cur.clear();
        } else if (c != ' ' && c != '\t') {
            cur += c;
        }
    }
    if (!cur.empty()) {
        out.push_back(cur);
    }
    return out;
}

std::vector<int> parse_int_list(std::string const& s, char const* what) {
    std::vector<int> out;
    for (auto const& tok : split_csv(s)) {
        try {
            out.push_back(std::stoi(tok));
        } catch (...) {
            throw std::runtime_error(std::string{"bad --"} + what + " token: " + tok);
        }
    }
    if (out.empty()) {
        throw std::runtime_error(std::string{"--"} + what + " is empty");
    }
    return out;
}

void emit_row(std::ostream& os,
              int ndim,
              int L,
              std::size_t nsites,
              std::size_t dofs,
              char const* action,
              char const* kernel,
              BudgetedResult const& r) {
    double const dof_per_s = static_cast<double>(dofs) / r.wall_s;
    os << ndim << ',' << L << ',' << nsites << ',' << dofs << ',' << action << ',' << kernel << ','
       << r.n_calls << ',' << r.wall_s << ',' << dof_per_s << '\n';
    os.flush();
}

void start_progress(
    int ndim, int L, std::size_t dofs, char const* action, char const* kernel) {
    std::fprintf(stderr,
                 "[bench] ndim=%d L=%-3d dofs=%-7zu  %-12s %-15s ... ",
                 ndim,
                 L,
                 dofs,
                 action,
                 kernel);
    std::fflush(stderr);
}

void end_progress(BudgetedResult const& r, std::size_t dofs) {
    double const mdof_per_s = static_cast<double>(dofs) / r.wall_s / 1e6;
    double const updates    = static_cast<double>(dofs) * static_cast<double>(r.n_calls);
    // U+2713 CHECK MARK — completes the start line left open with "... ".
    std::fprintf(stderr,
                 "\xE2\x9C\x93  calls=%-8lld  updates=%.2e  wall=%.3es  thr=%.2f Mdof/s\n",
                 r.n_calls,
                 updates,
                 r.wall_s,
                 mdof_per_s);
    std::fflush(stderr);
}

// One lattice live at a time — except for `compute_force`, which by
// signature needs both an input field and an output force buffer. The
// force buffer is therefore constructed only inside the compute_force
// scope and destroyed before this function returns, so the next call's
// field can reuse the same physical memory.
template <class Action, class Field>
void bench_kernels(std::ostream& os,
                   int ndim,
                   int L,
                   char const* name,
                   Action const& action,
                   Field& phi,
                   std::size_t dofs,
                   Budget budget) {
    std::size_t const nsites = phi.nsites();
    {
        start_progress(ndim, L, dofs, name, "s_full");
        auto const r =
            time_per_call_budgeted([&] { consume(action.s_full(phi)); }, dofs, budget);
        end_progress(r, dofs);
        emit_row(os, ndim, L, nsites, dofs, name, "s_full", r);
    }
    {
        Field force{phi.indexing()};
        start_progress(ndim, L, dofs, name, "compute_force");
        auto const r = time_per_call_budgeted(
            [&] { action.compute_force(phi, force); }, dofs, budget);
        end_progress(r, dofs);
        emit_row(os, ndim, L, nsites, dofs, name, "compute_force", r);
    }
}

int main_impl(int argc, char** argv) {
    using namespace reticolo;

    cli::Parser p{"bench_volume_scaling",
                  "Sweep s_full / compute_force throughput across (ndim, L, action) and emit CSV."};
    auto const& ndims_s = p.opt<std::string>(
        "ndims", std::string{"2,3,4"}, "comma-separated spatial dimensions");
    auto const& sizes_s = p.opt<std::string>(
        "sizes", std::string{"4,6,8,12,16,24,32"}, "comma-separated linear lattice extents");
    auto const& actions_s =
        p.opt<std::string>("actions",
                           std::string{"phi4,compact_u1,wilson_su2,wilson_su3"},
                           "comma-separated subset of {phi4,compact_u1,wilson_su2,wilson_su3}");
    auto const& budget_dofs =
        p.opt<double>("budget_dofs", 2e9, "stop a kernel after this many dof updates");
    auto const& budget_seconds =
        p.opt<double>("budget_seconds", 2.0, "stop a kernel after this many wall seconds");
    auto const& seed    = p.opt<unsigned long long>("seed", 42ULL, "RNG seed");
    auto const& outpath = p.opt<std::string>(
        "out", std::string{""}, "CSV output path (default stdout)");
    if (!p.parse(argc, argv)) {
        return 0;
    }

    log::off();

    auto const ndims        = parse_int_list(ndims_s, "ndims");
    auto const sizes        = parse_int_list(sizes_s, "sizes");
    auto const actions_list = split_csv(actions_s);
    std::set<std::string> const actions(actions_list.begin(), actions_list.end());

    static std::set<std::string> const known = {
        "phi4", "compact_u1", "wilson_su2", "wilson_su3"};
    for (auto const& a : actions) {
        if (known.find(a) == known.end()) {
            throw std::runtime_error("unknown --actions token: " + a);
        }
    }

    std::ofstream fout;
    std::ostream* osp = &std::cout;
    if (!outpath.empty()) {
        fout.open(outpath);
        if (!fout) {
            throw std::runtime_error("cannot open --out " + outpath);
        }
        osp = &fout;
    }
    std::ostream& os = *osp;
    os << "ndim,L,nsites,dofs,action,kernel,n_calls,wall_s,dof_per_s\n";

    Budget const budget{.max_dofs = budget_dofs, .max_seconds = budget_seconds};

    std::fprintf(stderr,
                 "[bench] grid: ndims={%s}  sizes={%s}  actions={%s}\n"
                 "[bench] budget: %.3g dof updates OR %.3g s per kernel\n",
                 ndims_s.c_str(),
                 sizes_s.c_str(),
                 actions_s.c_str(),
                 budget_dofs,
                 budget_seconds);
    std::fflush(stderr);

    auto const t_start = std::chrono::steady_clock::now();

    for (int const ndim : ndims) {
        auto const nd = static_cast<std::size_t>(ndim);
        for (int const L : sizes) {
            auto const Lz = static_cast<std::size_t>(L);
            FastRng rng{seed + (static_cast<unsigned long long>(ndim) * 1000U) +
                        static_cast<unsigned long long>(L)};

            // `dofs` here means real numbers stored in the field: 1 per site
            // for phi4, 1 per link for compact_u1 (an angle), 8 per SU(2)
            // link, 18 per SU(3) link. That puts wildly different field
            // types onto the same throughput axis (Mreal/s).
            if (actions.contains("phi4")) {
                Lattice<double>::SizeVec const shape(nd, Lz);
                Lattice<double> phi{shape};
                hot_init(phi, rng);
                act::Phi4<double> const action{.kappa = 0.18, .lambda = 1.0};
                bench_kernels(os, ndim, L, "phi4", action, phi, phi.nsites(), budget);
            }
            if (actions.contains("compact_u1")) {
                LinkLattice<double>::SizeVec const shape(nd, Lz);
                LinkLattice<double> theta{shape, 0.0};
                hot_init(theta, rng);
                action::CompactU1<double> const action{.beta = 1.0};
                bench_kernels(
                    os, ndim, L, "compact_u1", action, theta, theta.nlinks(), budget);
            }
            if (actions.contains("wilson_su2")) {
                using F = MatrixLinkLattice<gauge_group::SU2, double>;
                F::SizeVec const shape(nd, Lz);
                F theta{shape};
                hot_init(theta, rng);
                action::Wilson<gauge_group::SU2, double> const action{.beta = 2.4};
                bench_kernels(os,
                              ndim,
                              L,
                              "wilson_su2",
                              action,
                              theta,
                              gauge_group::SU2::n_real_components * theta.ndims() *
                                  theta.nsites(),
                              budget);
            }
            if (actions.contains("wilson_su3")) {
                using F = MatrixLinkLattice<gauge_group::SU3, double>;
                F::SizeVec const shape(nd, Lz);
                F theta{shape};
                hot_init(theta, rng);
                action::Wilson<gauge_group::SU3, double> const action{.beta = 6.0};
                bench_kernels(os,
                              ndim,
                              L,
                              "wilson_su3",
                              action,
                              theta,
                              gauge_group::SU3::n_real_components * theta.ndims() *
                                  theta.nsites(),
                              budget);
            }
        }
    }

    auto const elapsed_s = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - t_start)
                               .count();
    auto const n_cells =
        static_cast<long long>(ndims.size() * sizes.size() * actions.size());
    std::fprintf(stderr,
                 "[bench] \xE2\x9C\x93 done — %lld cells × 2 kernels in %.2f s\n",
                 n_cells,
                 elapsed_s);
    if (!outpath.empty()) {
        std::fprintf(stderr, "[bench] wrote %s\n", outpath.c_str());
    }
    std::fflush(stderr);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return main_impl(argc, argv);
    } catch (std::exception const& e) {
        std::fprintf(stderr, "bench_volume_scaling: %s\n", e.what());
        return 1;
    }
}
