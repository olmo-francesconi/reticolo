#include "../test_helpers.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>

#include <catch2/catch_test_macros.hpp>

#ifndef BENCH_VOLUME_SCALING_BINARY
    #error "BENCH_VOLUME_SCALING_BINARY compile definition is required"
#endif

using reticolo::test::run_and_require_exit;

namespace {

std::filesystem::path csv_path() {
    return std::filesystem::temp_directory_path() /
           ("reticolo_bench_volume_scaling_smoke_" + std::to_string(::getpid()) + ".csv");
}

}  // namespace

TEST_CASE("bench_volume_scaling emits a well-formed CSV", "[app][e2e][bench_volume_scaling]") {
    auto const out = csv_path();
    std::error_code ec;
    std::filesystem::remove(out, ec);

    // Tiny budget — enough rows to validate the schema, fast enough not
    // to noticeably slow the suite.
    std::string const cmd = std::string{BENCH_VOLUME_SCALING_BINARY} +
                            " --ndims=2 --sizes=4"
                            " --actions=phi4,wilson_su2"
                            " --budget_dofs=1e6 --budget_seconds=0.05"
                            " --seed=20260521 --out=" +
                            out.string();
    run_and_require_exit(cmd);
    REQUIRE(std::filesystem::exists(out));

    std::ifstream in{out};
    REQUIRE(in.good());

    std::string header;
    std::getline(in, header);
    REQUIRE(header == "ndim,L,nsites,dofs,action,kernel,n_calls,wall_s,dof_per_s");

    std::set<std::pair<std::string, std::string>> seen;
    std::string line;
    while (std::getline(in, line)) {
        std::stringstream ss{line};
        std::string ndim;
        std::string L;
        std::string nsites;
        std::string dofs;
        std::string action;
        std::string kernel;
        std::string n_calls;
        std::string wall_s;
        std::string dof_per_s;
        std::getline(ss, ndim, ',');
        std::getline(ss, L, ',');
        std::getline(ss, nsites, ',');
        std::getline(ss, dofs, ',');
        std::getline(ss, action, ',');
        std::getline(ss, kernel, ',');
        std::getline(ss, n_calls, ',');
        std::getline(ss, wall_s, ',');
        std::getline(ss, dof_per_s, ',');
        REQUIRE(std::stod(wall_s) > 0.0);
        REQUIRE(std::stoll(n_calls) > 0);
        seen.emplace(action, kernel);
    }

    REQUIRE(seen.count({"phi4", "s_full"}) == 1);
    REQUIRE(seen.count({"phi4", "compute_force"}) == 1);
    REQUIRE(seen.count({"wilson_su2", "s_full"}) == 1);
    REQUIRE(seen.count({"wilson_su2", "compute_force"}) == 1);

    std::filesystem::remove(out, ec);
}
