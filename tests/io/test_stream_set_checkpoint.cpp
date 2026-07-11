// HDF5 round-trip for the multi-stream checkpoint layout
// (io::Writer::rng_streams / io::Reader::rng_streams / io::checkpoint's
// StreamSet overloads): a restored set must continue bit-exact, and every
// mismatched resume (stream count, family kind, old single-rng layout) must
// throw instead of silently forking the chain.

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/rng/philox_rng.hpp>
#include <reticolo/core/rng/stream_set.hpp>
#include <reticolo/io/checkpoint.hpp>

#include "../test_helpers.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using namespace reticolo;

namespace {
// StreamSet no longer splits a fill across its site streams itself
// (normal_fill_sites / block() / visit_blocks() were removed — that split is
// now the owner's job, see updater::Hmc::sample_momenta_). Draw the same total
// count across every site stream plus one driver draw, in a fixed order, so
// the round-trip below still exercises the full stream set.
template <class R>
void fill_all_streams(StreamSet<R>& s, double* out, std::size_t n) {
    std::size_t const ns       = s.n_streams();
    std::size_t const base_cnt = n / ns;
    std::size_t const rem      = n % ns;
    std::size_t off            = 0;
    for (std::size_t k = 0; k < ns; ++k) {
        std::size_t const cnt = base_cnt + (k < rem ? 1 : 0);
        s.site_stream(k).normal_fill(out + off, cnt);
        off += cnt;
    }
}
}  // namespace

TEST_CASE("StreamSet checkpoint resumes bit-exact through save_config/load_config",
          "[io][checkpoint][stream]") {
    test::ScratchH5 const scratch{"ckpt_stream_set"};

    Lattice<double>::SizeVec const shape{4, 4, 4};
    Lattice<double> phi{shape};
    StreamSet<FastRng> rng{42, 4};

    std::size_t const n = phi.nsites();
    fill_all_streams(rng, phi.data(), n);  // odd-block spares live at save time
    io::save_config(scratch.path(), phi, rng, 5);

    std::vector<double> cont(n);
    fill_all_streams(rng, cont.data(), n);
    double const serial_cont = rng.uniform();  // driver draw, continues after the checkpoint

    Lattice<double> phi2{shape};
    StreamSet<FastRng> rng2{999, 4};
    REQUIRE(io::load_config(scratch.path(), phi2, rng2) == 5);
    for (std::size_t i = 0; i < n; ++i) {
        REQUIRE(phi2.data()[i] == phi.data()[i]);
    }
    std::vector<double> resumed(n);
    fill_all_streams(rng2, resumed.data(), n);
    REQUIRE(resumed == cont);
    REQUIRE(rng2.uniform() == serial_cont);
}

TEST_CASE("StreamSet checkpoint rejects a mismatched resume", "[io][checkpoint][stream]") {
    test::ScratchH5 const scratch{"ckpt_stream_set_reject"};

    Lattice<double>::SizeVec const shape{4, 4};
    Lattice<double> phi{shape};
    StreamSet<FastRng> rng{42, 4};
    io::save_config(scratch.path(), phi, rng, 0);

    SECTION("wrong n_streams") {
        StreamSet<FastRng> bad{42, 8};
        REQUIRE_THROWS(io::load_config(scratch.path(), phi, bad));
    }
    SECTION("wrong family kind") {
        StreamSet<PhiloxRng> bad{42, 4};
        REQUIRE_THROWS(io::load_config(scratch.path(), phi, bad));
    }
}

TEST_CASE("stream loader rejects an old single-FastRng layout", "[io][checkpoint][stream]") {
    test::ScratchH5 const scratch{"ckpt_single_rng_layout"};

    Lattice<double>::SizeVec const shape{4, 4};
    Lattice<double> phi{shape};
    FastRng single{9};
    io::save_config(scratch.path(), phi, single, 0);

    StreamSet<FastRng> bad{9, 4};
    REQUIRE_THROWS(io::load_config(scratch.path(), phi, bad));
}
