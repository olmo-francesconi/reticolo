#include <reticolo/core/field/lattice.hpp>
#include <reticolo/core/rng/fast_rng.hpp>
#include <reticolo/io/reader.hpp>
#include <reticolo/io/writer.hpp>

#include "../test_helpers.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <hdf5.h>

// HDF5 object-id leaks in src/io.
//
// `hid_t` is a resource handle, and both IO TUs report errors by THROWING from
// `hid_check` / `herr_check` — ~44 sites — so any id open at that point used to
// be abandoned. One path leaked on SUCCESS too: `H5Aget_space` inside the
// variable-length string read minted an id that was passed straight into
// `H5Dvlen_reclaim` and never closed.
//
// A leaked FILE-BOUND id (dataset, group, datatype, attribute) matters more than
// its bytes: HDF5 keeps the file object alive until every id derived from it is
// released, so the file never really closes. `detail::HidGuard`
// (src/io/hid_guard.hpp) is the fix.
//
// SCOPE, stated honestly. This counts live file-bound ids with
// `H5Fget_obj_count`, and that is all the public API can see:
//   * `H5Fget_obj_count` has no dataspace category — dataspaces are not bound to
//     a file — so the leaked `H5Aget_space` id above is NOT observable here. It
//     is a plain (small, per-call) memory leak, fixed and reviewed but untested.
//   * `H5Inmembers` would count dataspaces, but HDF5 2.x refuses it on
//     library-internal id types ("cannot call public function on library type").
//   * raw `hid_t` values increase monotonically and are never recycled, so they
//     measure allocations, not liveness.
// What IS covered — every id that pins the file, including all the throw paths.

namespace {

// Live file-bound ids across all open files.
std::size_t live_ids() {
    ssize_t const n = H5Fget_obj_count(
        H5F_OBJ_ALL, H5F_OBJ_DATASET | H5F_OBJ_GROUP | H5F_OBJ_DATATYPE | H5F_OBJ_ATTR);
    REQUIRE(n >= 0);
    return static_cast<std::size_t>(n);
}

}  // namespace

TEST_CASE("writing a file leaks no HDF5 object ids", "[io][hid]") {
    using namespace reticolo;
    reticolo::test::ScratchH5 const out{"hid_leak_write"};

    std::size_t const before = live_ids();
    {
        char const* argv[] = {"test_hid_leaks"};
        io::Writer w{out.path().string(), 1, argv};
        w.start_phase("prod");
        w.attr<std::string>("/cfg@note", "a variable-length string attribute");
        w.attr<double>("/cfg@beta", 2.3);

        auto s = w.series<double>("/prod/obs/s");
        for (int i = 0; i < 40; ++i) {
            s.append(static_cast<double>(i));
        }
        s.flush();

        Lattice<double> phi{{4, 4}};
        w.field("/prod/phi", phi);
        w.rng_state("/prod/rng", FastRng{7});
    }
    REQUIRE(live_ids() == before);
}

TEST_CASE("reading a file leaks no HDF5 object ids", "[io][hid]") {
    using namespace reticolo;
    reticolo::test::ScratchH5 const out{"hid_leak_read"};

    {
        char const* argv[] = {"test_hid_leaks"};
        io::Writer w{out.path().string(), 1, argv};
        w.attr<std::string>("/cfg@note", "vlen string");
        w.attr<double>("/cfg@beta", 2.3);
        Lattice<double> phi{{4, 4}};
        w.field("/phi", phi);
        w.rng_state("/rng", FastRng{7});
    }

    std::size_t const before = live_ids();
    {
        io::Reader r{out.path()};
        // Every read path, several times over, so a per-call leak accumulates
        // well past any one-off allocation.
        for (int i = 0; i < 8; ++i) {
            REQUIRE(r.attr<std::string>("/cfg@note") == "vlen string");
            REQUIRE(r.attr<double>("/cfg@beta") == 2.3);
            REQUIRE(r.field_shape("/phi").size() == 2);
            Lattice<double> phi{{4, 4}};
            r.field("/phi", phi);
            (void)r.rng_state("/rng");
            REQUIRE(r.has("/cfg@note"));
            REQUIRE_FALSE(r.has("/cfg@absent"));
        }
    }
    REQUIRE(live_ids() == before);
}

TEST_CASE("a failed read leaks no HDF5 object ids", "[io][hid]") {
    using namespace reticolo;
    reticolo::test::ScratchH5 const out{"hid_leak_throw"};

    {
        char const* argv[] = {"test_hid_leaks"};
        io::Writer w{out.path().string(), 1, argv};
        Lattice<double> phi{{4, 4}};
        w.field("/phi", phi);
    }

    // The throwing paths are the ones that used to abandon an open `dset`: the
    // validation mismatches reached via `fail()`, which unwinds through the guard.
    std::size_t const before = live_ids();
    {
        io::Reader r{out.path()};
        for (int i = 0; i < 8; ++i) {
            Lattice<double> wrong{{8, 8}};  // shape mismatch against the stored 4×4
            REQUIRE_THROWS(r.field("/phi", wrong));
            // NOTE the path: the OBJECT must exist and the ATTRIBUTE must not,
            // so the throw happens with the object id already open. A path whose
            // object is missing too fails at H5Oopen with nothing open, and
            // exercises none of this.
            REQUIRE_THROWS(r.attr<double>("/phi@missing"));
            REQUIRE_THROWS(r.attr<std::string>("/phi@missing_str"));
            REQUIRE_THROWS(r.rng_state("/phi"));  // not an RNG dataset
        }
    }
    REQUIRE(live_ids() == before);
}
