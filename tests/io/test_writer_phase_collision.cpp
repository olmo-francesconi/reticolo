#include <reticolo/io/writer.hpp>

#include <filesystem>
#include <stdexcept>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <unistd.h>

namespace {

struct TempH5 {
    std::filesystem::path path;
    explicit TempH5(std::string const& tag) {
        path = std::filesystem::temp_directory_path() /
               ("reticolo_" + tag + "_" + std::to_string(::getpid()) + ".h5");
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    ~TempH5() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    TempH5(TempH5 const&)            = delete;
    TempH5& operator=(TempH5 const&) = delete;
};

}  // namespace

using reticolo::io::Writer;

TEST_CASE("start_phase succeeds the first time and throws on reuse", "[io][phase]") {
    TempH5 f{"phase_collision"};

    Writer w{f.path};
    REQUIRE_NOTHROW(w.start_phase("therm"));
    REQUIRE_NOTHROW(w.start_phase("prod"));
    REQUIRE_THROWS_AS(w.start_phase("therm"), std::runtime_error);
}

TEST_CASE("start_phase rejects names whose root group already exists in the file", "[io][phase]") {
    TempH5 f{"phase_preexisting"};

    Writer w{f.path};
    // Create /prod by writing a series under it first.
    {
        auto s = w.series<double>("/prod/obs/s");
        s.append(1.0);
    }
    REQUIRE_THROWS_AS(w.start_phase("prod"), std::runtime_error);
}
