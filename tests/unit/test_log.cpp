// Logger behaviour that can actually break something: WHERE a line goes
// (stdout vs stderr), that severity filtering suppresses, that a replica scope
// binds per-thread, and that start() creates the workspace + stems the files.
//
// The presentation details (sigil glyphs, 4-cell tag padding, HHH:MM:SS.mmm
// formatting, multi-line continuation blanking) were asserted here too; those
// tests pinned cosmetics, broke on every formatting tweak, and could not
// produce a wrong result. Dropped deliberately — if the format becomes
// load-bearing for a downstream parser, test it there.

#include <reticolo/core/log/log.hpp>
#include <reticolo/core/log/log_helpers.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

// Alias is `rl`, not `log`, because Linux glibc's <math.h> (transitively
// pulled in via the action headers below) declares `::log(double)` at global
// scope. gcc/clang then refuse `namespace log = ...` as a redeclaration of
// a different kind of entity. macOS libc++ doesn't surface this.
namespace rl = reticolo::log;

namespace {

// RAII redirector for std::cout / std::cerr — the new logger writes
// directly to those (no set_stream hook), so we swap rdbufs for the
// duration of the test and restore them in the destructor.
//
// Also re-enables the logger: the shared test main (`tests/test_main.cpp`)
// calls `rl::off()` so unrelated tests stay silent; the log-specific tests
// need it back on. Restored to whatever it was on destruction.
struct StreamCapture {
    std::stringstream cout_buf;
    std::stringstream cerr_buf;
    std::streambuf* old_cout;
    std::streambuf* old_cerr;
    bool was_enabled;

    StreamCapture()
        : cout_buf{}, cerr_buf{}, old_cout{std::cout.rdbuf(cout_buf.rdbuf())},
          old_cerr{std::cerr.rdbuf(cerr_buf.rdbuf())}, was_enabled{rl::enabled()} {
        rl::on();
    }

    ~StreamCapture() {
        std::cout.rdbuf(old_cout);
        std::cerr.rdbuf(old_cerr);
        if (!was_enabled) {
            rl::off();
        }
    }

    StreamCapture(StreamCapture const&)            = delete;
    StreamCapture& operator=(StreamCapture const&) = delete;
    StreamCapture(StreamCapture&&)                 = delete;
    StreamCapture& operator=(StreamCapture&&)      = delete;
};

// Reset to console-only serial rendering without rl::start() — the public
// init opens a main log file, which the format-only tests don't want.
// White-box by design: this is the logger's own test.
void serial_mode() {
    rl::impl::cfg().replicas = false;
}

}  // namespace

TEST_CASE("info goes to stdout; warn and error go to stderr", "[log]") {
    serial_mode();
    rl::set_color(false);
    rl::set_min_level(rl::Level::debug);
    StreamCapture cap;

    rl::info("init", "info-line");
    rl::warn("hmc", "warn-line");
    rl::error("io", "error-line");

    REQUIRE(cap.cout_buf.str().find("info-line") != std::string::npos);
    REQUIRE(cap.cout_buf.str().find("warn-line") == std::string::npos);
    REQUIRE(cap.cerr_buf.str().find("warn-line") != std::string::npos);
    REQUIRE(cap.cerr_buf.str().find("error-line") != std::string::npos);
}

TEST_CASE("set_min_level suppresses lower-severity entries", "[log]") {
    serial_mode();
    rl::set_color(false);
    rl::set_min_level(rl::Level::warn);
    StreamCapture cap;

    rl::info("init", "should-be-dropped");
    rl::warn("init", "should-pass");

    REQUIRE(cap.cout_buf.str().find("should-be-dropped") == std::string::npos);
    REQUIRE(cap.cerr_buf.str().find("should-pass") != std::string::npos);

    rl::set_min_level(rl::Level::debug);  // restore for any later tests
}

TEST_CASE("Scope binds a run-id for the current thread; clears on exit", "[log]") {
    auto const tmp = std::filesystem::temp_directory_path() / "reticolo_log_test";
    std::filesystem::remove_all(tmp);
    rl::start(tmp, "run.h5", /*replicas=*/true);
    rl::set_color(false);
    StreamCapture cap;

    {
        auto _ = rl::scope("r042");
        rl::info("hmc", "inside");
    }
    rl::info("post", "outside");

    auto const s           = cap.cout_buf.str();
    auto const inside_pos  = s.find("inside");
    auto const outside_pos = s.find("outside");
    REQUIRE(inside_pos != std::string::npos);
    REQUIRE(outside_pos != std::string::npos);

    // Scoped line must carry the run id; unscoped line (called outside
    // any OpenMP parallel region) renders the `main` placeholder.
    REQUIRE(s.substr(0, inside_pos).find("r042") != std::string::npos);
    REQUIRE(s.substr(inside_pos).find("main") != std::string::npos);

    // No separate per-replica files: the single main log mirrors BOTH scoped
    // and unscoped lines.
    REQUIRE(!std::filesystem::exists(tmp / "run.r042.log"));
    REQUIRE(std::filesystem::exists(tmp / "run.log"));
    {
        std::ifstream mf{tmp / "run.log"};
        std::string const all{std::istreambuf_iterator<char>{mf}, std::istreambuf_iterator<char>{}};
        REQUIRE(all.find("inside") != std::string::npos);
        REQUIRE(all.find("outside") != std::string::npos);
    }
    std::filesystem::remove_all(tmp);

    serial_mode();  // restore mode for any later tests
}

TEST_CASE("start(ws, out) creates the workspace and stems files by the out name", "[log]") {
    auto const tmp = std::filesystem::temp_directory_path() / "reticolo_log_stem_test";
    std::filesystem::remove_all(tmp);
    rl::start(tmp, "llr_mu0.9_s43.h5", /*replicas=*/true);  // creates tmp itself
    rl::set_color(false);
    StreamCapture cap;

    {
        auto _ = rl::scope("r007");
        rl::info("hmc", "scoped line");
    }

    // Concurrent sims sharing a workspace must not collide on <stem>.log; the
    // main log is stemmed by the out name, and no per-replica files are written.
    REQUIRE(std::filesystem::exists(tmp / "llr_mu0.9_s43.log"));
    REQUIRE(!std::filesystem::exists(tmp / "llr_mu0.9_s43.r007.log"));
    REQUIRE(!std::filesystem::exists(tmp / "run.log"));
    std::filesystem::remove_all(tmp);

    serial_mode();
}
