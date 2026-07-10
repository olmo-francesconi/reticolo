#include <reticolo/action/gauge/wilson.hpp>
#include <reticolo/action/site/phi4.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/core/log_helpers.hpp>
#include <reticolo/math/group/su2.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
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

// Split captured output into individual lines for content-by-line assertions.
std::vector<std::string> lines_of(std::string const& s) {
    std::vector<std::string> out;
    std::istringstream is{s};
    std::string l;
    while (std::getline(is, l)) {
        out.push_back(l);
    }
    return out;
}

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

TEST_CASE("severity sigil distinguishes levels", "[log]") {
    serial_mode();
    rl::set_color(false);
    rl::set_min_level(rl::Level::debug);
    StreamCapture cap;

    rl::debug("init", "d");
    rl::info("init", "i");
    rl::warn("init", "w");
    rl::error("init", "e");

    auto const so = cap.cout_buf.str();
    auto const se = cap.cerr_buf.str();
    REQUIRE(so.find("·") != std::string::npos);  // debug
    REQUIRE(so.find("┃") != std::string::npos);  // info
    REQUIRE(se.find("⚠") != std::string::npos);  // warn
    REQUIRE(se.find("✖") != std::string::npos);  // error
}

TEST_CASE("tag is truncated and padded to 4 cells", "[log]") {
    serial_mode();
    rl::set_color(false);
    StreamCapture cap;

    rl::info("longtag", "msg-trunc");
    rl::info("hi", "msg-pad");

    auto const s = cap.cout_buf.str();
    // 4-char truncated tag + 2-space gutter:
    REQUIRE(s.find("long  msg-trunc") != std::string::npos);
    // 2-char tag padded with 2 spaces + 2-space gutter:
    REQUIRE(s.find("hi    msg-pad") != std::string::npos);
}

TEST_CASE("multi-line entry keeps sigil, blanks metadata on continuations", "[log]") {
    serial_mode();
    rl::set_color(false);
    StreamCapture cap;

    rl::info("init").line("first").line("second").line("third");

    auto const ls = lines_of(cap.cout_buf.str());
    REQUIRE(ls.size() == 3);
    // Every line begins with the info sigil.
    REQUIRE(ls[0].starts_with("┃"));
    REQUIRE(ls[1].starts_with("┃"));
    REQUIRE(ls[2].starts_with("┃"));
    // First line carries the tag; continuations do not.
    REQUIRE(ls[0].find("init") != std::string::npos);
    REQUIRE(ls[1].find("init") == std::string::npos);
    REQUIRE(ls[2].find("init") == std::string::npos);
    REQUIRE(ls[0].find("first") != std::string::npos);
    REQUIRE(ls[1].find("second") != std::string::npos);
    REQUIRE(ls[2].find("third") != std::string::npos);
}

TEST_CASE("elapsed timestamp uses HHH:MM:SS.mmm format", "[log]") {
    serial_mode();
    rl::set_color(false);
    StreamCapture cap;

    rl::info("init", "tick");

    std::regex const ts{R"(\d{3}:\d{2}:\d{2}\.\d{3})"};
    REQUIRE(std::regex_search(cap.cout_buf.str(), ts));
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

TEST_CASE("rl::act renders any action's describe() with the act tag", "[log]") {
    serial_mode();
    rl::set_color(false);
    StreamCapture cap;

    reticolo::action::Phi4<double> phi4{.kappa = 0.137, .lambda = 1.000};
    reticolo::action::Wilson<reticolo::math::group::SU2, double> w{.beta = 2.500};

    rl::act(phi4);
    rl::act(w);

    auto const s = cap.cout_buf.str();
    REQUIRE(s.find("act") != std::string::npos);
    REQUIRE(s.find("Phi4<double>") != std::string::npos);
    REQUIRE(s.find("κ=0.137") != std::string::npos);
    REQUIRE(s.find("λ=1.000") != std::string::npos);
    REQUIRE(s.find("Wilson<SU2>") != std::string::npos);
    REQUIRE(s.find("β=2.500") != std::string::npos);
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
