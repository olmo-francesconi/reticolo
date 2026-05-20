#include <reticolo/action/detail/gauge_group/su2.hpp>
#include <reticolo/action/phi4.hpp>
#include <reticolo/action/wilson.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/core/log_helpers.hpp>

#include <filesystem>
#include <iostream>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace log = reticolo::log;

namespace {

// RAII redirector for std::cout / std::cerr — the new logger writes
// directly to those (no set_stream hook), so we swap rdbufs for the
// duration of the test and restore them in the destructor.
//
// Also re-enables the logger: the shared test main (`tests/test_main.cpp`)
// calls `log::off()` so unrelated tests stay silent; the log-specific tests
// need it back on. Restored to whatever it was on destruction.
struct StreamCapture {
    std::stringstream cout_buf;
    std::stringstream cerr_buf;
    std::streambuf* old_cout;
    std::streambuf* old_cerr;
    bool was_enabled;

    StreamCapture()
        : old_cout{std::cout.rdbuf(cout_buf.rdbuf())}, old_cerr{std::cerr.rdbuf(cerr_buf.rdbuf())},
          was_enabled{log::enabled()} {
        log::on();
    }

    ~StreamCapture() {
        std::cout.rdbuf(old_cout);
        std::cerr.rdbuf(old_cerr);
        if (!was_enabled) {
            log::off();
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

}  // namespace

TEST_CASE("info goes to stdout; warn and error go to stderr", "[log]") {
    log::init_serial();
    log::set_color(false);
    log::set_min_level(log::Level::debug);
    StreamCapture cap;

    log::info("init", "info-line");
    log::warn("hmc", "warn-line");
    log::error("io", "error-line");

    REQUIRE(cap.cout_buf.str().find("info-line") != std::string::npos);
    REQUIRE(cap.cout_buf.str().find("warn-line") == std::string::npos);
    REQUIRE(cap.cerr_buf.str().find("warn-line") != std::string::npos);
    REQUIRE(cap.cerr_buf.str().find("error-line") != std::string::npos);
}

TEST_CASE("severity sigil distinguishes levels", "[log]") {
    log::init_serial();
    log::set_color(false);
    log::set_min_level(log::Level::debug);
    StreamCapture cap;

    log::debug("init", "d");
    log::info("init", "i");
    log::warn("init", "w");
    log::error("init", "e");

    auto const so = cap.cout_buf.str();
    auto const se = cap.cerr_buf.str();
    REQUIRE(so.find("·") != std::string::npos);  // debug
    REQUIRE(so.find("┃") != std::string::npos);  // info
    REQUIRE(se.find("⚠") != std::string::npos);  // warn
    REQUIRE(se.find("✖") != std::string::npos);  // error
}

TEST_CASE("tag is truncated and padded to 4 cells", "[log]") {
    log::init_serial();
    log::set_color(false);
    StreamCapture cap;

    log::info("longtag", "msg-trunc");
    log::info("hi", "msg-pad");

    auto const s = cap.cout_buf.str();
    // 4-char truncated tag + 2-space gutter:
    REQUIRE(s.find("long  msg-trunc") != std::string::npos);
    // 2-char tag padded with 2 spaces + 2-space gutter:
    REQUIRE(s.find("hi    msg-pad") != std::string::npos);
}

TEST_CASE("multi-line entry keeps sigil, blanks metadata on continuations", "[log]") {
    log::init_serial();
    log::set_color(false);
    StreamCapture cap;

    log::info("init").line("first").line("second").line("third");

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
    log::init_serial();
    log::set_color(false);
    StreamCapture cap;

    log::info("init", "tick");

    auto const s = cap.cout_buf.str();
    // The first ':' must be at position 3 inside the timestamp; check the
    // literal shape "HHH:MM:SS.mmm" (digits + the right separators) is present.
    bool found_shape = false;
    for (std::size_t i = 0; i + 13 <= s.size(); ++i) {
        auto sub = s.substr(i, 13);
        if (std::isdigit(static_cast<unsigned char>(sub[0])) &&
            std::isdigit(static_cast<unsigned char>(sub[1])) &&
            std::isdigit(static_cast<unsigned char>(sub[2])) && sub[3] == ':' &&
            std::isdigit(static_cast<unsigned char>(sub[4])) &&
            std::isdigit(static_cast<unsigned char>(sub[5])) && sub[6] == ':' &&
            std::isdigit(static_cast<unsigned char>(sub[7])) &&
            std::isdigit(static_cast<unsigned char>(sub[8])) && sub[9] == '.' &&
            std::isdigit(static_cast<unsigned char>(sub[10])) &&
            std::isdigit(static_cast<unsigned char>(sub[11])) &&
            std::isdigit(static_cast<unsigned char>(sub[12]))) {
            found_shape = true;
            break;
        }
    }
    REQUIRE(found_shape);
}

TEST_CASE("set_min_level suppresses lower-severity entries", "[log]") {
    log::init_serial();
    log::set_color(false);
    log::set_min_level(log::Level::warn);
    StreamCapture cap;

    log::info("init", "should-be-dropped");
    log::warn("init", "should-pass");

    REQUIRE(cap.cout_buf.str().find("should-be-dropped") == std::string::npos);
    REQUIRE(cap.cerr_buf.str().find("should-pass") != std::string::npos);

    log::set_min_level(log::Level::debug);  // restore for any later tests
}

TEST_CASE("log::act renders any action's describe() with the act tag", "[log]") {
    log::init_serial();
    log::set_color(false);
    StreamCapture cap;

    reticolo::action::Phi4<double> phi4{.kappa = 0.137, .lambda = 1.000};
    reticolo::action::Wilson<reticolo::gauge_group::SU2, double> w{.beta = 2.500};

    log::act(phi4);
    log::act(w);

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
    std::filesystem::create_directories(tmp);
    log::init_parallel(tmp, /*run_tag_width=*/4);
    log::set_color(false);
    StreamCapture cap;

    {
        auto _ = log::scope("r042");
        log::info("hmc", "inside");
    }
    log::info("post", "outside");

    auto const s           = cap.cout_buf.str();
    auto const inside_pos  = s.find("inside");
    auto const outside_pos = s.find("outside");
    REQUIRE(inside_pos != std::string::npos);
    REQUIRE(outside_pos != std::string::npos);

    // Scoped line must carry the run id; unscoped line (called outside
    // any OpenMP parallel region) renders the `main` placeholder.
    REQUIRE(s.substr(0, inside_pos).find("r042") != std::string::npos);
    REQUIRE(s.substr(inside_pos).find("main") != std::string::npos);

    // Per-run file was created.
    REQUIRE(std::filesystem::exists(tmp / "run.r042.log"));
    std::filesystem::remove_all(tmp);

    log::init_serial();  // restore mode for any later tests
}
