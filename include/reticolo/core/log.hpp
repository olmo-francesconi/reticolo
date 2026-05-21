#pragma once

// reticolo logger.
//
//  * Threadsafe via std::osyncstream (line-atomic) + a mutex around the
//    per-run file map.
//  * Severity = sigil:   ·  debug   ┃  info   ⚠  warn   ✖  error
//  * Single line format (parallel mode, run_tag_width = 4):
//
//      ┃ r000 HHH:MM:SS.mmm  init  lattice 32^4, β=6.0
//
//  * Multi-line entries preserve the sigil on continuation lines and
//    blank the metadata columns so the message column aligns.
//  * Run-id is bound per OpenMP iteration via RAII Scope (works with
//    schedule(dynamic) and N_sims > N_threads — same thread rebinds
//    each iteration).
//  * Per-run files in parallel mode: <outdir>/run.<runid>.log, lazy-open,
//    per-line flush.

#include <reticolo/core/build_info.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <mutex>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
    #include <omp.h>
#endif

#ifndef _WIN32
    #include <unistd.h>
#endif

namespace reticolo::log {

enum class Level : std::uint8_t { debug, info, warn, error };

// Verbosity mode for the updater `step()` methods (Hmc / Metropolis / Wolff).
// The counter always advances; only the line emission is gated. Extend with
// new modes as needed.
enum class Mode : std::uint8_t { normal, silent };

namespace detail {

struct Config {
    bool parallel_mode = false;
    std::filesystem::path outdir{"."};
    std::size_t run_tag_width = 4;
    Level min_level           = Level::info;
    bool color                = false;
    bool enabled              = true;  // global on/off — flipped via log::off()
};

inline Config& cfg() {
    static Config c;
    return c;
}

inline std::chrono::steady_clock::time_point& mono_start() {
    static auto t = std::chrono::steady_clock::now();
    return t;
}

inline std::chrono::system_clock::time_point& wall_start() {
    static auto t = std::chrono::system_clock::now();
    return t;
}

inline std::mutex& sink_mutex() {
    static std::mutex m;
    return m;
}

inline std::unordered_map<std::string, std::ofstream>& run_files() {
    static std::unordered_map<std::string, std::ofstream> m;
    return m;
}

inline std::vector<std::string>& scope_stack() {
    thread_local std::vector<std::string> s;
    return s;
}

inline std::string format_elapsed() {
    using namespace std::chrono;
    auto const ms_total = duration_cast<milliseconds>(steady_clock::now() - mono_start()).count();
    auto const hh       = ms_total / (3600LL * 1000);
    auto const mm       = (ms_total / (60LL * 1000)) % 60;
    auto const ss       = (ms_total / 1000) % 60;
    auto const mmm      = ms_total % 1000;
    return std::format("{:03}:{:02}:{:02}.{:03}", hh, mm, ss, mmm);
}

inline std::string format_wall(std::chrono::system_clock::time_point tp) {
    auto const tt = std::chrono::system_clock::to_time_t(tp);
    auto const ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()).count() % 1000;
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    constexpr int tm_year_epoch = 1900;
    return std::format("{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:03}",
                       tm.tm_year + tm_year_epoch,
                       tm.tm_mon + 1,
                       tm.tm_mday,
                       tm.tm_hour,
                       tm.tm_min,
                       tm.tm_sec,
                       static_cast<int>(ms));
}

inline std::string_view sigil(Level lv) noexcept {
    switch (lv) {
        case Level::debug:
            return "·";
        case Level::info:
            return "┃";
        case Level::warn:
            return "⚠";
        case Level::error:
            return "✖";
    }
    return "┃";
}

inline std::string_view color_seq(Level lv) noexcept {
    if (!cfg().color) {
        return {};
    }
    switch (lv) {
        case Level::debug:
            return "\033[2m";
        case Level::info:
            return {};
        case Level::warn:
            return "\033[33m";
        case Level::error:
            return "\033[31m";
    }
    return {};
}

inline std::string_view color_reset() noexcept {
    return cfg().color ? "\033[0m" : "";
}

inline std::string fit(std::string_view s, std::size_t w) {
    if (s.size() >= w) {
        return std::string(s.substr(0, w));
    }
    std::string r{s};
    r.append(w - s.size(), ' ');
    return r;
}

inline std::string current_run() {
    auto const& s = scope_stack();
    return s.empty() ? std::string{} : s.back();
}

inline std::ostream& sink_for(Level lv) {
    return (lv == Level::warn || lv == Level::error) ? std::cerr : std::cout;
}

// Must be called with sink_mutex() held by the caller.
inline std::ofstream* run_file_for_locked(std::string const& run_id) {
    if (!cfg().parallel_mode || run_id.empty()) {
        return nullptr;
    }
    auto& files = run_files();
    auto it     = files.find(run_id);
    if (it != files.end()) {
        return &it->second;
    }
    auto path     = cfg().outdir / std::format("run.{}.log", run_id);
    auto [ins, _] = files.emplace(run_id, std::ofstream(path, std::ios::out | std::ios::app));
    return &ins->second;
}

inline bool detect_color(int fd) noexcept {
    if (std::getenv("NO_COLOR") != nullptr) {
        return false;
    }
#ifdef _WIN32
    (void)fd;
    return false;
#else
    return isatty(fd) != 0;
#endif
}

}  // namespace detail

// Fluent multi-line builder. Move-only. Destructor emits the whole entry
// atomically — no interleaving across threads.
class Entry {
public:
    Entry(Level lv, std::string_view tag) : lv_{lv}, tag_{tag} {}
    Entry(Entry const&)            = delete;
    Entry& operator=(Entry const&) = delete;
    Entry(Entry&& other) noexcept
        : lv_{other.lv_}, tag_{std::move(other.tag_)}, lines_{std::move(other.lines_)},
          emitted_{other.emitted_} {
        other.emitted_ = true;
    }
    Entry& operator=(Entry&&) = delete;

    // No ref-qualifier on .line()/.param(): the fluent chain
    // `log::info("tag").line(...).param(...)` works on the materialised
    // temporary because non-static methods may be called on a prvalue.
    // Holding an lvalue Entry (e.g. inside a class's `describe()` method)
    // and calling `.line()`/`.param()` incrementally also works.
    Entry& line(std::string_view s) {
        lines_.emplace_back(s);
        return *this;
    }
    template <class... Args>
    Entry& line(std::format_string<Args...> fmt, Args&&... a) {
        lines_.emplace_back(std::format(fmt, std::forward<Args>(a)...));
        return *this;
    }

    // Parameter line — same as line() but with a fixed 2-space indent so
    // parameters render visually nested under the concept name above them.
    Entry& param(std::string_view s) {
        std::string buf{"  "};
        buf.append(s);
        lines_.emplace_back(std::move(buf));
        return *this;
    }
    template <class... Args>
    Entry& param(std::format_string<Args...> fmt, Args&&... a) {
        std::string buf{"  "};
        buf.append(std::format(fmt, std::forward<Args>(a)...));
        lines_.emplace_back(std::move(buf));
        return *this;
    }

    Entry& operator<<(std::string_view s) {
        lines_.emplace_back(s);
        return *this;
    }

    ~Entry() {
        if (emitted_ || lines_.empty()) {
            return;
        }
        if (!detail::cfg().enabled) {
            return;
        }
        if (static_cast<int>(lv_) < static_cast<int>(detail::cfg().min_level)) {
            return;
        }
        emit();
    }

private:
    void emit();

    Level lv_;
    std::string tag_;
    std::vector<std::string> lines_;
    bool emitted_ = false;
};

inline void Entry::emit() {
    using namespace detail;
    emitted_ = true;

    auto const sig  = sigil(lv_);
    auto const col  = color_seq(lv_);
    auto const rst  = color_reset();
    auto const ts   = format_elapsed();
    auto const tag4 = fit(tag_, 4);
    auto const run  = current_run();
    bool const par  = cfg().parallel_mode;
    auto const runW = cfg().run_tag_width;

    // Placeholder for unscoped lines in parallel mode:
    //   * inside an OpenMP parallel region → `----` (this is a bug — emit
    //     a stderr warning below);
    //   * outside any parallel region → `main` (legitimate main-thread log).
    auto const unscoped_placeholder = [runW]() -> std::string {
#ifdef _OPENMP
        if (omp_in_parallel()) {
            return std::string(runW, '-');
        }
#endif
        return fit("main", runW);
    };
    std::string const run_col =
        par ? fit(run.empty() ? unscoped_placeholder() : run, runW) + " " : std::string{};
    std::string const blank_run(run_col.size(), ' ');
    static constexpr std::string_view blank_ts  = "             ";  // 13 chars
    static constexpr std::string_view blank_tag = "    ";           // 4 chars

    std::string const first = std::format("{}{} {}{}  {}  ", col, sig, run_col, ts, tag4);
    std::string const cont =
        std::format("{}{} {}{}  {}  ", col, sig, blank_run, blank_ts, blank_tag);

    // One lock covers stdout/stderr emission AND the per-run-file map +
    // append. Line-atomic for the whole multi-line Entry — no other thread
    // can interleave between our lines.
    std::lock_guard lk{sink_mutex()};

#ifdef _OPENMP
    if (omp_in_parallel() && run.empty()) {
        std::cerr << "⚠ logger: log call inside parallel region without scope()\n";
    }
#endif

    auto& out = sink_for(lv_);
    for (std::size_t i = 0; i < lines_.size(); ++i) {
        out << (i == 0 ? first : cont) << lines_[i] << rst << '\n';
    }

    if (auto* rf = run_file_for_locked(run); rf != nullptr) {
        std::string const fp = std::format("{} {} {}  {}  ", sig, run, ts, tag4);
        std::string const fc(fp.size(), ' ');
        for (std::size_t i = 0; i < lines_.size(); ++i) {
            (*rf) << (i == 0 ? fp : fc) << lines_[i] << '\n';
        }
        rf->flush();
    }
}

// Free-function entry points -------------------------------------------------

inline Entry debug(std::string_view tag) {
    return {Level::debug, tag};
}
inline Entry info(std::string_view tag) {
    return {Level::info, tag};
}
inline Entry warn(std::string_view tag) {
    return {Level::warn, tag};
}
inline Entry error(std::string_view tag) {
    return {Level::error, tag};
}

template <class... Args>
inline void debug(std::string_view tag, std::format_string<Args...> fmt, Args&&... a) {
    Entry{Level::debug, tag}.line(fmt, std::forward<Args>(a)...);
}
template <class... Args>
inline void info(std::string_view tag, std::format_string<Args...> fmt, Args&&... a) {
    Entry{Level::info, tag}.line(fmt, std::forward<Args>(a)...);
}
template <class... Args>
inline void warn(std::string_view tag, std::format_string<Args...> fmt, Args&&... a) {
    Entry{Level::warn, tag}.line(fmt, std::forward<Args>(a)...);
}
template <class... Args>
inline void error(std::string_view tag, std::format_string<Args...> fmt, Args&&... a) {
    Entry{Level::error, tag}.line(fmt, std::forward<Args>(a)...);
}

// RAII scope guard. Bind `run_id` to the current thread for the lifetime of
// this object. Push/pop a thread-local stack so nested scopes work and so
// the same thread rebinds correctly across OpenMP loop iterations.
class Scope {
public:
    explicit Scope(std::string run_id) { detail::scope_stack().push_back(std::move(run_id)); }
    ~Scope() {
        auto& s = detail::scope_stack();
        if (!s.empty()) {
            s.pop_back();
        }
    }
    Scope(Scope const&)            = delete;
    Scope& operator=(Scope const&) = delete;
    Scope(Scope&&)                 = delete;
    Scope& operator=(Scope&&)      = delete;
};

[[nodiscard]] inline Scope scope(std::string_view run_id) {
    return Scope{std::string{run_id}};
}

// Init / config -------------------------------------------------------------

inline void init_serial() {
    detail::cfg().parallel_mode = false;
    detail::cfg().color         = detail::detect_color(fileno(stdout));
    detail::mono_start();
    detail::wall_start();
}

inline void init_parallel(std::filesystem::path outdir, std::size_t run_tag_width = 4) {
    detail::cfg().parallel_mode = true;
    detail::cfg().outdir        = std::move(outdir);
    detail::cfg().run_tag_width = run_tag_width;
    detail::cfg().color         = detail::detect_color(fileno(stdout));
    detail::mono_start();
    detail::wall_start();
}

inline void set_min_level(Level lv) {
    detail::cfg().min_level = lv;
}
inline void set_color(bool on) {
    detail::cfg().color = on;
}

// Global on/off — overrides `min_level`. Cheaper than `set_min_level(warn)`
// because it short-circuits before any formatting. `banner()` is also
// suppressed while off.
inline void off() {
    detail::cfg().enabled = false;
}
inline void on() {
    detail::cfg().enabled = true;
}
[[nodiscard]] inline bool enabled() {
    return detail::cfg().enabled;
}

namespace detail {
inline bool& banner_shown() {
    static bool b = false;
    return b;
}
}  // namespace detail

// Hero banner. Idempotent — calling more than once is a no-op so apps and
// `log::start` can both invoke it safely.
//
// Heavy-rule frame (┏ ━ ┓ ┃ ┗ ━ ┛) so the left wall reuses the same ┃ as
// the log sigil — the banner flows visually into the log lines below.
// ANSI Shadow figlet for "reticolo"; version spliced into the bottom rule.
// All metadata is compile-time-baked via <reticolo/core/build_info.hpp>;
// only the live OpenMP thread count is read at runtime.
inline void banner() {
    if (!detail::cfg().enabled || detail::banner_shown()) {
        return;
    }
    detail::banner_shown() = true;
    static constexpr std::array<std::string_view, 6> figlet{{
        "██████╗ ███████╗████████╗██╗ ██████╗ ██████╗ ██╗      ██████╗ ",
        "██╔══██╗██╔════╝╚══██╔══╝██║██╔════╝██╔═══██╗██║     ██╔═══██╗",
        "██████╔╝█████╗     ██║   ██║██║     ██║   ██║██║     ██║   ██║",
        "██╔══██╗██╔══╝     ██║   ██║██║     ██║   ██║██║     ██║   ██║",
        "██║  ██║███████╗   ██║   ██║╚██████╗╚██████╔╝███████╗╚██████╔╝",
        "╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝ ╚═════╝ ",
    }};

    // Figlet rows are 62 display cells wide (fixed ANSI Shadow glyph set).
    constexpr std::size_t figlet_cells = 62;
    constexpr std::size_t left_pad     = 5;
    constexpr std::size_t inner_width  = 74;  // total cells between walls
    constexpr std::size_t right_pad    = inner_width - left_pad - figlet_cells;

    int omp_threads = 1;
#ifdef _OPENMP
    omp_threads = omp_get_max_threads();
#endif

    auto rule = [](std::string_view corner_l, std::string_view corner_r, std::size_t cells) {
        std::string s{corner_l};
        for (std::size_t i = 0; i < cells; ++i) {
            s += "━";
        }
        s += corner_r;
        return s;
    };

    std::string const blank_row = std::string{"┃"} + std::string(inner_width, ' ') + "┃";

    // Bottom rule: ┗━━━ ... ━ v0.3.0 ━━━━━━┛
    std::string const tag       = std::format(" v{} ", build::version);
    std::size_t const tag_cells = tag.size();  // ASCII version — 1 cell/byte
    std::size_t const dashes    = inner_width > (tag_cells + 1) ? inner_width - tag_cells - 1 : 0;
    std::string bottom          = "┗";
    for (std::size_t i = 0; i < dashes; ++i) {
        bottom += "━";
    }
    bottom += tag;
    bottom += "━┛";

    std::lock_guard lk{detail::sink_mutex()};
    auto& out = std::cout;

    out << '\n';
    out << rule("┏", "┓", inner_width) << '\n';
    out << blank_row << '\n';
    for (auto const& row : figlet) {
        out << "┃" << std::string(left_pad, ' ') << row << std::string(right_pad, ' ') << "┃"
            << '\n';
    }
    out << blank_row << '\n';
    out << bottom << '\n';

    // Metadata block — same ┃ sigil as log lines, no frame, so it bridges
    // into the run log naturally.
    out << std::format("┃ branch   : {} @ {}\n", build::git_branch, build::git_commit);
    out << std::format("┃ compiler : {}\n", build::compiler);
    out << std::format("┃ build    : {} · {}\n", build::build_type, build::simd);
    out << std::format("┃ openmp   : {}\n",
                       build::openmp_enabled
                           ? std::format("{} thread{}", omp_threads, omp_threads == 1 ? "" : "s")
                           : std::string{"disabled"});
    out << std::format("┃ started  : {} (local)\n", detail::format_wall(detail::wall_start()));

    // Section break between banner metadata and the live log stream.
    // Heavy T-junction continues the running ┃ column into a horizontal rule.
    std::string sep{"┣"};
    for (std::size_t i = 0; i < inner_width; ++i) {
        sep += "━";
    }
    out << sep << '\n';
}

// One-shot app entry point: init the logger in parallel mode with the
// per-run files landing next to the given HDF5 output, then print the
// banner. Idempotent — `init_parallel` overwrites cfg cheaply, banner
// guards on its static flag.
inline void start(std::filesystem::path const& output_path) {
    auto dir = output_path.parent_path();
    init_parallel(dir.empty() ? std::filesystem::path{"."} : std::move(dir),
                  /*run_tag_width=*/4);
    banner();
}

// Serial-mode shorthand for apps that don't have multiple replicas /
// per-run files. Same idempotency story.
inline void start() {
    init_serial();
    banner();
}

}  // namespace reticolo::log
