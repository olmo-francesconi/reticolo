#pragma once

// reticolo logger.
//
//  * Threadsafe: one mutex covers console + file emission, so multi-line
//    entries are atomic across threads.
//  * Severity = sigil:   ·  debug   ┃  info   ⚠  warn   ✖  error
//  * Single line format (replica mode, run_tag_width = 4):
//
//      ┃ r000 HHH:MM:SS.mmm  init  lattice 32^4, β=6.0
//
//  * Multi-line entries preserve the sigil on continuation lines and
//    blank the metadata columns so the message column aligns.
//  * `log::start(workspace, out_name[, replicas])` is the single init:
//    creates the workspace folder, opens <ws>/<stem>.log (stem = out_name
//    minus extension — sweep-safe when sims share a workspace) and mirrors
//    every entry into it, then prints the banner.
//  * With replicas=true the run-id column is rendered so scoped lines carry
//    their run id; everything still lands in the single main log (there are no
//    separate per-replica files).
//  * Run-id is bound per thread via RAII `log::scope(id)` (works with
//    schedule(dynamic) and N_sims > N_threads — same thread rebinds each
//    iteration). Library code that owns a run id (llr::Replica) binds it
//    internally; an app binds a scope only when it runs its own logging
//    code inside a parallel region.

#include <reticolo/core/build_info.hpp>
#include <reticolo/core/host_info.hpp>

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
#include <system_error>
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

namespace impl {

struct Config {
    bool replicas = false;
    std::filesystem::path workspace{"."};
    std::string stem{"run"};
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

inline std::ofstream& main_file() {
    static std::ofstream f;
    return f;
}

inline std::string& bound_run() {
    thread_local std::string s;
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
    return bound_run();
}

inline std::ostream& sink_for(Level lv) {
    return (lv == Level::warn || lv == Level::error) ? std::cerr : std::cout;
}

// True when a message at `lv` can never be emitted under the current config.
// Checked BEFORE any formatting so suppressed calls (log::off in tests and
// benches, debug below min_level in production) cost a couple of loads, not
// a std::format + vector allocation per call.
[[nodiscard]] inline bool suppressed(Level lv) noexcept {
    return !cfg().enabled || static_cast<int>(lv) < static_cast<int>(cfg().min_level);
}

// Optional banner extensions filled by the cuda umbrella. Core cannot include
// <cuda_runtime.h> (the one-way core ← cuda rule) nor see the nvcc version
// macros — those are only defined in the .cu TUs the cuda headers reach. So
// cuda/device_info.hpp sets these hooks at load time (a [[gnu::constructor]]);
// when non-null, banner() calls them. A pure-host build leaves them null and
// falls back to the compile-time host compiler / omits the gpu row. Raw
// function pointers, not a registry: nullable extension points, set once.
using BannerHook = std::string (*)();

// Device description for the `gpu` row (e.g. "Tesla T4 · sm_75 · …").
inline BannerHook& gpu_banner_hook() noexcept {
    static BannerHook hook = nullptr;
    return hook;
}

// nvcc toolkit version (e.g. "12.8") for the `compiler` row; when set, the row
// reads `nvcc <version> (<host compiler>)`.
inline BannerHook& nvcc_banner_hook() noexcept {
    static BannerHook hook = nullptr;
    return hook;
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

}  // namespace impl

// Fluent multi-line builder. Move-only. Destructor emits the whole entry
// atomically — no interleaving across threads.
class Entry {
public:
    // Suppression is latched at construction: a suppressed Entry skips all
    // line()/param() formatting, so building a never-emitted entry costs
    // nothing beyond the level check.
    Entry(Level lv, std::string_view tag) : lv_{lv}, tag_{tag}, suppressed_{impl::suppressed(lv)} {}
    Entry(Entry const&)            = delete;
    Entry& operator=(Entry const&) = delete;
    Entry(Entry&& other) noexcept
        : lv_{other.lv_}, tag_{std::move(other.tag_)}, lines_{std::move(other.lines_)},
          suppressed_{other.suppressed_}, emitted_{other.emitted_} {
        other.emitted_ = true;
    }
    Entry& operator=(Entry&&) = delete;

    // No ref-qualifier on .line()/.param(): the fluent chain
    // `log::info("tag").line(...).param(...)` works on the materialised
    // temporary because non-static methods may be called on a prvalue.
    // Holding an lvalue Entry (e.g. inside a class's `describe()` method)
    // and calling `.line()`/`.param()` incrementally also works.
    Entry& line(std::string_view s) {
        if (suppressed_) {
            return *this;
        }
        lines_.emplace_back(s);
        return *this;
    }
    template <class... Args>
    Entry& line(std::format_string<Args...> fmt, Args&&... a) {
        if (suppressed_) {
            return *this;
        }
        lines_.emplace_back(std::format(fmt, std::forward<Args>(a)...));
        return *this;
    }

    // Parameter line — same as line() but with a fixed 2-space indent so
    // parameters render visually nested under the concept name above them.
    Entry& param(std::string_view s) {
        if (suppressed_) {
            return *this;
        }
        std::string buf{"  "};
        buf.append(s);
        lines_.emplace_back(std::move(buf));
        return *this;
    }
    template <class... Args>
    Entry& param(std::format_string<Args...> fmt, Args&&... a) {
        if (suppressed_) {
            return *this;
        }
        std::string buf{"  "};
        buf.append(std::format(fmt, std::forward<Args>(a)...));
        lines_.emplace_back(std::move(buf));
        return *this;
    }

    Entry& operator<<(std::string_view s) {
        if (suppressed_) {
            return *this;
        }
        lines_.emplace_back(s);
        return *this;
    }

    ~Entry() {
        if (emitted_ || lines_.empty()) {
            return;
        }
        if (impl::suppressed(lv_)) {
            return;
        }
        emit();
    }

private:
    void emit();

    Level lv_;
    std::string tag_;
    std::vector<std::string> lines_;
    bool suppressed_ = false;
    bool emitted_    = false;
};

inline void Entry::emit() {
    using namespace impl;
    emitted_ = true;

    auto const sig  = sigil(lv_);
    auto const col  = color_seq(lv_);
    auto const rst  = color_reset();
    auto const ts   = format_elapsed();
    auto const tag4 = fit(tag_, 4);
    auto const run  = current_run();
    bool const par  = cfg().replicas;
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
    // Plain (no-ANSI) variants for the main log file.
    std::string const pfirst = std::format("{} {}{}  {}  ", sig, run_col, ts, tag4);
    std::string const pcont  = std::format("{} {}{}  {}  ", sig, blank_run, blank_ts, blank_tag);

    // One lock covers stdout/stderr emission AND the file writes.
    // Line-atomic for the whole multi-line Entry — no other thread can
    // interleave between our lines.
    std::scoped_lock const lk{sink_mutex()};

#ifdef _OPENMP
    if (omp_in_parallel() && run.empty()) {
        std::cerr << "⚠ logger: log call inside parallel region without scope()\n";
    }
#endif

    auto& out = sink_for(lv_);
    for (std::size_t i = 0; i < lines_.size(); ++i) {
        out << (i == 0 ? first : cont) << lines_[i] << rst << '\n';
    }

    if (auto& mf = main_file(); mf.is_open()) {
        for (std::size_t i = 0; i < lines_.size(); ++i) {
            mf << (i == 0 ? pfirst : pcont) << lines_[i] << '\n';
        }
        mf.flush();
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
    if (impl::suppressed(Level::debug)) {
        return;
    }
    Entry{Level::debug, tag}.line(fmt, std::forward<Args>(a)...);
}
template <class... Args>
inline void info(std::string_view tag, std::format_string<Args...> fmt, Args&&... a) {
    if (impl::suppressed(Level::info)) {
        return;
    }
    Entry{Level::info, tag}.line(fmt, std::forward<Args>(a)...);
}
template <class... Args>
inline void warn(std::string_view tag, std::format_string<Args...> fmt, Args&&... a) {
    if (impl::suppressed(Level::warn)) {
        return;
    }
    Entry{Level::warn, tag}.line(fmt, std::forward<Args>(a)...);
}
template <class... Args>
inline void error(std::string_view tag, std::format_string<Args...> fmt, Args&&... a) {
    if (impl::suppressed(Level::error)) {
        return;
    }
    Entry{Level::error, tag}.line(fmt, std::forward<Args>(a)...);
}

// RAII scope guard. Bind `run_id` to the current thread for the lifetime of
// this object; the previous binding is restored on destruction, so the same
// thread rebinds correctly across OpenMP loop iterations.
class Scope {
public:
    explicit Scope(std::string run_id)
        : prev_{std::exchange(impl::bound_run(), std::move(run_id))} {}
    ~Scope() { impl::bound_run() = std::move(prev_); }
    Scope(Scope const&)            = delete;
    Scope& operator=(Scope const&) = delete;
    Scope(Scope&&)                 = delete;
    Scope& operator=(Scope&&)      = delete;

private:
    std::string prev_;
};

[[nodiscard]] inline Scope scope(std::string_view run_id) {
    return Scope{std::string{run_id}};
}

// RAII global-suppression guard: silence ALL log output for its lifetime and
// restore the previous enabled state on destruction. Nests correctly (saves and
// restores rather than forcing on). Used to wrap noisy setup — e.g. per-replica
// ensemble construction, where every lattice / RNG / HMC ctor self-announces —
// so the driver can print one compact summary instead of N× the boilerplate.
class Quiet {
public:
    Quiet() : prev_{std::exchange(impl::cfg().enabled, false)} {}
    ~Quiet() { impl::cfg().enabled = prev_; }
    Quiet(Quiet const&)            = delete;
    Quiet& operator=(Quiet const&) = delete;
    Quiet(Quiet&&)                 = delete;
    Quiet& operator=(Quiet&&)      = delete;

private:
    bool prev_;
};

[[nodiscard]] inline Quiet quiet() {
    return Quiet{};
}

// Format a byte count with a binary unit, e.g. 2048 → "2.0 KiB". A display
// helper for log lines (slab footprints, buffer sizes, …).
[[nodiscard]] inline std::string human_bytes(std::size_t n) {
    auto v           = static_cast<double>(n);
    char const* unit = "B";
    for (char const* u : std::array<char const*, 3>{"KiB", "MiB", "GiB"}) {
        if (v < 1024.0) {
            break;
        }
        v /= 1024.0;
        unit = u;
    }
    return std::format("{:.1f} {}", v, unit);
}

// Init / config -------------------------------------------------------------

inline void set_min_level(Level lv) {
    impl::cfg().min_level = lv;
}
inline void set_color(bool on) {
    impl::cfg().color = on;
}

// Global on/off — overrides `min_level`. Cheaper than `set_min_level(warn)`
// because it short-circuits before any formatting. `banner()` is also
// suppressed while off.
inline void off() {
    impl::cfg().enabled = false;
}
inline void on() {
    impl::cfg().enabled = true;
}
[[nodiscard]] inline bool enabled() {
    return impl::cfg().enabled;
}

namespace impl {
inline bool& banner_shown() {
    static bool b = false;
    return b;
}
}  // namespace impl

// Hero banner. Idempotent — calling more than once is a no-op so apps and
// `log::start` can both invoke it safely.
//
// Heavy-rule frame (┏ ━ ┓ ┃ ┗ ━ ┛) so the left wall reuses the same ┃ as
// the log sigil — the banner flows visually into the log lines below.
// ANSI Shadow figlet for "reticolo"; version spliced into the bottom rule.
// Build metadata (branch/compiler/simd) is compile-time-baked via
// <reticolo/core/build_info.hpp>; the host/cpu/threads rows are read live via
// <reticolo/core/host_info.hpp>, and the gpu row is filled by the cuda umbrella
// through impl::gpu_banner_hook() when the app links the CUDA backend.
inline void banner() {
    if (!impl::cfg().enabled || impl::banner_shown()) {
        return;
    }
    impl::banner_shown() = true;
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

    std::scoped_lock const lk{impl::sink_mutex()};
    auto& mf  = impl::main_file();
    auto emit = [&](std::string const& s) {
        std::cout << s;
        if (mf.is_open()) {
            mf << s;
        }
    };

    emit("\n");
    emit(rule("┏", "┓", inner_width) + "\n");
    emit(blank_row + "\n");
    for (auto const& row : figlet) {
        emit("┃" + std::string(left_pad, ' ') + std::string{row} + std::string(right_pad, ' ') +
             "┃\n");
    }
    emit(blank_row + "\n");
    emit(bottom + "\n");

    // Metadata block — same ┃ sigil as log lines, no frame, so it bridges
    // into the run log naturally.
    auto const cores = host::logical_cores();
    std::string compiler_line{build::compiler};
    if (auto* const hook = impl::nvcc_banner_hook(); hook != nullptr) {
        if (auto const ver = hook(); !ver.empty()) {
            compiler_line = std::format("nvcc {} ({})", ver, build::compiler);
        }
    }
    emit(std::format("┃ branch   : {} @ {}\n", build::git_branch, build::git_commit));
    emit(std::format("┃ compiler : {}\n", compiler_line));
    emit(std::format("┃ build    : {} · {}\n", build::build_type, build::simd));
    emit(std::format("┃ host     : {}\n", host::name()));
    emit(std::format("┃ cpu      : {} · {} logical cores\n", host::cpu_brand(), cores));
    emit(std::format("┃ threads  : {}\n",
                     build::openmp_enabled ? std::format("OpenMP {} of {}", omp_threads, cores)
                                           : std::format("serial (1 of {})", cores)));
    if (auto* const hook = impl::gpu_banner_hook(); hook != nullptr) {
        if (auto const gpu = hook(); !gpu.empty()) {
            emit(std::format("┃ gpu      : {}\n", gpu));
        }
    }
    emit(std::format("┃ started  : {} (local)\n", impl::format_wall(impl::wall_start())));

    // Section break between banner metadata and the live log stream.
    // Heavy T-junction continues the running ┃ column into a horizontal rule.
    std::string sep{"┣"};
    for (std::size_t i = 0; i < inner_width; ++i) {
        sep += "━";
    }
    emit(sep + "\n");
    if (mf.is_open()) {
        mf.flush();
    }
}

// The single app entry point. Creates the workspace folder, opens the main
// log file <workspace>/<stem>.log (stem = out_name minus extension) and
// prints the banner. With replicas=true, scoped lines carry a run-id column
// (all output still goes to the single main log).
inline void
start(std::filesystem::path const& workspace, std::string_view out_name, bool replicas = false) {
    auto& c     = impl::cfg();
    c.replicas  = replicas;
    c.workspace = workspace.empty() ? std::filesystem::path{"."} : workspace;
    c.stem      = std::filesystem::path{out_name}.stem().string();
    c.color     = impl::detect_color(fileno(stdout));
    impl::mono_start();
    impl::wall_start();

    std::error_code ec;
    std::filesystem::create_directories(c.workspace, ec);
    auto const log_path = c.workspace / (c.stem + ".log");
    auto& mf            = impl::main_file();
    if (mf.is_open()) {
        mf.close();
    }
    mf.open(log_path);
    if (!mf.is_open()) {
        std::cerr << "⚠ logger: cannot open " << log_path.string() << " — console only\n";
    }
    banner();
}

}  // namespace reticolo::log
