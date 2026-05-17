#pragma once

#include <chrono>
#include <iostream>
#include <ostream>
#include <string>
#include <string_view>

namespace reticolo::log {

namespace detail {

inline std::ostream*& stream_ptr() noexcept {
    static std::ostream* p = &std::clog;
    return p;
}

inline int& depth() noexcept {
    thread_local int d = 0;
    return d;
}

inline std::string indent() {
    return std::string(static_cast<std::size_t>(depth()) * 2, ' ');
}

}  // namespace detail

inline std::ostream& stream() noexcept {
    return *detail::stream_ptr();
}

inline void set_stream(std::ostream& os) noexcept {
    detail::stream_ptr() = &os;
}

inline void info(std::string_view msg) {
    stream() << detail::indent() << "[i] " << msg << '\n';
}

inline void warn(std::string_view msg) {
    stream() << detail::indent() << "[!] " << msg << '\n';
}

inline void error(std::string_view msg) {
    stream() << detail::indent() << "[E] " << msg << '\n';
}

// RAII timing block. Prints `+ name` on entry, `- name (Xms)` on exit;
// nested sections indent.
class Section {
public:
    explicit Section(std::string name)
        : name_{std::move(name)}, start_{std::chrono::steady_clock::now()} {
        stream() << detail::indent() << "+ " << name_ << '\n';
        ++detail::depth();
    }

    Section(Section const&)            = delete;
    Section& operator=(Section const&) = delete;
    Section(Section&&)                 = delete;
    Section& operator=(Section&&)      = delete;

    ~Section() {
        --detail::depth();
        auto const dt = std::chrono::steady_clock::now() - start_;
        auto const ms = std::chrono::duration_cast<std::chrono::milliseconds>(dt).count();
        stream() << detail::indent() << "- " << name_ << " (" << ms << " ms)\n";
    }

private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace reticolo::log
