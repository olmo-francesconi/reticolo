#pragma once

#include <reticolo/core/log.hpp>
#include <reticolo/io/writer.hpp>

#include <concepts>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <cxxopts.hpp>

// =============================================================================
//  Command-line parser.
//
//  - `req<T>("name", "desc")` and `opt<T>("name", default, "desc")` register
//    a variable and return a `T const&` to stable storage owned by the parser.
//    The reference is valid only AFTER `parse(argc, argv)` returns.
//  - `parse` is one cxxopts call under the hood. Throws on missing required,
//    bad value, or unknown flag; prints help and exits on `--help` / `-h`.
//  - `stamp_into(io::Writer&)` iterates every registered slot and writes its
//    resolved value to `/vars@<name>`. The `Writer` constructor accepts a
//    `Parser const*` so apps usually never call this explicitly.
//
//  Supported value types: int, long, long long, unsigned int, unsigned long,
//  unsigned long long, float, double, std::string. The set is intersected with
//  what `io::Writer::attr<T>` instantiates, so every registered var stamps.
// =============================================================================

namespace reticolo::cli {

template <class T>
concept SupportedVarType =
    std::same_as<T, int> || std::same_as<T, long> || std::same_as<T, long long> ||
    std::same_as<T, unsigned int> || std::same_as<T, unsigned long> ||
    std::same_as<T, unsigned long long> || std::same_as<T, float> || std::same_as<T, double> ||
    std::same_as<T, std::string>;

namespace detail {

template <class T>
std::string to_default_string(T const& v) {
    if constexpr (std::same_as<T, std::string>) {
        return v;
    } else {
        return std::to_string(v);
    }
}

struct VarSlotBase {
    std::string name;  // cxxopts spec; "short,long" or a single name.
    std::string desc;
    bool required = false;

    VarSlotBase()                                  = default;
    VarSlotBase(VarSlotBase const&)                = delete;
    VarSlotBase& operator=(VarSlotBase const&)     = delete;
    VarSlotBase(VarSlotBase&&) noexcept            = default;
    VarSlotBase& operator=(VarSlotBase&&) noexcept = default;
    virtual ~VarSlotBase()                         = default;

    virtual void add_to(cxxopts::Options& opts) const          = 0;
    virtual void read_from(cxxopts::ParseResult const& result) = 0;
    virtual void stamp_into(io::Writer& w) const               = 0;

    // The canonical key cxxopts uses to look up this var: the long name if
    // "short,long" was given, otherwise the only name. Also used as the
    // /vars@<canonical_name> attribute.
    [[nodiscard]] std::string canonical_name() const {
        auto const comma = name.find(',');
        if (comma == std::string::npos) {
            return name;
        }
        return name.substr(comma + 1);
    }
};

template <SupportedVarType T>
struct VarSlot : VarSlotBase {
    T value{};
    std::optional<T> default_value;

    void add_to(cxxopts::Options& opts) const override {
        auto spec = cxxopts::value<T>();
        if (default_value.has_value()) {
            spec = spec->default_value(to_default_string(*default_value));
        }
        // Accepts cxxopts' `"short,long"` format, or a single name (long if
        // ≥2 chars per cxxopts' rule, short otherwise). For physics shorthand
        // like `L`, write `"L,lattice"` to get both `-L` and `--lattice`.
        opts.add_options()(name, desc, spec);
    }

    void read_from(cxxopts::ParseResult const& result) override {
        auto const key = canonical_name();
        if (required && result.count(key) == 0) {
            throw std::runtime_error{"cli::Parser: required flag --" + key + " missing"};
        }
        value = result[key].template as<T>();
    }

    void stamp_into(io::Writer& w) const override { w.attr<T>("/vars@" + canonical_name(), value); }
};

}  // namespace detail

class Parser {
public:
    Parser(std::string prog_name, std::string prog_desc = {})
        : name_{std::move(prog_name)}, desc_{std::move(prog_desc)} {}

    Parser(Parser const&)                = delete;
    Parser& operator=(Parser const&)     = delete;
    Parser(Parser&&) noexcept            = default;
    Parser& operator=(Parser&&) noexcept = default;
    ~Parser()                            = default;

    // Register a required flag. Returns a reference to its storage that
    // becomes meaningful after `parse(...)` returns.
    template <SupportedVarType T>
    T const& req(std::string name, std::string desc = {}) {
        auto slot      = std::make_unique<detail::VarSlot<T>>();
        slot->name     = std::move(name);
        slot->desc     = std::move(desc);
        slot->required = true;
        T const& ref   = slot->value;
        slots_.push_back(std::move(slot));
        return ref;
    }

    // Register an optional flag with a default. Reference is valid before
    // parse() (holds the default) and may be updated by parse().
    template <SupportedVarType T>
    T const& opt(std::string name, T default_value, std::string desc = {}) {
        auto slot           = std::make_unique<detail::VarSlot<T>>();
        slot->name          = std::move(name);
        slot->desc          = std::move(desc);
        slot->required      = false;
        slot->default_value = default_value;
        slot->value         = default_value;
        T const& ref        = slot->value;
        slots_.push_back(std::move(slot));
        return ref;
    }

    // Parse argv. Returns true on a normal parse. Returns false (after
    // printing usage to `out`) when the user passed `--help` / `-h` —
    // apps write `if (!p.parse(argc, argv)) return 0;` as the canonical
    // first line of main, which lets the app's RAII destructors run cleanly
    // instead of being bypassed by `std::exit`. Bad-argument failures
    // (missing required flag, parse error, unknown flag) propagate as
    // `std::runtime_error` / `cxxopts::exceptions::*`.
    bool parse(int argc, char const* const* argv, std::ostream& out = std::cout) {
        cxxopts::Options opts{name_, desc_};
        opts.add_options()("h,help", "Show this help and exit");
        for (auto const& s : slots_) {
            s->add_to(opts);
        }

        cxxopts::ParseResult result;
        try {
            result = opts.parse(argc, argv);
        } catch (cxxopts::exceptions::exception const& e) {
            log::error("cli", "{}", e.what());
            throw;
        }

        if (result.count("help") > 0) {
            out << opts.help() << '\n';
            return false;
        }

        try {
            for (auto& s : slots_) {
                s->read_from(result);
            }
        } catch (std::exception const& e) {
            log::error("cli", "{}", e.what());
            throw;
        }
        parsed_ = true;
        return true;
    }

    // Stamp every registered var at /vars@<name>. Called automatically by the
    // Writer constructor when given a Parser pointer.
    void stamp_into(io::Writer& w) const {
        if (!parsed_) {
            throw std::runtime_error{"cli::Parser::stamp_into called before parse()"};
        }
        for (auto const& s : slots_) {
            s->stamp_into(w);
        }
    }

    [[nodiscard]] bool parsed() const noexcept { return parsed_; }

private:
    std::string name_;
    std::string desc_;
    std::vector<std::unique_ptr<detail::VarSlotBase>> slots_;
    bool parsed_ = false;
};

}  // namespace reticolo::cli
