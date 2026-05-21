#pragma once

#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace reticolo::tune {

// Raw-binary load: `nsites()` doubles in lattice index order. Caller must
// pass --n_therm=0 when using this; we don't enforce it but the warmup
// would overwrite the loaded state.
template <class Field>
void load_field_raw(Field& field, std::string const& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        throw std::runtime_error("load_field_raw: cannot open " + path);
    }
    std::size_t const got = std::fread(field.data(), sizeof(double), field.nsites(), f);
    std::fclose(f);
    if (got != field.nsites()) {
        throw std::runtime_error("load_field_raw: shape mismatch (got " +
                                 std::to_string(got) + " doubles, expected " +
                                 std::to_string(field.nsites()) + ")");
    }
}

template <class Field>
void save_field_raw(Field const& field, std::string const& path) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        throw std::runtime_error("save_field_raw: cannot open " + path + " for writing");
    }
    std::fwrite(field.data(), sizeof(double), field.nsites(), f);
    std::fclose(f);
}

}  // namespace reticolo::tune
