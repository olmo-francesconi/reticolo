// Single translation unit whose only purpose is to give clang-tidy a complete
// view of the public umbrella header. CI tidies this TU instead of running
// once per app: every public header is reached through `<reticolo/reticolo.hpp>`
// and surfaced via the HeaderFilterRegex in `.clang-tidy`, so tidy cost stays
// constant as the app set grows.
//
// This TU is not linked into any binary; it exists purely for the lint scan.

#include <reticolo/reticolo.hpp>

namespace {
[[maybe_unused]] inline constexpr int reticolo_amalgamation_marker = 0;
}  // namespace
