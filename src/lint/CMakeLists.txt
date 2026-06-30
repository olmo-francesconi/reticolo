# Tiny static lib whose only TU is `amalgamation.cpp`. Builds in a few seconds,
# produces an unused .a, but lands in compile_commands.json so clang-tidy can
# analyse the umbrella header end-to-end in one pass.
# Not EXCLUDE_FROM_ALL on purpose: the TU compiles in a couple of seconds and
# being part of the default build guarantees its compile_commands.json entry is
# fresh after every configure, so run-clang-tidy can find it without surprises.
add_library(reticolo_lint_amalgamation STATIC amalgamation.cpp)
target_link_libraries(reticolo_lint_amalgamation PRIVATE reticolo::reticolo)
target_compile_features(reticolo_lint_amalgamation PRIVATE cxx_std_20)
reticolo_configure_warnings(reticolo_lint_amalgamation)
