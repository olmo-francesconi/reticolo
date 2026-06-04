# In-tree aggregator: compile every example that ships a CMakeLists against the
# already-configured reticolo::reticolo target (their find-or-fetch block is a
# no-op here). Only reached when RETICOLO_BUILD_EXAMPLES=ON; primarily a CI
# compile-check. `10_smoothed_synthetic_benchmark` is python-only (no binary).
add_subdirectory(01_phi4_tuning)
add_subdirectory(02_on_sigma_critical)
add_subdirectory(03_phi4_algorithm_tuning)
add_subdirectory(04_phi4_llr)
add_subdirectory(05_u1_llr)
add_subdirectory(06_bose_gas_llr)
add_subdirectory(07_su2_llr)
add_subdirectory(08_volume_scaling)
add_subdirectory(09_u1_llr_smoothed)
