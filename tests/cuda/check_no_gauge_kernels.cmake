# Lint gate: "zero re-implemented per-link SU(2)/SU(3) matrix algebra in the
# CUDA device path." mul/mul_adj/adj_mul/traceless_antiherm/expi in SU2Device
# and SU3Device (cuda/gauge/su{2,3}_device.cuh) must be one-line forwards to
# the single-sourced math::su{2,3}:: formula functions
# (math/group/formula/su{2,3}_formula.hpp) — never a hand-copied
# re-implementation. Checks two things per device file (`//` comments stripped
# first, so a prose mention of the algorithm in a header comment can't trip
# this):
#   1. each of the 5 forwarding calls is present verbatim;
#   2. tell-tale identifiers from the old hand-copied bodies (`c0_max`, the
#      SU(3) Cayley-Hamilton coefficient; `cset(`, the old SU(2) private
#      complex-multiply-accumulate helper) are gone — they only reappear if
#      the real math gets pasted back in.
#
# Scanned: SRC_DIR/gauge/su2_device.cuh, SRC_DIR/gauge/su3_device.cuh.
#
# Invoked as: cmake -DSRC_DIR=<include/reticolo/cuda> -P check_no_gauge_kernels.cmake

set(su2_file "${SRC_DIR}/gauge/su2_device.cuh")
set(su3_file "${SRC_DIR}/gauge/su3_device.cuh")
foreach(f "${su2_file}" "${su3_file}")
    if(NOT EXISTS "${f}")
        message(FATAL_ERROR "check_no_gauge_kernels.cmake: missing ${f}")
    endif()
endforeach()

set(offenders "")

file(READ "${su2_file}" su2_content)
string(REGEX REPLACE "//[^\n]*" "" su2_content "${su2_content}")
foreach(needle
        "math::su2::mul_2x2(o, a, b);"
        "math::su2::mul_adj_2x2(o, a, b);"
        "math::su2::adj_mul_2x2(o, a, b);"
        "math::su2::traceless_antiherm_2x2(out, in);"
        "math::su2::exp_su2(v, p, dt);")
    string(FIND "${su2_content}" "${needle}" pos)
    if(pos EQUAL -1)
        list(APPEND offenders "su2_device.cuh: missing forward '${needle}'")
    endif()
endforeach()
if(su2_content MATCHES "cset\\(")
    list(APPEND offenders "su2_device.cuh: re-implemented complex-multiply helper (cset) found")
endif()

file(READ "${su3_file}" su3_content)
string(REGEX REPLACE "//[^\n]*" "" su3_content "${su3_content}")
foreach(needle
        "math::su3::mul_3x3(out, a, b);"
        "math::su3::mul_adj_3x3(out, a, b);"
        "math::su3::adj_mul_3x3(out, a, b);"
        "math::su3::traceless_antiherm_3x3(out, in);"
        "math::su3::exp_su3(v, p, dt);")
    string(FIND "${su3_content}" "${needle}" pos)
    if(pos EQUAL -1)
        list(APPEND offenders "su3_device.cuh: missing forward '${needle}'")
    endif()
endforeach()
if(su3_content MATCHES "c0_max")
    list(APPEND offenders "su3_device.cuh: re-implemented Cayley-Hamilton coefficients (c0_max) found")
endif()

if(offenders)
    message(FATAL_ERROR "gauge matrix algebra re-implemented in CUDA device path: ${offenders}")
endif()
message(STATUS "no re-implemented gauge matrix algebra in CUDA device path")
