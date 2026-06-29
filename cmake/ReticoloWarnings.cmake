# Project warning policy.
#
# Apply with `reticolo_configure_warnings(<target>)`. The target may be
# INTERFACE, STATIC, or executable; flags are added as INTERFACE for
# INTERFACE targets and PRIVATE otherwise.
#
# All flags are guarded by `$<COMPILE_LANGUAGE:CXX>` so they apply only to C++
# compiles, never CUDA. This matters transitively: reticolo::core is INTERFACE
# and reticolo::io PUBLIC-links it, so a nvcc-compiled target that links io would
# otherwise inherit host -Wall/-Werror on its .cu compile and break. nvcc carries
# its own diagnostics; the host warning policy is a C++-only concern.

function(reticolo_configure_warnings target)
    get_target_property(_type ${target} TYPE)
    if(_type STREQUAL "INTERFACE_LIBRARY")
        set(_scope INTERFACE)
    else()
        set(_scope PRIVATE)
    endif()

    set(_gnu_clang_warnings
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Wcast-align
        -Wunused
        -Woverloaded-virtual
        -Wconversion
        -Wsign-conversion
        -Wdouble-promotion
        -Wformat=2
        -Wimplicit-fallthrough
    )
    # -Wnull-dereference: dropped. GCC's static analyzer produces false positives
    # at -O3 on std::vector::operator[] (cannot prove data() != nullptr after
    # sized construction). Real null derefs are caught by the asan/ubsan job.

    # Clang's -Wpedantic flags __COUNTER__ via -Wc2y-extensions; Catch2's
    # TEST_CASE expansion lives in user TUs, so SYSTEM doesn't suppress it.
    # -Wno-unknown-warning-option goes first so older clang versions silently
    # accept the c2y-specific flag they don't recognise.
    set(_clang_quiet -Wno-unknown-warning-option -Wno-c2y-extensions)

    set(_gcc_extra
        -Wmisleading-indentation
        -Wduplicated-cond
        -Wduplicated-branches
        -Wlogical-op
        -Wuseless-cast
    )

    set(_msvc_warnings /W4 /permissive-)

    # Guard every flag by COMPILE_LANGUAGE:CXX (see header comment) so CUDA
    # compiles that link these targets never inherit the host warning policy.
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${target} ${_scope}
            "$<$<COMPILE_LANGUAGE:CXX>:${_gnu_clang_warnings};${_clang_quiet}>")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${target} ${_scope}
            "$<$<COMPILE_LANGUAGE:CXX>:${_gnu_clang_warnings};${_gcc_extra}>")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${target} ${_scope}
            "$<$<COMPILE_LANGUAGE:CXX>:${_msvc_warnings}>")
    endif()

    if(RETICOLO_WARNINGS_AS_ERRORS)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
            target_compile_options(${target} ${_scope} "$<$<COMPILE_LANGUAGE:CXX>:-Werror>")
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            target_compile_options(${target} ${_scope} "$<$<COMPILE_LANGUAGE:CXX>:/WX>")
        endif()
    endif()
endfunction()
