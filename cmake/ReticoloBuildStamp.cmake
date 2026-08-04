# Build-time git stamping.
#
# `/run@commit` in every HDF5 file is the run record — CLAUDE.md's "argv + git
# SHA is the run record" invariant depends on it being the commit that actually
# produced the binary. Capturing it with execute_process() at configure time
# does not do that: it freezes on the last `cmake` invocation, so any commit (or
# branch switch) made afterwards is silently misrecorded.
#
# So the stamp is regenerated at BUILD time into a header, and attached to the
# ONE target that writes the record (reticolo::io). Two consequences worth
# knowing:
#
#   * A commit changes exactly one TU's inputs (src/io/writer.cpp), not every TU
#     in the project — which is what would happen if the values stayed as
#     compile definitions on the header-only reticolo::core INTERFACE.
#   * `reticolo::core` stays header-only. Routing the stamp through
#     core/sys/build_info.hpp instead would pull it into log.hpp, hence into
#     essentially every TU, and every commit would trigger a full rebuild.
#
# The console banner (log.hpp) still shows the configure-time values, so it can
# lag behind the file stamp between reconfigures. That is the pre-existing
# behaviour and it is cosmetic; the authoritative record is the HDF5 attribute.

function(reticolo_attach_build_stamp target)
    set(_dir "${PROJECT_BINARY_DIR}/generated")
    set(_hdr "${_dir}/reticolo_build_stamp.h")
    file(MAKE_DIRECTORY "${_dir}")

    # Generate once now so the header exists for the very first compile (and for
    # tooling / compile_commands.json consumers that parse before any build).
    # OUTPUT_QUIET: section 3 reports the stamp, so the script's own line would
    # be a duplicate landing in the middle of the dependency section. At build
    # time it is not quiet — there the line is the only notice you get.
    execute_process(COMMAND ${CMAKE_COMMAND}
        -DRETICOLO_SRC=${PROJECT_SOURCE_DIR} -DRETICOLO_OUT=${_hdr}
        -P ${PROJECT_SOURCE_DIR}/cmake/WriteBuildStamp.cmake
        OUTPUT_QUIET)

    # An always-out-of-date target: the script runs on every build but rewrites
    # the header only when the commit/branch actually differ, so the dependent
    # TU recompiles only on a real change.
    if(NOT TARGET reticolo_build_stamp)
        add_custom_target(reticolo_build_stamp
            BYPRODUCTS "${_hdr}"
            COMMAND ${CMAKE_COMMAND}
                    -DRETICOLO_SRC=${PROJECT_SOURCE_DIR} -DRETICOLO_OUT=${_hdr}
                    -P ${PROJECT_SOURCE_DIR}/cmake/WriteBuildStamp.cmake
            COMMENT "reticolo: checking git build stamp"
            VERBATIM
        )
    endif()

    add_dependencies(${target} reticolo_build_stamp)
    target_include_directories(${target} PRIVATE "${_dir}")
endfunction()
