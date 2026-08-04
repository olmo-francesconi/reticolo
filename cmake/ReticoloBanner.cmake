# Configure-time console output.
#
# Deliberately mirrors the RUNTIME banner in include/reticolo/core/log/log.hpp:
# same ANSI Shadow figlet, same ┏━┓ frame, same `┃ key : value` metadata block,
# same ┣━ section rules. A configure and a run should look like the same tool.
#
# Everything framed goes through message(NOTICE), which prints verbatim —
# message(STATUS) would prefix every line with "-- " and wreck the logo.
# NOTICE writes to stderr, STATUS to stdout, so capture with `2>&1` if you want
# one interleaved log (tools/check.sh and the CI workflow already do).
#
# Third-party output (find_package / FetchContent subprojects) keeps its own
# "-- " STATUS prefix and is indented under its dependency heading via
# CMAKE_MESSAGE_INDENT. The visual split is intentional: `┃` is reticolo
# talking, `--` is somebody else's build system.

set(RETICOLO_BANNER_WIDTH 74)  # inner width, matches log.hpp's frame

function(reticolo_note text)
    message(NOTICE "${text}")
endfunction()

function(_reticolo_repeat glyph count out_var)
    set(_s "")
    foreach(_i RANGE 1 ${count})
        string(APPEND _s "${glyph}")
    endforeach()
    set(${out_var} "${_s}" PARENT_SCOPE)
endfunction()

# ┃ text
function(reticolo_line text)
    message(NOTICE "┃ ${text}")
endfunction()

# ┣━━━━━━ (section rule)
function(reticolo_rule)
    _reticolo_repeat("━" ${RETICOLO_BANNER_WIDTH} _bar)
    message(NOTICE "┣${_bar}")
endfunction()

# ┗━━━━━━ (close)
function(reticolo_close)
    _reticolo_repeat("━" ${RETICOLO_BANNER_WIDTH} _bar)
    message(NOTICE "┗${_bar}")
endfunction()

# The logo. Printed BEFORE project() so it is genuinely the first thing on a
# cold configure; the compiler-identification lines project() emits are indented
# underneath it (and vanish on every reconfigure, being cached).
function(reticolo_logo version tag)
    set(_figlet
        "██████╗ ███████╗████████╗██╗ ██████╗ ██████╗ ██╗      ██████╗ "
        "██╔══██╗██╔════╝╚══██╔══╝██║██╔════╝██╔═══██╗██║     ██╔═══██╗"
        "██████╔╝█████╗     ██║   ██║██║     ██║   ██║██║     ██║   ██║"
        "██╔══██╗██╔══╝     ██║   ██║██║     ██║   ██║██║     ██║   ██║"
        "██║  ██║███████╗   ██║   ██║╚██████╗╚██████╔╝███████╗╚██████╔╝"
        "╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝ ╚═════╝ "
    )
    # Figlet rows are 62 display cells; centre them in the frame.
    set(_pad_l 5)
    _reticolo_repeat("━" ${RETICOLO_BANNER_WIDTH} _bar)
    _reticolo_repeat(" " ${RETICOLO_BANNER_WIDTH} _blank)
    _reticolo_repeat(" " ${_pad_l} _lpad)
    math(EXPR _pad_r "${RETICOLO_BANNER_WIDTH} - 62 - ${_pad_l}")
    _reticolo_repeat(" " ${_pad_r} _rpad)

    message(NOTICE "")
    message(NOTICE "┏${_bar}┓")
    message(NOTICE "┃${_blank}┃")
    foreach(_row IN LISTS _figlet)
        message(NOTICE "┃${_lpad}${_row}${_rpad}┃")
    endforeach()
    message(NOTICE "┃${_blank}┃")

    # Bottom rule carries the phase tag + version, like log.hpp splices v<ver>.
    set(_tag " ${tag} ━ v${version} ")
    # string(LENGTH) counts BYTES. The tag contains exactly one ━ (3 bytes in
    # UTF-8), so its display width is byte length - 2. Everything else is ASCII.
    string(LENGTH "${_tag}" _tag_bytes)
    math(EXPR _tag_len "${_tag_bytes} - 2")
    math(EXPR _dashes "${RETICOLO_BANNER_WIDTH} - ${_tag_len} - 1")
    _reticolo_repeat("━" ${_dashes} _lead)
    message(NOTICE "┗${_lead}${_tag}━┛")
endfunction()

# --- dependency mini-sections -------------------------------------------------
# Each dependency prints a heading, lets its own find_package/FetchContent chatter
# land indented underneath, then a one-line verdict. The verdict is also appended
# to a global list so section 2 can close with a recap.

function(reticolo_dep_begin name kind)
    # Prefix is "┃ ── <name> ── <kind> " = 10 display cells + the two labels;
    # the line spans ┃ plus RETICOLO_BANNER_WIDTH cells.
    string(LENGTH "${name}${kind}" _n)
    math(EXPR _dashes "${RETICOLO_BANNER_WIDTH} - 9 - ${_n}")
    if(_dashes LESS 3)
        set(_dashes 3)
    endif()
    _reticolo_repeat("─" ${_dashes} _bar)
    message(NOTICE "┃")
    message(NOTICE "┃ ── ${name} ── ${kind} ${_bar}")
    list(APPEND CMAKE_MESSAGE_INDENT "    ")
    set(CMAKE_MESSAGE_INDENT "${CMAKE_MESSAGE_INDENT}" PARENT_SCOPE)
endfunction()

function(reticolo_dep_end)
    list(POP_BACK CMAKE_MESSAGE_INDENT)
    set(CMAKE_MESSAGE_INDENT "${CMAKE_MESSAGE_INDENT}" PARENT_SCOPE)
endfunction()

# status is one of: ok | skip. `detail` is free text, `consumer` the target(s).
function(reticolo_dep_record status name version kind consumer detail)
    if(status STREQUAL "ok")
        # Pad the version (ASCII) so every arrow lines up down the section.
        set(_v "${version}")
        string(APPEND _v "                    ")
        string(SUBSTRING "${_v}" 0 12 _v)
        set(_d "${detail}")
        string(APPEND _d "                                        ")
        string(SUBSTRING "${_d}" 0 34 _d)
        message(NOTICE "┃    ✓ ${_v}${_d}→ ${consumer}")
    else()
        # A skip has no version and no consumer — say why instead of leaving
        # empty columns that read like a formatting bug.
        message(NOTICE "┃    ⊘ skipped — ${detail}")
    endif()
    # The recap's last column shows the consumer for a resolved dep and the
    # reason for a skipped one — an empty cell there reads as a bug.
    if(status STREQUAL "ok")
        set(_last "${consumer}")
    else()
        set(_last "${detail}")
    endif()
    set_property(GLOBAL APPEND PROPERTY RETICOLO_DEP_ROWS
                 "${name}|${version}|${kind}|${_last}|${status}")
endfunction()

function(reticolo_dep_recap)
    get_property(_rows GLOBAL PROPERTY RETICOLO_DEP_ROWS)
    set(_n 0)
    set(_ok 0)
    set(_skip 0)
    foreach(_r IN LISTS _rows)
        math(EXPR _n "${_n} + 1")
        string(REPLACE "|" ";" _f "${_r}")
        list(GET _f 4 _st)
        if(_st STREQUAL "ok")
            math(EXPR _ok "${_ok} + 1")
        else()
            math(EXPR _skip "${_skip} + 1")
        endif()
    endforeach()
    message(NOTICE "┃")
    message(NOTICE "┃ dependencies                     ${_n} required · ${_ok} resolved · ${_skip} skipped")
    foreach(_r IN LISTS _rows)
        string(REPLACE "|" ";" _f "${_r}")
        list(GET _f 0 _name)
        list(GET _f 1 _ver)
        list(GET _f 2 _kind)
        list(GET _f 3 _use)
        list(GET _f 4 _st)
        if(_st STREQUAL "ok")
            set(_shown "${_ver}")
        else()
            set(_shown "-")  # ASCII: the columns are padded by BYTE length
        endif()
        string(APPEND _name "            ")
        string(SUBSTRING "${_name}" 0 10 _name)
        string(APPEND _shown "            ")
        string(SUBSTRING "${_shown}" 0 10 _shown)
        string(APPEND _kind "            ")
        string(SUBSTRING "${_kind}" 0 10 _kind)
        message(NOTICE "┃   ${_name}${_shown}${_kind}${_use}")
    endforeach()
endfunction()

# --- build-phase banner -------------------------------------------------------
# `cmake --build` has no hook for "print this once at the start", so the logo is
# an always-dirty custom target that everything else reaches through
# reticolo::io. USES_TERMINAL puts it in Ninja's console pool, so it is
# serialised rather than interleaved with compile output.
#
# The console pool is what makes this work: Ninja gives the target the terminal
# and suppresses its own status lines while it runs, so the logo can never be
# interleaved mid-frame with compile output. Verified landing first on both a
# cold build (where sleef/Catch2 are also ready to run) and an incremental one.
function(reticolo_attach_build_banner target)
    if(NOT TARGET reticolo_banner)
        add_custom_target(reticolo_banner
            COMMAND ${CMAKE_COMMAND}
                    -DRETICOLO_LOGO_VERSION=${PROJECT_VERSION}
                    -P ${PROJECT_SOURCE_DIR}/cmake/PrintBanner.cmake
            COMMENT "reticolo"   # else Ninja echoes the whole cmake -P command line
            USES_TERMINAL
            VERBATIM
        )
    endif()
    add_dependencies(${target} reticolo_banner)
endfunction()
