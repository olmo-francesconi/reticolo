set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)

doxygen_add_docs(
    docs 
    src/reticolo.hpp 
    "${CMAKE_CURRENT_SOURCE_DIR}/mainpage.md")
