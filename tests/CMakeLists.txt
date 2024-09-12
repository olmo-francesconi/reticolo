add_executable(lattice lattice.cpp)
target_link_libraries(lattice PRIVATE reticolo)


add_executable(lattice_uptr lattice_uptr.cpp)
target_link_libraries(lattice_uptr PRIVATE reticolo)