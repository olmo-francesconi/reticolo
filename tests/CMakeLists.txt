add_executable(lattice lattice.cpp)
target_link_libraries(lattice PRIVATE ${PROJECT_NAME})


add_executable(lattice_uptr lattice_uptr.cpp)
target_link_libraries(lattice_uptr PRIVATE ${PROJECT_NAME})

add_executable(field_rw field_rw.cpp)
target_link_libraries(field_rw PRIVATE ${PROJECT_NAME})