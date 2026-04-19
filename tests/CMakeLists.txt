add_executable(lattice lattice.cpp)
target_link_libraries(lattice PRIVATE ${PROJECT_NAME})

add_executable(lattice_uptr lattice_uptr.cpp)
target_link_libraries(lattice_uptr PRIVATE ${PROJECT_NAME})

add_executable(field_rw field_rw.cpp)
target_link_libraries(field_rw PRIVATE ${PROJECT_NAME})

add_executable(hdf5_basic hdf5_basic.cpp)
target_link_libraries(hdf5_basic PRIVATE ${PROJECT_NAME})

add_executable(hdf5_expandable_lattice hdf5_expandable_lattice.cpp)
target_link_libraries(hdf5_expandable_lattice PRIVATE ${PROJECT_NAME})

add_executable(registry_metadata registry_metadata.cpp)
target_link_libraries(registry_metadata PRIVATE ${PROJECT_NAME})

add_executable(runtime_api runtime_api.cpp)
target_link_libraries(runtime_api PRIVATE ${PROJECT_NAME})

add_test(NAME lattice COMMAND lattice 4 1)
add_test(NAME lattice_uptr COMMAND lattice_uptr 4 1)
add_test(NAME field_rw COMMAND field_rw)
add_test(NAME hdf5_basic COMMAND hdf5_basic)
add_test(NAME hdf5_expandable_lattice COMMAND hdf5_expandable_lattice)
add_test(NAME registry_metadata COMMAND registry_metadata)
add_test(NAME runtime_api COMMAND runtime_api)
