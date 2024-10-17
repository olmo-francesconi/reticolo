# Make an automatic library - will be static or dynamic based on user setting
add_library(${PROJECT_NAME} INTERFACE)

# link external libraries
target_link_libraries(${PROJECT_NAME} INTERFACE hdf5::hdf5)
target_link_libraries(${PROJECT_NAME} INTERFACE OpenMP::OpenMP_CXX)
target_link_libraries(${PROJECT_NAME} INTERFACE yaml-cpp::yaml-cpp)
target_link_libraries(${PROJECT_NAME} INTERFACE cxxopts::cxxopts)

# include
target_include_directories(${PROJECT_NAME} INTERFACE .)

# All users of this library will need at least C++20
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)

# Set the lattice size variable
if (DEFINED SMALL_LATTICE)
    if (SMALL_LATTICE)
        target_compile_definitions(${PROJECT_NAME} INTERFACE SMALL_LATTICE)
    endif()
endif()


# # Library
# add_library(${PROJECT_NAME} SHARED

#     reticolo/tools/logger.cpp
# )

# # Link libraries
# target_link_libraries(${PROJECT_NAME} PUBLIC hdf5::hdf5)
# target_link_libraries(${PROJECT_NAME} PUBLIC yaml-cpp::yaml-cpp)
# target_link_libraries(${PROJECT_NAME} PUBLIC cxxopts::cxxopts)

# # properties
# set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION})

# # compile features
# target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)

# # includes
# target_include_directories(${PROJECT_NAME} PUBLIC .)

# # install rules
# install(TARGETS ${PROJECT_NAME}
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
# # PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

