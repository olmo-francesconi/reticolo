# Make an automatic library - will be static or dynamic based on user setting
add_library(${PROJECT_NAME} INTERFACE)

# target HDF5
target_link_libraries(${PROJECT_NAME} INTERFACE hdf5::hdf5)

# include
target_include_directories(${PROJECT_NAME} INTERFACE ${PROJECT_SOURCE_DIR}/src)

# All users of this library will need at least C++20
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)
