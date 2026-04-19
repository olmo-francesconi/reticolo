# Make an automatic library - will be static or dynamic based on user setting
add_library(${PROJECT_NAME} INTERFACE)
add_library(reticolo::reticolo ALIAS ${PROJECT_NAME})

reticolo_configure_warnings(${PROJECT_NAME})

if(TARGET hdf5::hdf5)
    target_link_libraries(${PROJECT_NAME} INTERFACE hdf5::hdf5)
elseif(TARGET HDF5::HDF5)
    target_link_libraries(${PROJECT_NAME} INTERFACE HDF5::HDF5)
else()
    target_include_directories(${PROJECT_NAME} SYSTEM INTERFACE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} INTERFACE ${HDF5_LIBRARIES})
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} INTERFACE OpenMP::OpenMP_CXX)
    target_compile_definitions(${PROJECT_NAME} INTERFACE RETICOLO_USE_OPENMP)
endif()

if(DEFINED RETICOLO_HDF5_VERSION_MAJOR)
    target_compile_definitions(${PROJECT_NAME} INTERFACE RETICOLO_HDF5_VERSION_MAJOR=${RETICOLO_HDF5_VERSION_MAJOR})
endif()
if(DEFINED RETICOLO_HDF5_VERSION_MINOR)
    target_compile_definitions(${PROJECT_NAME} INTERFACE RETICOLO_HDF5_VERSION_MINOR=${RETICOLO_HDF5_VERSION_MINOR})
endif()
if(RETICOLO_HDF5_NATIVE_COMPLEX)
    target_compile_definitions(${PROJECT_NAME} INTERFACE RETICOLO_HDF5_NATIVE_COMPLEX=1)
else()
    target_compile_definitions(${PROJECT_NAME} INTERFACE RETICOLO_HDF5_NATIVE_COMPLEX=0)
endif()

target_link_libraries(${PROJECT_NAME} INTERFACE yaml-cpp::yaml-cpp)
target_link_libraries(${PROJECT_NAME} INTERFACE cxxopts::cxxopts)

target_include_directories(
    ${PROJECT_NAME}
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# All users of this library will need at least C++20
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)

# Set the lattice size variable
if (DEFINED SMALL_LATTICE)
    if (SMALL_LATTICE)
        target_compile_definitions(${PROJECT_NAME} INTERFACE SMALL_LATTICE)
    endif()
endif()
