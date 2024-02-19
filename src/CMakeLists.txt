# Make an automatic library - will be static or dynamic based on user setting
add_library(reticolo INTERFACE)

# target HDF5
target_link_libraries(reticolo INTERFACE HDF5::HDF5)

# include 
target_include_directories(reticolo INTERFACE ".")

# All users of this library will need at least C++20
target_compile_features(reticolo INTERFACE cxx_std_20)