cmake \
    -S . \
    -B ./build \
    -DCMAKE_BUILD_TYPE="Release" \
    -DCMAKE_INSTALL_PREFIX="/opt/reticolo" \
    -DCMAKE_CXX_COMPILER="/opt/homebrew/opt/llvm/bin/clang++" \
    -DHDF5_ROOT="/opt/hdf5/llvm/c-threadsafe"
