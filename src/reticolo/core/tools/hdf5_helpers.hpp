#pragma once

#include <H5Ipublic.h>

namespace reticolo {

template <typename T>
inline auto make_H5_Type() -> hid_t;

}