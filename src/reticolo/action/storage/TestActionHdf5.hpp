#pragma once

#include <H5Ipublic.h>
#include <H5Tpublic.h>

#include "reticolo/action/TestAction.hpp"
#include "reticolo/core/storage/Hdf5TypeRegistry.hpp"

namespace reticolo {

template <>
inline auto make_H5_Type<action::TestAction<RealF>::Observables>() -> hid_t {
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(action::TestAction<RealF>::Observables));
    H5Tinsert(DataTypeHid, "a", HOFFSET(action::TestAction<RealF>::Observables, a), H5T_NATIVE_FLOAT);
    H5Tinsert(DataTypeHid, "b", HOFFSET(action::TestAction<RealF>::Observables, b), H5T_NATIVE_FLOAT);
    return DataTypeHid;
}

template <>
inline auto make_H5_Type<action::TestAction<RealD>::Observables>() -> hid_t {
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(action::TestAction<RealD>::Observables));
    H5Tinsert(DataTypeHid, "a", HOFFSET(action::TestAction<RealD>::Observables, a), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "b", HOFFSET(action::TestAction<RealD>::Observables, b), H5T_NATIVE_DOUBLE);
    return DataTypeHid;
}

}  // namespace reticolo
