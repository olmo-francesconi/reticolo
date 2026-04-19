#pragma once

#include <H5Ipublic.h>
#include <H5Tpublic.h>

#include "reticolo/action/RelativisticBoseGas.hpp"
#include "reticolo/core/storage/Hdf5TypeRegistry.hpp"

namespace reticolo {

template <>
inline auto make_H5_Type<action::RelativisticBoseGas<RealF>::Observables>() -> hid_t {
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(action::RelativisticBoseGas<RealF>::Observables));
    H5Tinsert(DataTypeHid, "phi2", HOFFSET(action::RelativisticBoseGas<RealF>::Observables, phi2), H5T_NATIVE_FLOAT);
    H5Tinsert(DataTypeHid, "density", HOFFSET(action::RelativisticBoseGas<RealF>::Observables, density),
              H5T_NATIVE_FLOAT);
    return DataTypeHid;
}

template <>
inline auto make_H5_Type<action::RelativisticBoseGas<RealD>::Observables>() -> hid_t {
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(action::RelativisticBoseGas<RealD>::Observables));
    H5Tinsert(DataTypeHid, "phi2", HOFFSET(action::RelativisticBoseGas<RealD>::Observables, phi2), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "density", HOFFSET(action::RelativisticBoseGas<RealD>::Observables, density),
              H5T_NATIVE_DOUBLE);
    return DataTypeHid;
}

}  // namespace reticolo
