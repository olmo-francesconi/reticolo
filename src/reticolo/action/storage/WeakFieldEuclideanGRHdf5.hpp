#pragma once

#include <H5Ipublic.h>
#include <H5Tpublic.h>

#include "reticolo/action/WeakFieldEuclideanGR.hpp"
#include "reticolo/core/storage/Hdf5TypeRegistry.hpp"

namespace reticolo {

template <>
inline auto make_H5_Type<action::WeakFieldEuclideanGR<RealF>::Observables>() -> hid_t {
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(action::WeakFieldEuclideanGR<RealF>::Observables));
    H5Tinsert(DataTypeHid, "R", HOFFSET(action::WeakFieldEuclideanGR<RealF>::Observables, R), H5T_NATIVE_FLOAT);
    return DataTypeHid;
}

template <>
inline auto make_H5_Type<action::WeakFieldEuclideanGR<RealD>::Observables>() -> hid_t {
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(action::WeakFieldEuclideanGR<RealD>::Observables));
    H5Tinsert(DataTypeHid, "R", HOFFSET(action::WeakFieldEuclideanGR<RealD>::Observables, R), H5T_NATIVE_DOUBLE);
    return DataTypeHid;
}

}  // namespace reticolo
