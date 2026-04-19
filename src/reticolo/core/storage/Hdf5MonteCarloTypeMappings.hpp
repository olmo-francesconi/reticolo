#pragma once

#include <H5Ipublic.h>
#include <H5Tpublic.h>

#include "reticolo/core/storage/Hdf5TypeRegistry.hpp"
#include "reticolo/modules/montecarlo/MonteCarloData.hpp"

namespace reticolo {

template <>
inline auto make_H5_Type<MMonteCarlo::data<RealF>>() -> hid_t {
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(MMonteCarlo::data<RealF>));
    H5Tinsert(DataTypeHid, "acceptance", HOFFSET(MMonteCarlo::data<RealF>, _Acceptance), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "S", HOFFSET(MMonteCarlo::data<RealF>, _S), H5T_NATIVE_DOUBLE);
    return DataTypeHid;
}

template <>
inline auto make_H5_Type<MMonteCarlo::data<RealD>>() -> hid_t {
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(MMonteCarlo::data<RealD>));
    H5Tinsert(DataTypeHid, "acceptance", HOFFSET(MMonteCarlo::data<RealD>, _Acceptance), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "S", HOFFSET(MMonteCarlo::data<RealD>, _S), H5T_NATIVE_DOUBLE);
    return DataTypeHid;
}

template <>
inline auto make_H5_Type<MMonteCarlo::data<ComplexF>>() -> hid_t {
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(MMonteCarlo::data<ComplexF>));
    H5Tinsert(DataTypeHid, "acceptance", HOFFSET(MMonteCarlo::data<ComplexF>, _Acceptance), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "S_re", HOFFSET(MMonteCarlo::data<ComplexF>, _SRe), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "S_im", HOFFSET(MMonteCarlo::data<ComplexF>, _SIm), H5T_NATIVE_DOUBLE);
    return DataTypeHid;
}

template <>
inline auto make_H5_Type<MMonteCarlo::data<ComplexD>>() -> hid_t {
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(MMonteCarlo::data<ComplexD>));
    H5Tinsert(DataTypeHid, "acceptance", HOFFSET(MMonteCarlo::data<ComplexD>, _Acceptance), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "S_re", HOFFSET(MMonteCarlo::data<ComplexD>, _SRe), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "S_im", HOFFSET(MMonteCarlo::data<ComplexD>, _SIm), H5T_NATIVE_DOUBLE);
    return DataTypeHid;
}

}  // namespace reticolo
