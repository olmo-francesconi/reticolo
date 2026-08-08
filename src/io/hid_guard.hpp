#pragma once

#include <hdf5.h>

// RAII for HDF5 object ids — private to src/io/, the only place `<hdf5.h>` is
// allowed (see the reticolo::io target invariant).
//
// `hid_t` is a resource handle, not a pointer, so no standard smart pointer
// covers it, and each id class has its own closer (H5Oclose / H5Sclose /
// H5Tclose / H5Aclose / H5Gclose / H5Dclose). Both TUs report errors by
// THROWING from `hid_check` / `herr_check`, ~44 call sites between them, and a
// throw there used to abandon every id already open in that scope.
//
// Leaking an id is not "a few bytes": HDF5 keeps the whole file object alive
// until every id derived from it is released, so one leaked dataspace pins the
// file — which is why the leak showed up as files that never really closed
// rather than as growing memory.
//
// The closer is a template non-type parameter, so a guard is exactly one hid_t
// wide and the close call devirtualises to a direct call.

namespace reticolo::io::detail {

template <herr_t (*Close)(hid_t)>
class HidGuard {
public:
    HidGuard() = default;
    explicit HidGuard(hid_t id) noexcept : id_{id} {}

    HidGuard(HidGuard const&)            = delete;
    HidGuard& operator=(HidGuard const&) = delete;

    HidGuard(HidGuard&& o) noexcept : id_{o.id_} { o.id_ = H5I_INVALID_HID; }
    HidGuard& operator=(HidGuard&& o) noexcept {
        if (this != &o) {
            reset();
            id_   = o.id_;
            o.id_ = H5I_INVALID_HID;
        }
        return *this;
    }

    ~HidGuard() { reset(); }

    [[nodiscard]] hid_t get() const noexcept { return id_; }
    [[nodiscard]] bool valid() const noexcept { return id_ >= 0; }

    // Close early, where the original code closed before a deliberate throw and
    // the ordering is worth keeping explicit.
    void reset() noexcept {
        if (id_ >= 0) {
            Close(id_);
        }
        id_ = H5I_INVALID_HID;
    }

private:
    hid_t id_ = H5I_INVALID_HID;
};

using ObjId   = HidGuard<H5Oclose>;
using GroupId = HidGuard<H5Gclose>;
using DsetId  = HidGuard<H5Dclose>;
using SpaceId = HidGuard<H5Sclose>;
using TypeId  = HidGuard<H5Tclose>;
using AttrId  = HidGuard<H5Aclose>;
using PropId  = HidGuard<H5Pclose>;

}  // namespace reticolo::io::detail
