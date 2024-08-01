/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: lattice/FieldBase.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "reticolo/types/core.hpp"
#include "reticolo/types/hfield.hpp"

namespace reticolo {

class FieldBase {
  public:
    /* Define default destructor of base class */
    virtual ~FieldBase() = default;
    virtual void resizeData(size_t newSize) = 0;
    virtual auto dataSize() -> size_t = 0;
};

template <typename T>
class Field : public FieldBase {
  public:
    std::vector<T> data;

    void resizeData(size_t newSize) override { data.resize(newSize); }
    auto dataSize() -> size_t override { return data.size(); }
};

enum class FieldType {
    RealFloat,
    RealDouble,
    ComplexFloat,
    ComplexDouble,
    MetricFloat,
    MetricDouble,
};

class FieldFactory {
  public:
    static auto makefield(FieldType Type) -> std::unique_ptr<FieldBase> {
        switch (Type) {
            case FieldType::RealFloat:
                return std::make_unique<Field<RealF>>();
            case FieldType::RealDouble:
                return std::make_unique<Field<RealD>>();
            case FieldType::ComplexFloat:
                return std::make_unique<Field<ComplexF>>();
            case FieldType::ComplexDouble:
                return std::make_unique<Field<ComplexD>>();
            case FieldType::MetricFloat:
                return std::make_unique<Field<HField<RealF>>>();
            case FieldType::MetricDouble:
                return std::make_unique<Field<HField<RealD>>>();
            default:
                throw std::runtime_error("no matching Field type");
        };
    }
};

}  // namespace reticolo
