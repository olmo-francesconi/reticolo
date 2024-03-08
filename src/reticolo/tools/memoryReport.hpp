/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tools/io_utils.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

******************************************************************************/

#pragma once

#include <cstddef>
#include <format>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "reticolo/tools/io_utils.hpp"
#include "reticolo/types/core.hpp"

namespace reticolo {

class MemoryReportLayer {
  private:
    MemoryReportLayer*              _Parent;
    std::vector<MemoryReportLayer*> _Daughters;
    std::string                     _Name;
    uint                            _Depth{0};
    size_t                          _Tot{0};
    std::map<std::string, size_t>   _LayerReport;

  public:
    explicit MemoryReportLayer(std::string LayerName, MemoryReportLayer* Parent = nullptr)
        : _Parent(Parent), _Name(std::move(LayerName)) {}
    auto getName() -> std::string { return _Name; }
    auto getParent() -> MemoryReportLayer* { return _Parent; }
    void increaseDepth() {
        _Depth++;
        for (auto* const Daughter : _Daughters) {
            Daughter->increaseDepth();
        }
    }

    [[nodiscard]] auto getDepth() const -> int { return _Depth; }
    [[nodiscard]] auto getTot() const -> size_t { return _Tot; }

    void addDaughter(MemoryReportLayer* NewDaughter) {
        _Daughters.push_back(NewDaughter);
        _Tot += NewDaughter->getTot();
        NewDaughter->increaseDepth();
    }
    void addEntry(const std::string& key, size_t value) {
        _LayerReport.insert({key, value});
        _Tot += value;
    }

    auto getFullReport() -> std::string {
        std::string Res;
        Res += std::format("{:.<48}[{}]\n", std::string(2 * _Depth, ' ') + _Name, IO::pretty_bytes(_Tot));
        for (const auto& [key, value] : _LayerReport) {
            Res += std::format("{:<48}[{}]\n", std::string(2 * (_Depth + 1), ' ') + key, IO::pretty_bytes(value));
        }
        for (auto* const Daughter : _Daughters) {
            Res += Daughter->getFullReport();
        }
        return Res;
    }

    auto getShortReport() -> std::string {
        std::string Res;
        Res += std::format("{:.<48}[{}]\n", std::string(2 * _Depth, ' ') + _Name, IO::pretty_bytes(_Tot));
        size_t OtherTot = 0;
        for (const auto& [_, value] : _LayerReport) {
            OtherTot += value;
        }
        Res += std::format("{:<48}[{}]\n", std::string(2 * (_Depth + 1), ' ') + "Others", IO::pretty_bytes(OtherTot));
        for (auto* const Daughter : _Daughters) {
            Res += Daughter->getShortReport();
        }
        return Res;
    }
};
}  // namespace reticolo