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
    MemoryReportLayer*              m_Parent;
    std::vector<MemoryReportLayer*> m_Daughters;
    std::string                     m_Name;
    uint                            m_Depth{0};
    size_t                          m_Tot{0};
    std::map<std::string, size_t>   m_LayerReport;

  public:
    explicit MemoryReportLayer(std::string LayerName, MemoryReportLayer* Parent = nullptr)
        : m_Parent(Parent), m_Name(std::move(LayerName)) {}
    auto getName() -> std::string { return m_Name; }
    auto getParent() -> MemoryReportLayer* { return m_Parent; }
    void increaseDepth() {
        m_Depth++;
        for (auto* const Daughter : m_Daughters) {
            Daughter->increaseDepth();
        }
    }

    [[nodiscard]] auto getDepth() const -> int { return m_Depth; }
    [[nodiscard]] auto getTot() const -> size_t { return m_Tot; }

    void addDaughter(MemoryReportLayer* NewDaughter) {
        m_Daughters.push_back(NewDaughter);
        m_Tot += NewDaughter->getTot();
        NewDaughter->increaseDepth();
    }
    void addEntry(const std::string& key, size_t value) {
        m_LayerReport.insert({key, value});
        m_Tot += value;
    }

    auto getFullReport() -> std::string {
        std::string Res;
        Res += std::format("{:.<48}[{}]\n", std::string(2 * m_Depth, ' ') + m_Name, IO::pretty_bytes(m_Tot));
        for (const auto& [key, value] : m_LayerReport) {
            Res += std::format("{:<48}[{}]\n", std::string(2 * (m_Depth + 1), ' ') + key, IO::pretty_bytes(value));
        }
        for (auto* const Daughter : m_Daughters) {
            Res += Daughter->getFullReport();
        }
        return Res;
    }

    auto getShortReport() -> std::string {
        std::string Res;
        Res += std::format("{:.<48}[{}]\n", std::string(2 * m_Depth, ' ') + m_Name, IO::pretty_bytes(m_Tot));
        size_t OtherTot = 0;
        for (const auto& [_, value] : m_LayerReport) {
            OtherTot += value;
        }
        Res += std::format("{:<48}[{}]\n", std::string(2 * (m_Depth + 1), ' ') + "Others", IO::pretty_bytes(OtherTot));
        for (auto* const Daughter : m_Daughters) {
            Res += Daughter->getShortReport();
        }
        return Res;
    }
};
}  // namespace reticolo
