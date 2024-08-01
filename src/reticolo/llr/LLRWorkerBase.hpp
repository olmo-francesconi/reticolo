/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: llr/LLEWorkerBase.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

// #include <memory>
// #include <stdexcept>

// namespace reticolo {

// enum class LLRWorkerType {
//     Metropolis,
//     HMC,
//     HMCmet,
// };

// class LLRWorkerFactory {
//   public:
//     static auto makeLLRWorker(LLRWorkerType Type) -> std::unique_ptr<LLRWorkerBase> {
//         switch (Type) {
//             case LLRWorkerType::Metropolis:
//                 return std::make_unique<LLR>();
//             default:
//                 throw std::runtime_error("no matching LLRWorker type");
//         };
//     }
// }
