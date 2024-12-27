#ifndef BFC_PASSES_HH
#define BFC_PASSES_HH

#include "mlir/Pass/Pass.h"

namespace mlir::bf {
#define GEN_PASS_DECL
#include "Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Transforms/Passes.h.inc"

} // namespace mlir::bf

#endif // BFC_PASSES_HH
