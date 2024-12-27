#ifndef BFC_PASSES_HH
#define BFC_PASSES_HH

#include "mlir/Pass/Pass.h"

namespace mlir::bf {
#define GEN_PASS_DECL
#include "Conversion/Passes.h.inc"

namespace conversions {
#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"
} // namespace conversions

} // namespace mlir::bf

#endif // BFC_PASSES_HH
