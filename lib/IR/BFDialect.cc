#include "IR/BFDialect.hh"
#include "IR/BFDialect.cpp.inc"
#include "IR/BFOps.hh"

namespace mlir::bf {
void BFDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "IR/BFOps.cpp.inc"
        >();
}
} // namespace mlir::bf

#define GET_OP_CLASSES
#include "IR/BFOps.cpp.inc"
