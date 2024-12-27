set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(BFOps.h.inc -gen-op-decls)
mlir_tablegen(BFOps.cpp.inc -gen-op-defs)
mlir_tablegen(BFDialect.h.inc -gen-dialect-decls -dialect=bf)
mlir_tablegen(BFDialect.cpp.inc -gen-dialect-defs -dialect=bf)

add_public_tablegen_target(MLIRBfOpsIncGen)
