add_mlir_dialect_library(MLIRBF
    BFDialect.cc
    DEPENDS MLIRBfOpsIncGen
    ADDITIONAL_HEADERS ${PROJECT_SOURCE_DIR}/include/IR
)
