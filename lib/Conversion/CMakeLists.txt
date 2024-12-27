add_mlir_library(MLIRBFConversions
        BfToStandard.cc

        ADDITIONAL_HEADER_DIRS

        DEPENDS
        BFPassesIncGen

        LINK_LIBS PUBLIC
        MLIRBF
        MLIRIR
        MLIRPass
        MLIRSCFDialect
        MLIRTransforms
        MLIRArithDialect
        MLIRFuncDialect
        MLIRArithDialect
        MLIRControlFlowDialect
        MLIRLLVMDialect
        MLIRSCFDialect
)
