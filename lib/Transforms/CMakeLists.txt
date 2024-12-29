add_mlir_library(MLIRBFTransforms
        LiftLoopStartEnd.cc
        ValidateBF.cc
        PromoteDataPointer.cc
        OptimizeUnmodifiedLoad.cc
        AttachDebugInfo.cc

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
