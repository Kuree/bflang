add_llvm_executable(bflang driver.cc)

target_link_libraries(bflang PRIVATE
        MLIRIR
        MLIRBFTransforms
        MLIRBFConversions
        MLIRTargetLLVMIRExport
        MLIRArithToLLVM
        MLIRFuncToLLVM
        MLIRControlFlowToLLVM
        MLIRBuiltinToLLVMIRTranslation
        MLIRLLVMToLLVMIRTranslation
        MLIRExecutionEngineUtils

        clangDriver
        clangFrontend
)

install(TARGETS bflang RUNTIME DESTINATION bin)
