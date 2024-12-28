add_llvm_executable(bfc driver.cc)

target_link_libraries(bfc PRIVATE
        MLIRIR
        MLIRBFTransforms
        MLIRBFConversions
        MLIRTargetLLVMIRExport
        MLIRArithToLLVM
        MLIRFuncToLLVM
        MLIRControlFlowToLLVM
        MLIRBuiltinToLLVMIRTranslation
        MLIRLLVMToLLVMIRTranslation

        lldCommon
        lldCOFF
        lldELF
        lldMachO
        lldMinGW
        lldWasm
)
