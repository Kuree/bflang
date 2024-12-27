add_llvm_executable(bfc driver.cc)

target_link_libraries(bfc PRIVATE
        lld
        MLIRIR
        MLIRBFTransforms
        MLIRBFConversions
        MLIRTargetLLVMIRExport
)
