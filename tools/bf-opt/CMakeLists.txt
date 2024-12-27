add_llvm_executable(bf-opt bf-opt.cc)
llvm_update_compile_flags(bf-opt)

target_link_libraries(
        bf-opt
        PRIVATE MLIROptLib
        MLIRBF
        MLIRBFConversions
        MLIRBFTransforms)
