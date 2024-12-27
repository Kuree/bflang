#ifndef CONVERSION_PASSES_TD
#define CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def BFToStandard : Pass<"bf-to-standard"> {
  let summary = "Lower `bf` to standard MLIR dialects.";

  let description = [{
    This pass lowers the `bf` dialect to standard MLIR, a mixture of llvm,
    cf, and arith.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::func::FuncDialect",
    "mlir::cf::ControlFlowDialect",
  ];
}

#endif  // CONVERSION_PASSES_TD