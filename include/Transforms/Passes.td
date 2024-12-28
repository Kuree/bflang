#ifndef TRANSFORMS_PASSES_TD
#define TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def LiftLoopStartEnd : Pass<"lift-loop-start-end"> {
  let summary = "Lift `bf.loop.start` and `bf.loop.end` to `bf.loop`";

  let dependentDialects = [
    "mlir::bf::BFDialect"
  ];
}

def ValidateBF : Pass<"validate-bf"> {
  let summary = "Validate BF dialect";
}

def PromoteDataPointer : Pass<"promote-data-ptr"> {
  let summary = "Promote data pointer to local stack";
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::arith::ArithDialect"
  ];
}

def OptimizeUnModifiedLoad : Pass<"optimize-unmodified-load"> {
  let summary = "Optimize unmodified load";
  let dependentDialects = [
    "mlir::cf::ControlFlowDialect",
    "mlir::arith::ArithDialect"
  ];
}

#endif  // TRANSFORMS_PASSES_TD