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

#endif  // TRANSFORMS_PASSES_TD