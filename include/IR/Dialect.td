#ifndef IR_DIALECT_TD
#define IR_DIALECT_TD

include "mlir/IR/OpBase.td"

def BF_Dialect : Dialect {
  let name = "bf";
  let summary = "A dialect for brainfuck";
  let description = [{
    The bf dialect represents the operators specified by the brainfuck
    language.
  }];

  let cppNamespace = "::mlir::bf";
}

#endif  // IR_DIALECT_TD