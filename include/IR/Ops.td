#ifndef IR_OPS_TD
#define IR_OPS_TD

include "Dialect.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


class BF_Op<string mnemonic, list<Trait> traits = []> : Op<BF_Dialect, mnemonic, traits> {
    let assemblyFormat = "attr-dict";
}

def BF_IncrementPtr : BF_Op<"ptr.increment"> {
  let summary = "Increment data pointer - `>`";
}

def BF_DecrementPtr : BF_Op<"ptr.decrement"> {
  let summary = "Decrement data pointer - '<`";
}

def BF_IncrementData : BF_Op<"data.increment"> {
  let summary = "Increment byte at data pointer by one - `+`";
}

def BF_DecrementData : BF_Op<"data.decrement"> {
  let summary = "Decrement byte at data pointer by one - `-`";
}

def BF_Output : BF_Op<"output"> {
  let summary = "Output the byte at the data pointer - `.`";
}

def BF_Input : BF_Op<"input"> {
  let summary = "Accept one byte of input, storing its value in the byte at the data pointer - `,`";
}

def BF_LoopStart : BF_Op<"loop.start"> {
  let summary = "If the byte at the data pointer is zero, then instead of moving the instruction pointer forward to the next command, jump it forward to the command after the matching `] command";
}

def BF_LoopEnd : BF_Op<"loop.end"> {
  let summary = "If the byte at the data pointer is nonzero, then instead of moving the instruction pointer forward to the next command, jump it back to the command after the matching `[` command.";
}

def BF_Loop : BF_Op<"loop", [RecursiveMemoryEffects, SingleBlockImplicitTerminator<"LoopYield">]> {
  let summary = "High-level loop region to represent a `[]` code block";

  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat = "$region attr-dict";
}

def BF_LoopYield : BF_Op<"loop.yield", [Terminator, HasParent<"Loop">]> {
  let summary = "Terminator for `bf.loop`";
}

#endif  // IR_OPS_TD
