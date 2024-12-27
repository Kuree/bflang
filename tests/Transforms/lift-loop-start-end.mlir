// RUN: bf-opt --lift-loop-start-end --allow-unregistered-dialect --split-input-file %s | FileCheck %s

func.func @single() {
  bf.loop.start
  "op.op"(): () -> ()
  bf.loop.end
  return
}

// CHECK-LABEL: @single
// CHECK-NEXT: bf.loop {
// CHECK-NEXT: "op.op"
// CHECK-NEXT: }
// CHECK-NEXT: return

// -----

func.func @nested() {
  bf.loop.start
  "op.op1"(): () -> ()
  bf.loop.start
  "op.op2"(): () -> ()
  bf.loop.end
  "op.op3"(): () -> ()
  bf.loop.end
  return
}

// CHECK-LABEL: @nested
// CHECK-NEXT: bf.loop
// CHECK-NEXT: "op.op1"
// CHECK-NEXT: bf.loop
// CHECK-NEXT: "op.op2"
// CHECK-NEXT: }
// CHECK: "op.op3"
// CHECK-NEXT: }
// CHECK-NEXT: return
