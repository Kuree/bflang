// RUN: bflang -S --emit-mlir %s -o - | FileCheck %s

[]

// CHECK: func @main()
// CHECK: cond_br
