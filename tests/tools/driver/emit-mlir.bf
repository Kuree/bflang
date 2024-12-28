// RUN: bflang -S --emit-mlir %s -o - | FileCheck %s

[]

// CHECK: func.func @main()
// CHECK: cf.cond_br
