// RUN: bflang -S --emit-llvm %s -o - | FileCheck %s

[]

// CHECK: define i32 @main()
