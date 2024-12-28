// RUN: bflang -S --emit-bf %s -o - | FileCheck %s

[.]

// CHECK: func.func @main()
// CHECK: bf.loop
