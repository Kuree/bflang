// RUN: bflang -S --emit-bf %s -o - | FileCheck %s

[.]

// CHECK: func @main()
// CHECK: loop
