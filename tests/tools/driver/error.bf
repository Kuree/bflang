// RUN: not bflang %s -o - 2>&1 | FileCheck %s

>[>
// CHECK: error: missing matching ']'
// CHECK: >[>
// CHECK: ^
