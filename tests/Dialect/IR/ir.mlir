// RUN: bf-opt %s | FileCheck %s

bf.ptr.increment
// CHECK: bf.ptr.increment

bf.ptr.decrement
// CHECK: bf.ptr.decrement

bf.data.increment
// CHECK: bf.data.increment

bf.data.decrement
// CHECK: bf.data.decrement

bf.output
// CHECK: bf.output

bf.input
// CHECK: bf.input

bf.loop.start
// CHECK: bf.loop.start

bf.loop.end
// CHECK: bf.loop.end

bf.loop {

}
// CHECK: bf.loop {
// CHECK-NEXT: }
