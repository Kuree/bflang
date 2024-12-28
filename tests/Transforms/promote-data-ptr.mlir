// RUN: bf-opt --promote-data-ptr %s | FileCheck %s

llvm.mlir.global private @__data_ptr(0 : i32) {addr_space = 0 : i32} : i32

func.func @foo(%arg0: i32) {
  %0 = llvm.mlir.addressof @__data_ptr : !llvm.ptr
  %1 = llvm.load %0 : !llvm.ptr -> i32
  llvm.store %arg0, %0: i32, !llvm.ptr
  return
}

// CHECK-NOT: @__data_ptr
// CHECK-LABEL: @foo
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK: %[[ALLOCA:.*]] = llvm.alloca %[[C1]] x i32 : (i32) -> !llvm.ptr
// CHECK: llvm.store %[[C0]], %[[ALLOCA]] : i32, !llvm.ptr
// CHECK: llvm.store %arg0, %[[ALLOCA]] : i32, !llvm.ptr
