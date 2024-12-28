// RUN: bf-opt --optimize-unmodified-load %s -o - | FileCheck %s

llvm.mlir.global private @__data(dense<0> : tensor<30000xi8>) {addr_space = 0 : i32} : !llvm.array<30000 x i8>

func.func @direct_addr() -> i8 {
  %addr = llvm.mlir.addressof @__data : !llvm.ptr
  %0 = llvm.load %addr: !llvm.ptr -> i8
  return %0: i8
}

// CHECK-LABEL: @direct_addr
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i8
// CHECK: return %[[ZERO]]

func.func @gep_addr() -> i8 {
  %addr = llvm.mlir.addressof @__data : !llvm.ptr
  %gep = llvm.getelementptr %addr[1] : (!llvm.ptr) -> !llvm.ptr, i8
  %0 = llvm.load %gep: !llvm.ptr -> i8
  return %0: i8
}

// CHECK-LABEL: @gep_addr
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i8
// CHECK: return %[[ZERO]]

func.func @gep_addr_same_store(%arg0: i8) -> i8 {
  %addr = llvm.mlir.addressof @__data : !llvm.ptr
  %gep = llvm.getelementptr %addr[1] : (!llvm.ptr) -> !llvm.ptr, i8
  llvm.store %arg0, %gep: i8, !llvm.ptr
  %0 = llvm.load %gep: !llvm.ptr -> i8
  return %0: i8
}

// CHECK-LABEL: @gep_addr_same_store
// CHECK-NOT: arith.constant 0

func.func @gep_addr_val_store(%arg0: i8, %arg1: i32) -> i8 {
  %addr = llvm.mlir.addressof @__data : !llvm.ptr
  %gep = llvm.getelementptr %addr[%arg1] : (!llvm.ptr, i32) -> !llvm.ptr, i8
  llvm.store %arg0, %gep: i8, !llvm.ptr
  %0 = llvm.load %gep: !llvm.ptr -> i8
  return %0: i8
}

// CHECK-LABEL: @gep_addr_val_store
// CHECK-NOT: arith.constant 0
