// RUN: bf-opt --bf-to-standard --split-input-file %s | FileCheck %s

func.func @ptr_increment() {
  bf.ptr.increment
  return
}

// CHECK-LABEL: @ptr_increment
// CHECK: %[[PTR:.*]] = llvm.mlir.addressof @__data_ptr : !llvm.ptr
// CHECK: %[[PTR_VAL:.*]] = llvm.load %[[PTR]] : !llvm.ptr -> i32
// CHECK: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: %[[ADD:.*]] = arith.addi %[[PTR_VAL]], %[[ONE]] : i32
// CHECK: llvm.store %[[ADD]], %[[PTR]] : i32, !llvm.ptr

// -----

func.func @ptr_decrement() {
  bf.ptr.decrement
  return
}

// CHECK-LABEL: @ptr_decrement
// CHECK: %[[PTR:.*]] = llvm.mlir.addressof @__data_ptr : !llvm.ptr
// CHECK: %[[PTR_VAL:.*]] = llvm.load %[[PTR]] : !llvm.ptr -> i32
// CHECK: %[[NEG_ONE:.*]] = arith.constant -1 : i32
// CHECK: %[[ADD:.*]] = arith.addi %[[PTR_VAL]], %[[NEG_ONE]] : i32
// CHECK: llvm.store %[[ADD]], %[[PTR]] : i32, !llvm.ptr

// -----

func.func @data_increment() {
  bf.data.increment
  return
}

// CHECK-LABEL: @data_increment
// CHECK: %[[DATA_PTR:.*]] = llvm.mlir.addressof @__data_ptr : !llvm.ptr
// CHECK: %[[DATA_PTR_VAL:.*]] = llvm.load %[[DATA_PTR]] : !llvm.ptr -> i32
// CHECK: %[[DATA:.*]] = llvm.mlir.addressof @__data : !llvm.ptr
// CHECK: %[[GEP:.*]] = llvm.getelementptr %[[DATA]][%[[DATA_PTR_VAL]]] : (!llvm.ptr, i32) -> !llvm.ptr, i8
// CHECK: %[[VAL:.*]] = llvm.load %[[GEP]] : !llvm.ptr -> i8
// CHECK: %[[ONE:.*]] = arith.constant 1 : i8
// CHECK: %[[ADD:.*]] = arith.addi %[[VAL]], %[[ONE]] : i8
// CHECK: llvm.store %[[ADD]], %[[GEP]] : i8, !llvm.ptr

// -----

func.func @data_decrement() {
  bf.data.decrement
  return
}

// CHECK-LABEL: @data_decrement
// CHECK: %[[DATA_PTR:.*]] = llvm.mlir.addressof @__data_ptr : !llvm.ptr
// CHECK: %[[DATA_PTR_VAL:.*]] = llvm.load %[[DATA_PTR]] : !llvm.ptr -> i32
// CHECK: %[[DATA:.*]] = llvm.mlir.addressof @__data : !llvm.ptr
// CHECK: %[[GEP:.*]] = llvm.getelementptr %[[DATA]][%[[DATA_PTR_VAL]]] : (!llvm.ptr, i32) -> !llvm.ptr, i8
// CHECK: %[[VAL:.*]] = llvm.load %[[GEP]] : !llvm.ptr -> i8
// CHECK: %[[NEG_ONE:.*]] = arith.constant -1 : i8
// CHECK: %[[ADD:.*]] = arith.addi %[[VAL]], %[[NEG_ONE]] : i8
// CHECK: llvm.store %[[ADD]], %[[GEP]] : i8, !llvm.ptr


// -----

func.func @output() {
  bf.output
  return
}

// CHECK-LABEL: @output
// CHECK: %[[DATA_PTR:.*]] = llvm.mlir.addressof @__data_ptr : !llvm.ptr
// CHECK: %[[DATA_PTR_VAL:.*]] = llvm.load %[[DATA_PTR]] : !llvm.ptr -> i32
// CHECK: %[[DATA:.*]] = llvm.mlir.addressof @__data : !llvm.ptr
// CHECK: %[[GEP:.*]] = llvm.getelementptr %[[DATA]][%[[DATA_PTR_VAL]]] : (!llvm.ptr, i32) -> !llvm.ptr, i8
// CHECK: %[[VAL:.*]] = llvm.load %[[GEP]] : !llvm.ptr -> i8
// CHECK: call @putchar(%[[VAL]]) : (i8) -> ()

// -----

func.func @input() {
  bf.input
  return
}

// CHECK-LABEL: @input
// CHECK: %[[DATA_PTR:.*]] = llvm.mlir.addressof @__data_ptr : !llvm.ptr
// CHECK: %[[DATA_PTR_VAL:.*]] = llvm.load %[[DATA_PTR]] : !llvm.ptr -> i32
// CHECK: %[[DATA:.*]] = llvm.mlir.addressof @__data : !llvm.ptr
// CHECK: %[[GEP:.*]] = llvm.getelementptr %[[DATA]][%[[DATA_PTR_VAL]]] : (!llvm.ptr, i32) -> !llvm.ptr, i8
// CHECK: %[[VAL:.*]] = call @getchar() : () -> i8
// CHECK: llvm.store %[[VAL]], %[[GEP]] : i8, !llvm.ptr
