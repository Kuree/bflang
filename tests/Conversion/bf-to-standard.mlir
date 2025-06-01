// RUN: bf-opt --bf-to-standard --split-input-file --allow-unregistered-dialect %s | FileCheck %s
// RUN: bf-opt --bf-to-standard=data-array-size=300 --split-input-file --allow-unregistered-dialect %s | FileCheck %s --check-prefix=ARRAY

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

// CHECK: llvm.mlir.global private @__data(dense<0> : tensor<30000xi8>)
// CHECK-SAME: !llvm.array<30000 x i8>

// ARRAY: llvm.mlir.global private @__data(dense<0> : tensor<300xi8>)
// ARRAY-SAME: !llvm.array<300 x i8>

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


// -----

func.func @loop() {
  "op.op"() : () -> ()
  bf.loop {
    "op.op"() : () -> ()
  }
  return
}

// CHECK-LABEL: @loop
// CHECK:   "op.op"() : () -> ()
// CHECK:   %[[ZERO:.*]] = arith.constant 0 : i8
// CHECK:   %[[DATA_PTR_BEFORE:.*]] = llvm.mlir.addressof @__data_ptr : !llvm.ptr
// CHECK:   %[[DATA_PTR_VAL_BEFORE:.*]] = llvm.load %[[DATA_PTR_BEFORE]] : !llvm.ptr -> i32
// CHECK:   %[[DATA_BEFORE:.*]] = llvm.mlir.addressof @__data : !llvm.ptr
// CHECK:   %[[GEP_BEFORE:.*]] = llvm.getelementptr %[[DATA_BEFORE]][%[[DATA_PTR_VAL_BEFORE]]] : (!llvm.ptr, i32) -> !llvm.ptr, i8
// CHECK:   %[[DATA_VAL_BEFORE:.*]] = llvm.load %[[GEP_BEFORE]] : !llvm.ptr -> i8
// CHECK:   %[[CMP_EQ_ZERO:.*]] = arith.cmpi eq, %[[DATA_VAL_BEFORE]], %[[ZERO]] : i8
// CHECK:   cf.cond_br %[[CMP_EQ_ZERO]], ^[[AFTER_LOOP_BB:.*]], ^[[LOOP_BB:.*]]
// CHECK: ^[[LOOP_BB]]:
// CHECK:   "op.op"() : () -> ()
// CHECK:   %[[DATA_PTR_AFTER:.*]] = llvm.mlir.addressof @__data_ptr : !llvm.ptr
// CHECK:   %[[DATA_PTR_VAL_AFTER:.*]] = llvm.load %[[DATA_PTR_AFTER]] : !llvm.ptr -> i32
// CHECK:   %[[DATA_AFTER:.*]] = llvm.mlir.addressof @__data : !llvm.ptr
// CHECK:   %[[GEP_AFTER:.*]] = llvm.getelementptr %[[DATA_AFTER]][%[[DATA_PTR_VAL_AFTER]]] : (!llvm.ptr, i32) -> !llvm.ptr, i8
// CHECK:   %[[DATA_VAL_AFTER:.*]] = llvm.load %[[GEP_AFTER]] : !llvm.ptr -> i8
// CHECK:   %[[CMP_NE_ZERO:.*]] = arith.cmpi ne, %[[DATA_VAL_AFTER]], %[[ZERO]] : i8
// CHECK:   cf.cond_br %[[CMP_NE_ZERO]], ^[[LOOP_BB]], ^[[AFTER_LOOP_BB]]
// CHECK: ^[[AFTER_LOOP_BB]]:
// CHECK-NEXT: return
