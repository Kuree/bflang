// RUN: bf-opt --attach-debug-info --mlir-print-debuginfo %s | FileCheck %s

llvm.mlir.global private @__data_ptr(0 : i32) {addr_space = 0 : i32} : i32
llvm.mlir.global private @__data(dense<0> : tensor<30000xi8>) {addr_space = 0 : i32} : !llvm.array<30000 x i8>

func.func @main() {
  %c0 = arith.constant 0: i32
  %0 = llvm.mlir.addressof @__data_ptr : !llvm.ptr
  llvm.store %c0, %0: i32, !llvm.ptr
  return
}

// CHECK-DAG: #[[BASIC_TY_0:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "uint32_t", sizeInBits = 32, encoding = DW_ATE_unsigned>
// CHECK-DAG: #[[BASIC_TY_1:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "uint8_t", sizeInBits = 8, encoding = DW_ATE_unsigned>
// CHECK-DAG: #[[DI_FILE:.*]] = #llvm.di_file<"{{.*}}" in "{{.*}}">
// CHECK-DAG: #[[DI_UNIT:.*]] = #llvm.di_compile_unit<id = distinct[{{.*}}]<>, sourceLanguage = DW_LANG_C11, file = #[[DI_FILE]], producer = "bflang", isOptimized = false, emissionKind = Full>
// CHECK-DAG: #[[DI_COMPOSITE_TYPE:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type, {{.*}}baseType = #[[BASIC_TY_1]], sizeInBits = 240000>
// CHECK-DAG: #[[VAR0:.*]] = #llvm.di_global_variable<scope = #[[DI_UNIT]], name = "__data_ptr", file = #[[DI_FILE]], line = 0, type = #[[BASIC_TY_0]], isLocalToUnit = true, isDefined = true>
// CHECK-DAG: #[[VAR1:.*]] = #llvm.di_global_variable<scope = #[[DI_UNIT]], name = "__data", file = #[[DI_FILE]], line = 0, type = #[[DI_COMPOSITE_TYPE]], isLocalToUnit = true, isDefined = true>
// CHECK-DAG: #[[VAR_EXPR0:.*]] = #llvm.di_global_variable_expression<var = #[[VAR0]], expr = <>>
// CHECK-DAG: #[[VAR_EXPR1:.*]] = #llvm.di_global_variable_expression<var = #[[VAR1]], expr = <>>

// CHECK-LABEL: @__data_ptr
// CHECK-SAME: dbg_expr = #[[VAR_EXPR0]]

// CHECK-LABEL: @__data
// CHECK-SAME: dbg_expr = #[[VAR_EXPR1]]

// CHECK-LABEL: @main
// CHECK: llvm.mlir.addressof @__data_ptr : !llvm.ptr loc(#[[LOC:.*]])

// CHECK: #[[INT_TYPE:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
// CHECK: #[[SUBROUTINE:.*]] = #llvm.di_subroutine_type<types = #[[INT_TYPE]]>
// CHECK: #[[SUBPROGRAM:.*]] = #llvm.di_subprogram<id = distinct[{{.*}}]<>, compileUnit = #[[DI_UNIT]], scope = #di_file, name = "main"
// CHECK-SAME: file = #[[DI_FILE]], subprogramFlags = Definition, type = #[[SUBROUTINE]]>

// CHECK: #[[LOC]] = loc(fused<#[[SUBPROGRAM]]>[#{{.*}}])
