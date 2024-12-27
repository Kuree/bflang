#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Conversion/Passes.hh"
#include "IR/BFDialect.hh"

int main(int argc, char *argv[]) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::cf::ControlFlowDialect, mlir::arith::ArithDialect,
                    mlir::scf::SCFDialect, mlir::func::FuncDialect,
                    mlir::LLVM::LLVMDialect, mlir::bf::BFDialect>();
    mlir::MlirOptMainConfig config;
    mlir::bf::conversions::registerPasses();
    return mlir::failed(
        mlir::MlirOptMain(argc, argv, "Brainfuck Optimization Tool", registry));
}