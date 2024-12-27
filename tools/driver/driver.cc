#include "lld/Common/Driver.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"

#include "Conversion/Passes.hh"
#include "IR/BFOps.hh"
#include "Transforms/Passes.hh"

namespace {
enum OptLevel { O0, O1 };

llvm::cl::opt<bool> emitAssembly{
    "S", llvm::cl::desc("Produce an assembly file in MLIR.")};
llvm::cl::opt<std::string> outputFileName{
    "o", llvm::cl::desc("Write output to file."), llvm::cl::value_desc("file"),
    llvm::cl::init("-")};

llvm::cl::opt<OptLevel> optimizationLevel(
    llvm::cl::desc("Choose optimization level:"),
    llvm::cl::values(clEnumVal(O0, "No optimization"),
                     clEnumVal(O1, "Enable default optimizations")));

llvm::cl::opt<std::string> inputFileName(llvm::cl::Positional,
                                         llvm::cl::desc("<input file>"));

void parseCode(
    llvm::SourceMgr &sourceMgr, uint32_t bufferId, mlir::OpBuilder &builder,
    const std::function<mlir::FileLineColLoc(uint32_t, uint32_t)> &getLoc) {
    auto ref = sourceMgr.getMemoryBuffer(bufferId)->getBuffer();
    uint32_t line = 1, column = 1;
    for (auto c : ref) {
        switch (c) {
        case '\n': {
            line += 1;
            column = 0;
            break;
        }
        case EOF: {
            return;
        }
        case '>': {
            auto loc = getLoc(line, column);
            builder.create<mlir::bf::IncrementPtr>(loc);
            break;
        }
        case '<': {
            auto loc = getLoc(line, column);
            builder.create<mlir::bf::DecrementPtr>(loc);
            break;
        }
        case '+': {
            auto loc = getLoc(line, column);
            builder.create<mlir::bf::IncrementData>(loc);
            break;
        }
        case '-': {
            auto loc = getLoc(line, column);
            builder.create<mlir::bf::DecrementData>(loc);
            break;
        }
        case '.': {
            auto loc = getLoc(line, column);
            builder.create<mlir::bf::Output>(loc);
            break;
        }
        case ',': {
            auto loc = getLoc(line, column);
            builder.create<mlir::bf::Input>(loc);
            break;
        }
        case '[': {
            auto loc = getLoc(line, column);
            builder.create<mlir::bf::LoopStart>(loc);
            break;
        }
        case ']': {
            auto loc = getLoc(line, column);
            builder.create<mlir::bf::LoopEnd>(loc);
            break;
        }
        default: {
            // noop
        }
        }
        ++column;
    }
}

mlir::OwningOpRef<mlir::ModuleOp> parseCodeModule(llvm::SourceMgr &sourceMgr,
                                                  uint32_t bufferId,
                                                  mlir::MLIRContext &context) {
    std::string filename =
        inputFileName.empty() ? "<stdin>" : inputFileName.getValue();
    mlir::OpBuilder builder(&context);
    auto getLoc = [&](uint32_t line, uint32_t col) {
        return builder.getAttr<mlir::FileLineColLoc>(filename, line, col);
    };
    auto dummyLoc = getLoc(0, 0);
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(dummyLoc);

    // set up a main function
    builder.setInsertionPointToEnd(module->getBody());
    auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
    auto func = builder.create<mlir::func::FuncOp>(dummyLoc, "main", funcType);
    auto *block = func.addEntryBlock();
    builder.setInsertionPointToEnd(block);

    // parse logic
    parseCode(sourceMgr, bufferId, builder, getLoc);

    auto zero = builder.create<mlir::arith::ConstantIntOp>(dummyLoc, 0, 32);
    builder.create<mlir::func::ReturnOp>(dummyLoc,
                                         llvm::SmallVector<mlir::Value>{zero});

    return module;
}

void loadPasses(mlir::PassManager &pm) {
    pm.addPass(mlir::bf::createValidateBF());
    pm.addPass(mlir::bf::createLiftLoopStartEnd());

    pm.addPass(mlir::bf::createBFToStandard());
}

} // namespace

int main(int argc, char *argv[]) {
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();
    llvm::cl::ParseCommandLineOptions(argc, argv);

    mlir::MLIRContext mlirContext;
    mlirContext
        .loadDialect<mlir::cf::ControlFlowDialect, mlir::arith::ArithDialect,
                     mlir::scf::SCFDialect, mlir::func::FuncDialect,
                     mlir::LLVM::LLVMDialect, mlir::bf::BFDialect>();
    // don't show the op to user
    mlirContext.printOpOnDiagnostic(false);
    llvm::SourceMgr sourceMgr;
    mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &mlirContext);

    auto source = llvm::MemoryBuffer::getFileOrSTDIN(inputFileName);
    if (!source) {
        auto error = source.getError();
        llvm::errs() << "error opening file " << inputFileName << ": "
                     << error.message() << "\n";
        return EXIT_FAILURE;
    }
    auto bufferId = sourceMgr.AddNewSourceBuffer(std::move(*source), {});
    auto mlirModule = parseCodeModule(sourceMgr, bufferId, mlirContext);

    mlir::PassManager pm(&mlirContext);
    loadPasses(pm);
    if (mlir::failed(mlir::applyPassManagerCLOptions(pm)) ||
        mlir::failed(pm.run(*mlirModule)))
        return EXIT_FAILURE;

    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(*mlirModule, llvmContext);
    // need to write it to a file and then call lld to make it an executable

    return EXIT_SUCCESS;
}