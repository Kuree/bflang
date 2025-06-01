#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Frontend/Utils.h"
#include <clang/Frontend/TextDiagnosticBuffer.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>

#include "Conversion/Passes.hh"
#include "IR/BFOps.hh"
#include "Transforms/Passes.hh"

namespace {
enum OptLevel { O0, O1 };

llvm::cl::OptionCategory
    compilerCategory("Compiler Options",
                     "Options for controlling the compilation process.");

llvm::cl::opt<bool> emitAssembly{
    "S", llvm::cl::desc("Produce an assembly file in MLIR."),
    llvm::cl::cat(compilerCategory)};

llvm::cl::opt<bool> emitLLVM{"emit-llvm", llvm::cl::desc("Emit LLVM IR"),
                             llvm::cl::cat(compilerCategory)};

llvm::cl::opt<bool> emitMLIR{"emit-mlir", llvm::cl::desc("Emit MLIR IR"),
                             llvm::cl::cat(compilerCategory)};

llvm::cl::opt<bool> emitBfMLIR{"emit-bf",
                               llvm::cl::desc("Emit MLIR in BF dialect"),
                               llvm::cl::cat(compilerCategory)};

llvm::cl::opt<uint32_t> arraySize{
    "array-size",
    llvm::cl::desc("Array size. Per initial standard it's set to 30,000"),
    llvm::cl::init(30000l), llvm::cl::cat(compilerCategory)};

llvm::cl::opt<std::string> outputFileName{
    "o", llvm::cl::desc("Write output to file."), llvm::cl::value_desc("file"),
    llvm::cl::Required, llvm::cl::cat(compilerCategory)};

llvm::cl::opt<OptLevel> optimizationLevel{
    llvm::cl::desc("Choose optimization level:"),
    llvm::cl::values(clEnumVal(O0, "No optimization"),
                     clEnumVal(O1, "Enable default optimizations")),
    llvm::cl::cat(compilerCategory)};

llvm::cl::opt<std::string> inputFileName(llvm::cl::Positional,
                                         llvm::cl::desc("<input file>"));

llvm::cl::opt<bool> enableDebug("g",
                                llvm::cl::desc("Generate debug information"),
                                llvm::cl::cat(compilerCategory));

llvm::cl::opt<std::string> targetTriple{
    "target", llvm::cl::desc("Generate code for the given target"),
    llvm::cl::init(LLVM_DEFAULT_TARGET_TRIPLE),
    llvm::cl::cat(compilerCategory)};

llvm::cl::opt<std::string> linkerName{
    "fuse-ld", llvm::cl::init(""), llvm::cl::cat(compilerCategory),
    llvm::cl::desc("Linker name"), llvm::cl::AlwaysPrefix};

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

    if (emitAssembly && emitBfMLIR)
        return;

    pm.addPass(mlir::bf::createBFToStandard());

    if (optimizationLevel == OptLevel::O1) {
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::bf::createPromoteDataPointer());
        pm.addPass(mlir::createMem2Reg());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::bf::createOptimizeUnModifiedLoad());
    }

    if (enableDebug)
        pm.addPass(mlir::bf::createAttachDebugInfo());

    if (emitAssembly && emitMLIR)
        return;

    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
}

struct TempFileGuard {
    explicit TempFileGuard(llvm::sys::fs::TempFile &tempFile)
        : tempFile(tempFile) {}
    llvm::sys::fs::TempFile &tempFile;

    ~TempFileGuard() {
        auto error = tempFile.discard();
        if (error) {
            llvm::errs() << "error: failed to delete " << tempFile.TmpName;
        }
    }
};

mlir::LogicalResult runLinker(llvm::Module &module) {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();

    std::string error;
    auto *target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
    if (!target) {
        llvm::errs() << error << "\n";
        return mlir::failure();
    }

    llvm::TargetOptions opt;
    auto targetMachine = target->createTargetMachine(
        targetTriple, "generic", "", opt, llvm::Reloc::PIC_);

    module.setDataLayout(targetMachine->createDataLayout());
    module.setTargetTriple(targetTriple);

    auto outName = std::string{outputFileName.getValue()};
    auto tempFile = llvm::sys::fs::TempFile::create(outName + ".bfc.%%%%%%.o");
    if (!tempFile) {
        llvm::errs() << "error: unable to create temp file\n";
        return mlir::failure();
    }
    TempFileGuard fileGuard(*tempFile);

    llvm::raw_fd_ostream os(tempFile->FD, false);

    llvm::legacy::PassManager pass;
    auto FileType = llvm::CodeGenFileType::ObjectFile;

    if (targetMachine->addPassesToEmitFile(pass, os, nullptr, FileType)) {
        llvm::errs() << "error: target can't emit a file of this type\n";
        return mlir::failure();
    }

    pass.run(module);
    os.flush();

    clang::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagOpts =
        new clang::DiagnosticOptions();
    clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagID(
        new clang::DiagnosticIDs());
    clang::TextDiagnosticPrinter diagBuffer(llvm::errs(), &*diagOpts);
    clang::DiagnosticsEngine diagnosticsEngine(diagID, diagOpts, &diagBuffer,
                                               false);
    clang::driver::Driver driver("", targetTriple, diagnosticsEngine);
    llvm::SmallVector<const char *> args = {"", "-o", outName.c_str(),
                                            tempFile->TmpName.c_str()};
    auto linkerNameOpt = "-fuse-ld" + linkerName;
    if (!linkerName.empty()) {
        args.emplace_back(linkerNameOpt.c_str());
    }
    auto targetStr = "--target=" + targetTriple.getValue();
    args.emplace_back(targetStr.c_str());

    auto compilation = driver.BuildCompilation(args);
    if (!compilation) {
        llvm::errs() << "error: fail to build compilation job\n";
        return mlir::failure();
    }

    llvm::SmallVector<std::pair<int, const clang::driver::Command *>>
        failingCommands;
    auto res = driver.ExecuteCompilation(*compilation, failingCommands);

    if (res) {
        for (auto [i, cmd] : failingCommands) {
            driver.generateCompilationDiagnostics(*compilation, *cmd);
        }
        llvm::errs() << "error: failed to link\n";
        return mlir::failure();
    }
    return mlir::success();
}

template <typename T> mlir::LogicalResult emitIRToOutput(T *module) {
    if (outputFileName == "-") {
        llvm::outs() << *module;
        return mlir::success();
    }

    std::error_code ec;
    llvm::raw_fd_ostream os(outputFileName, ec);
    if (ec) {
        llvm::errs() << ec.message() << "\n";
        return mlir::failure();
    }
    os << *module;
    return mlir::success();
}

mlir::LogicalResult checkCLIArgs() {
    if (!emitAssembly && (emitBfMLIR || emitLLVM || emitMLIR)) {
        llvm::errs() << "'-S' must be specified to emit IR\n";
        return mlir::failure();
    }
    return mlir::success();
}

} // namespace

int main(int argc, char *argv[]) {
    llvm::cl::HideUnrelatedOptions(compilerCategory);
    llvm::cl::ParseCommandLineOptions(argc, argv);
    if (mlir::failed(checkCLIArgs()))
        return EXIT_FAILURE;

    mlir::MLIRContext mlirContext;
    mlirContext
        .loadDialect<mlir::cf::ControlFlowDialect, mlir::arith::ArithDialect,
                     mlir::scf::SCFDialect, mlir::func::FuncDialect,
                     mlir::LLVM::LLVMDialect, mlir::bf::BFDialect>();
    mlir::registerBuiltinDialectTranslation(mlirContext);
    mlir::registerLLVMDialectTranslation(mlirContext);

    // don't show the op to user
    mlirContext.printOpOnDiagnostic(false);
    llvm::SourceMgr sourceMgr;
    mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &mlirContext);

    auto source = llvm::MemoryBuffer::getFileOrSTDIN(
        inputFileName.empty() ? std::string{"-"} : inputFileName.getValue());
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
    if (mlir::failed(pm.run(*mlirModule)))
        return EXIT_FAILURE;

    if (emitAssembly && (emitBfMLIR || emitMLIR)) {
        return mlir::failed(emitIRToOutput(mlirModule->getOperation()))
                   ? EXIT_FAILURE
                   : EXIT_SUCCESS;
    }

    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(*mlirModule, llvmContext);

    if (optimizationLevel != OptLevel::O0) {
        auto optPipeline = mlir::makeOptimizingTransformer(3, 0, nullptr);
        auto error = optPipeline(llvmModule.get());
        if (error) {
            llvm::errs() << "unable to optimize LLVM IR\n";
            return EXIT_FAILURE;
        }
    }

    if (emitAssembly && emitLLVM) {
        return mlir::failed(emitIRToOutput(llvmModule.get())) ? EXIT_FAILURE
                                                              : EXIT_SUCCESS;
    }

    // need to write it to a file and then call lld to make it an executable
    if (mlir::failed(runLinker(*llvmModule)))
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
