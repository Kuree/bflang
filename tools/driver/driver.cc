#include "lld/Common/Driver.h"
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
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#include "Conversion/Passes.hh"
#include "IR/BFOps.hh"
#include "Transforms/Passes.hh"

LLD_HAS_DRIVER(elf)

namespace {
enum OptLevel { O0, O1 };

llvm::cl::opt<bool> emitAssembly{
    "S", llvm::cl::desc("Produce an assembly file in MLIR.")};

llvm::cl::opt<bool> emitLLVM{"emit-llvm", llvm::cl::desc("Emit LLVM IR")};

llvm::cl::opt<bool> emitMLIR{"emit-mlir", llvm::cl::desc("Emit MLIR IR")};

llvm::cl::opt<bool> emitBfMLIR{"emit-bf",
                               llvm::cl::desc("Emit MLIR in BF dialect")};

llvm::cl::opt<std::string> outputFileName{
    "o", llvm::cl::desc("Write output to file."), llvm::cl::value_desc("file"),
    llvm::cl::Required};

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

    if (emitAssembly && emitMLIR)
        return;

    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
}

mlir::LogicalResult runLinker(llvm::Module &module) {
    auto tempFile = llvm::sys::fs::TempFile::create("bfc.%%%%%%.bc");
    if (!tempFile) {
        llvm::errs() << "error: unable to create temp file\n";
        return mlir::failure();
    }
    llvm::raw_fd_ostream os(tempFile->FD, false);
    llvm::WriteBitcodeToFile(module, os);
    os.flush();

    // call the linker
#ifdef __gnu_linux__
    std::string linkerName = "ld.lld";
#else
    llvm::errs() << "Unsupported operating system\n";
    return mlir::failure();
#endif
    llvm::SmallVector<const char *> args = {linkerName.c_str(),
                                            "-z",
                                            "relro",
                                            "--hash-style=gnu",
                                            "--build-id",
                                            "--eh-frame-hdr",
                                            "-m",
                                            "elf_x86_64",
                                            "-pie",
                                            "-dynamic-linker",
                                            "/lib64/ld-linux-x86-64.so.2"};

#ifndef __x86_64__
    llvm::errs() << "Unsupported architecture\n";
    return mlir::failure();
#endif
    args.emplace_back("/lib/x86_64-linux-gnu/Scrt1.o");
    args.emplace_back(tempFile->TmpName.c_str());
    args.emplace_back("-o");
    auto outName = std::string{outputFileName.getValue()};
    args.emplace_back(outName.c_str());

    // the linker flag construction is from here
    // https://clang.llvm.org/doxygen/classclang_1_1driver_1_1tools_1_1gnutools_1_1Linker.html
    // copied from `clang -v` on linux

    args.emplace_back("-L/lib/x86_64-linux-gnu");
    args.emplace_back("-lc");

    auto ldError = lld::lldMain(args, llvm::outs(), llvm::errs(),
                                {{lld::Gnu, &lld::elf::link}});
    if (ldError.retCode) {
        (void)tempFile->discard();
        return mlir::failure();
    }
    auto error = tempFile->discard();
    if (error) {
        llvm::errs() << "error: " << error << "\n";
        return mlir::failure();
    }
    return mlir::success();
}

std::optional<llvm::DataLayout> getDataLayout() {
    std::string error;
    llvm::InitializeNativeTarget();
    auto *const target =
        llvm::TargetRegistry::lookupTarget(LLVM_DEFAULT_TARGET_TRIPLE, error);
    if (!target) {
        llvm::errs() << "error: " << error << "\n";
        return std::nullopt;
    }

    auto tm =
        target->createTargetMachine(LLVM_DEFAULT_TARGET_TRIPLE, "", "", {}, {});
    return tm->createDataLayout();
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
    llvmModule->setTargetTriple(LLVM_DEFAULT_TARGET_TRIPLE);
    auto dataLayout = getDataLayout();
    if (!dataLayout) {
        llvm::errs() << "Unable to get data layout\n";
        return EXIT_FAILURE;
    }
    llvmModule->setDataLayout(*dataLayout);
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
