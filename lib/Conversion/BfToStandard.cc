#include "Conversion/Passes.hh"
#include "IR/BFOps.hh"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

#define GEN_PASS_DEF_BFTOSTANDARD
#include "Conversion/Passes.h.inc"

namespace {

auto constexpr kDataPointerName = "__data_ptr";
auto constexpr kDataArrayName = "__data";
// TODO: make the data array size configurable
auto constexpr kDataArraySize = 30000l;
auto constexpr kGetCharName = "getchar";
auto constexpr kPutCharName = "putchar";

void insertGlobalSymbols(mlir::Operation *rootOp) {
    auto *symbolOp = mlir::SymbolTable::getNearestSymbolTable(rootOp);
    assert(symbolOp);
    mlir::SymbolTable symbolTable(symbolOp);
    mlir::OpBuilder builder(rootOp);
    builder.setInsertionPointToEnd(&symbolOp->getRegion(0).back());
    auto loc = builder.getUnknownLoc();

    auto i32 = builder.getI32Type();
    auto i8 = builder.getI8Type();

    if (!symbolTable.lookup(kDataPointerName))
        builder.create<mlir::LLVM::GlobalOp>(
            loc, i32, /*isConstant*/ false, mlir::LLVM::Linkage::Private,
            kDataPointerName, builder.getZeroAttr(i32));

    auto array = builder.getType<mlir::LLVM::LLVMArrayType>(i8, kDataArraySize);
    if (!symbolTable.lookup(kDataArrayName))
        builder.create<mlir::LLVM::GlobalOp>(
            loc, array, /*isConstant*/ false, mlir::LLVM::Linkage::Private,
            kDataArrayName,
            builder.getZeroAttr(mlir::RankedTensorType::get(
                llvm::SmallVector{kDataArraySize}, i8)));

    // functions
    auto getCharTy = builder.getFunctionType({}, {i8});
    auto putCharTy = builder.getFunctionType({i8}, {});
    if (!symbolTable.lookup(kGetCharName))
        builder.create<mlir::func::FuncOp>(loc, kGetCharName, getCharTy);
    if (!symbolTable.lookup(kPutCharName))
        builder.create<mlir::func::FuncOp>(loc, kPutCharName, putCharTy);
}

struct ConvertPtrIncrement : mlir::OpConversionPattern<mlir::bf::IncrementPtr> {
    using mlir::OpConversionPattern<
        mlir::bf::IncrementPtr>::OpConversionPattern;

    mlir::LogicalResult
    matchAndRewrite(mlir::bf::IncrementPtr op, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {
        rewriter.setInsertionPoint(op);
        auto loc = rewriter.getUnknownLoc();
        auto ptrTy = rewriter.getType<mlir::LLVM::LLVMPointerType>();
        auto i32 = rewriter.getI32Type();
        auto ptr = rewriter.create<mlir::LLVM::AddressOfOp>(loc, ptrTy,
                                                            kDataPointerName);
        auto load = rewriter.create<mlir::LLVM::LoadOp>(loc, i32, ptr);
        // add 1
        auto one = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, 32);
        auto add = rewriter.create<mlir::arith::AddIOp>(op.getLoc(), load, one);
        rewriter.create<mlir::LLVM::StoreOp>(loc, add, ptr);
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct BfToStandard : impl::BFToStandardBase<BfToStandard> {
    void runOnOperation() override {
        // first, populate the global values
        insertGlobalSymbols(getOperation());
        // second, split basic blocks based on the loop operations
        // direct dialect conversion
        auto context = &getContext();

        mlir::RewritePatternSet patterns(context);

        mlir::ConversionTarget target(*context);
        target.addIllegalDialect<mlir::bf::BFDialect>();
        if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                      std::move(patterns))))
            return signalPassFailure();
    }
};

} // namespace

namespace mlir::bf {
std::unique_ptr<::mlir::Pass> createBFToStandard() {
    return std::make_unique<BfToStandard>();
}
} // namespace mlir::bf