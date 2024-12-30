#include "Conversion/Passes.hh"
#include "IR/BFOps.hh"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
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
                llvm::SmallVector{static_cast<int64_t>(kDataArraySize)}, i8)));

    // functions
    auto getCharTy = builder.getFunctionType({}, {i8});
    auto putCharTy = builder.getFunctionType({i8}, {});
    if (!symbolTable.lookup(kGetCharName)) {
        auto funcOp =
            builder.create<mlir::func::FuncOp>(loc, kGetCharName, getCharTy);
        funcOp.setVisibility(mlir::SymbolTable::Visibility::Private);
    }

    if (!symbolTable.lookup(kPutCharName)) {
        auto funcOp =
            builder.create<mlir::func::FuncOp>(loc, kPutCharName, putCharTy);
        funcOp.setVisibility(mlir::SymbolTable::Visibility::Private);
    }
}

std::pair<mlir::Value, mlir::Value> getDataPointer(mlir::OpBuilder &builder) {
    auto loc = builder.getUnknownLoc();
    auto ptrTy = builder.getType<mlir::LLVM::LLVMPointerType>();
    auto i32 = builder.getI32Type();
    auto ptr =
        builder.create<mlir::LLVM::AddressOfOp>(loc, ptrTy, kDataPointerName);
    auto load = builder.create<mlir::LLVM::LoadOp>(loc, i32, ptr);
    return {ptr, load};
}

std::pair<mlir::Value, mlir::Value>
getDataPointerValue(mlir::OpBuilder &builder, bool createLoad = true) {
    auto [ptr, ptrValue] = getDataPointer(builder);
    auto loc = ptr.getLoc();
    auto ptrTy = builder.getType<mlir::LLVM::LLVMPointerType>();
    auto i8 = builder.getI8Type();
    auto dataPtr =
        builder.create<mlir::LLVM::AddressOfOp>(loc, ptrTy, kDataArrayName);
    auto gep = builder.create<mlir::LLVM::GEPOp>(
        loc, ptrTy, i8, dataPtr, llvm::SmallVector<mlir::Value>{ptrValue});
    mlir::Value load;
    if (createLoad)
        load = builder.create<mlir::LLVM::LoadOp>(loc, i8, gep);
    return {gep, load};
}

struct ConvertPtrIncrement : mlir::OpConversionPattern<mlir::bf::IncrementPtr> {
    using mlir::OpConversionPattern<
        mlir::bf::IncrementPtr>::OpConversionPattern;

    mlir::LogicalResult
    matchAndRewrite(mlir::bf::IncrementPtr op, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {
        rewriter.setInsertionPoint(op);
        auto loc = rewriter.getUnknownLoc();
        auto [ptr, load] = getDataPointer(rewriter);
        // add 1
        auto one = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, 32);
        auto add = rewriter.create<mlir::arith::AddIOp>(op.getLoc(), load, one);
        rewriter.create<mlir::LLVM::StoreOp>(loc, add, ptr);
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct ConvertPtrDecrement : mlir::OpConversionPattern<mlir::bf::DecrementPtr> {
    using mlir::OpConversionPattern<
        mlir::bf::DecrementPtr>::OpConversionPattern;

    mlir::LogicalResult
    matchAndRewrite(mlir::bf::DecrementPtr op, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {
        rewriter.setInsertionPoint(op);
        auto loc = rewriter.getUnknownLoc();
        auto [ptr, load] = getDataPointer(rewriter);
        // add -1
        auto negOne = rewriter.create<mlir::arith::ConstantIntOp>(loc, -1, 32);
        auto add =
            rewriter.create<mlir::arith::AddIOp>(op.getLoc(), load, negOne);
        rewriter.create<mlir::LLVM::StoreOp>(loc, add, ptr);
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct ConvertDataIncrement
    : mlir::OpConversionPattern<mlir::bf::IncrementData> {
    using mlir::OpConversionPattern<
        mlir::bf::IncrementData>::OpConversionPattern;

    mlir::LogicalResult
    matchAndRewrite(mlir::bf::IncrementData op, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {
        rewriter.setInsertionPoint(op);
        auto loc = rewriter.getUnknownLoc();
        auto [gep, load] = getDataPointerValue(rewriter);
        // add 1
        auto one = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, 8);
        auto add = rewriter.create<mlir::arith::AddIOp>(op.getLoc(), load, one);
        rewriter.create<mlir::LLVM::StoreOp>(loc, add, gep);
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct ConvertDataDecrement
    : mlir::OpConversionPattern<mlir::bf::DecrementData> {
    using mlir::OpConversionPattern<
        mlir::bf::DecrementData>::OpConversionPattern;

    mlir::LogicalResult
    matchAndRewrite(mlir::bf::DecrementData op, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {
        rewriter.setInsertionPoint(op);
        auto loc = rewriter.getUnknownLoc();
        auto [gep, load] = getDataPointerValue(rewriter);
        // add -1
        auto negOne = rewriter.create<mlir::arith::ConstantIntOp>(loc, -1, 8);
        auto add =
            rewriter.create<mlir::arith::AddIOp>(op.getLoc(), load, negOne);
        rewriter.create<mlir::LLVM::StoreOp>(loc, add, gep);
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct ConvertOutput : mlir::OpConversionPattern<mlir::bf::Output> {
    using mlir::OpConversionPattern<mlir::bf::Output>::OpConversionPattern;

    mlir::LogicalResult
    matchAndRewrite(mlir::bf::Output op, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {
        rewriter.setInsertionPoint(op);
        auto [gep, load] = getDataPointerValue(rewriter);
        // call putchar
        rewriter.create<mlir::func::CallOp>(
            op.getLoc(), kPutCharName, llvm::SmallVector<mlir::Type>{},
            llvm::SmallVector<mlir::Value>{load});
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct ConvertInput : mlir::OpConversionPattern<mlir::bf::Input> {
    using mlir::OpConversionPattern<mlir::bf::Input>::OpConversionPattern;

    mlir::LogicalResult
    matchAndRewrite(mlir::bf::Input op, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {
        rewriter.setInsertionPoint(op);
        auto [gep, load] = getDataPointerValue(rewriter, /*createLoad*/ false);
        // call getchar
        auto i8 = rewriter.getI8Type();
        auto val = rewriter.create<mlir::func::CallOp>(
            op.getLoc(), kGetCharName, llvm::SmallVector<mlir::Type>{i8},
            llvm::SmallVector<mlir::Value>{});
        rewriter.create<mlir::LLVM::StoreOp>(gep.getLoc(), val.getResult(0),
                                             gep);
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct ConvertLoop : mlir::OpConversionPattern<mlir::bf::Loop> {
    using mlir::OpConversionPattern<mlir::bf::Loop>::OpConversionPattern;

    mlir::LogicalResult
    matchAndRewrite(mlir::bf::Loop op, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {
        auto *currentBlock = op->getBlock();

        auto loc = rewriter.getUnknownLoc();
        rewriter.setInsertionPoint(op);
        auto zero = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, 8);

        rewriter.setInsertionPointAfter(op);
        auto *loopBlock =
            rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
        rewriter.setInsertionPointToStart(loopBlock);
        auto *dstBlock =
            rewriter.splitBlock(loopBlock, rewriter.getInsertionPoint());

        {
            // handle `[`
            rewriter.setInsertionPoint(op);
            auto [gep, load] = getDataPointerValue(rewriter);
            auto cmp = rewriter.create<mlir::arith::CmpIOp>(
                op.getLoc(), mlir::arith::CmpIPredicate::eq, load, zero);
            // if zero, jump to the dst block
            rewriter.create<mlir::cf::CondBranchOp>(op.getLoc(), cmp, dstBlock,
                                                    loopBlock);
        }

        {
            // handle the inner body
            if (!op.getRegion().empty()) {
                auto *srcBlock = &op.getRegion().front();
                rewriter.inlineBlockBefore(srcBlock, loopBlock,
                                           loopBlock->end());
            }
        }

        {
            // handle `]`
            rewriter.setInsertionPointToEnd(loopBlock);
            auto [gep, load] = getDataPointerValue(rewriter);
            auto cmp = rewriter.create<mlir::arith::CmpIOp>(
                op.getLoc(), mlir::arith::CmpIPredicate::ne, load, zero);
            // if non-zero, jump to loop block
            rewriter.create<mlir::cf::CondBranchOp>(op.getLoc(), cmp, loopBlock,
                                                    dstBlock);
        }

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct RemoveYield : mlir::OpConversionPattern<mlir::bf::LoopYield> {
    using mlir::OpConversionPattern<mlir::bf::LoopYield>::OpConversionPattern;
    mlir::LogicalResult
    matchAndRewrite(mlir::bf::LoopYield op, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const override {
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
        patterns.insert<ConvertPtrIncrement, ConvertPtrDecrement,
                        ConvertDataIncrement, ConvertDataDecrement,
                        ConvertOutput, ConvertInput, ConvertLoop, RemoveYield>(
            context);

        mlir::ConversionTarget target(*context);
        target.addIllegalDialect<mlir::bf::BFDialect>();
        target.addLegalDialect<mlir::arith::ArithDialect>();
        target.addLegalDialect<mlir::cf::ControlFlowDialect>();
        target.addLegalDialect<mlir::LLVM::LLVMDialect>();
        target.addLegalDialect<mlir::func::FuncDialect>();
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