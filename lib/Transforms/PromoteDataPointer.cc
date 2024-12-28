#include "Conversion/Passes.hh"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_PROMOTEDATAPOINTER
#include "Transforms/Passes.h.inc"

namespace {
auto constexpr kDataPointerName = "__data_ptr";

struct PromoteDataPtrAddressOf
    : mlir::OpRewritePattern<mlir::LLVM::AddressOfOp> {
    using mlir::OpRewritePattern<mlir::LLVM::AddressOfOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(mlir::LLVM::AddressOfOp op,
                    mlir::PatternRewriter &rewriter) const override {
        // assume we have run canonicalizer patterns before, which leaves
        // only one addressof op
        auto sym = op.getGlobalName();
        if (sym != kDataPointerName)
            return mlir::failure();

        auto *topBlock = &op->getParentRegion()->front();
        rewriter.setInsertionPointToStart(topBlock);
        mlir::SymbolTableCollection symbolTableCollection;
        auto globalOp = op.getGlobal(symbolTableCollection);
        auto ty = globalOp.getType();
        auto one =
            rewriter.create<mlir::arith::ConstantIntOp>(op->getLoc(), 1, 32);
        auto alloca = rewriter.create<mlir::LLVM::AllocaOp>(
            op.getLoc(), op.getType(), ty, one);
        // 0-initialized
        auto zero = rewriter.create<mlir::arith::ConstantOp>(
            op->getLoc(), ty, rewriter.getZeroAttr(ty));
        rewriter.create<mlir::LLVM::StoreOp>(op->getLoc(), zero, alloca);
        rewriter.replaceOp(op, alloca);
        rewriter.eraseOp(globalOp);
        return mlir::success();
    }
};

struct PromoteDataPointer : impl::PromoteDataPointerBase<PromoteDataPointer> {
    void runOnOperation() override {
        auto *context = &getContext();
        mlir::RewritePatternSet patterns(context);
        patterns.insert<PromoteDataPtrAddressOf>(context);
        if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
                getOperation(), std::move(patterns))))
            return signalPassFailure();
    }
};
} // namespace

namespace mlir::bf {
std::unique_ptr<::mlir::Pass> createPromoteDataPointer() {
    return std::make_unique<PromoteDataPointer>();
}
} // namespace mlir::bf