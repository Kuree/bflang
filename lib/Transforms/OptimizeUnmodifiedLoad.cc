#include "Conversion/Passes.hh"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_OPTIMIZEUNMODIFIEDLOAD
#include "Transforms/Passes.h.inc"

namespace {

auto constexpr kDataArrayName = "__data";

struct UnmodifiedLoadGep : mlir::OpRewritePattern<mlir::LLVM::LoadOp> {
    using mlir::OpRewritePattern<mlir::LLVM::LoadOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(mlir::LLVM::LoadOp op,
                    mlir::PatternRewriter &rewriter) const override {
        // observation:
        // we could use sparse data flow analysis to figure out the basic
        // block predecessors in a graph and see if the address has been
        // modified. however, unless we infer the address range from the loop
        // the constant address accessing must be the block that does not
        // have a predecessor. This is because BF does not have a conditional
        // jump without a loop construct

        auto *block = op->getBlock();
        if (!block->getPredecessors().empty())
            return mlir::failure();

        auto addr = op.getAddr().getDefiningOp<mlir::LLVM::GEPOp>();
        if (!addr)
            return mlir::failure();
        if (addr.getIndices().size() != 1)
            return mlir::failure();
        // must be a constant
        auto addrIdx = addr.getIndices()[0];
        if (mlir::isa<mlir::Value>(addrIdx))
            return mlir::failure();
        // gep base must be addressof the data
        auto baseAddr = addr.getBase().getDefiningOp<mlir::LLVM::AddressOfOp>();
        if (!baseAddr || baseAddr.getGlobalName() != kDataArrayName)
            return mlir::failure();

        auto prevOp = op->getPrevNode();
        while (prevOp) {
            if (auto store = mlir::dyn_cast<mlir::LLVM::StoreOp>(prevOp)) {
                if (auto storeAddr =
                        store.getAddr().getDefiningOp<mlir::LLVM::GEPOp>()) {
                    if (storeAddr.getIndices().size() != 1 ||
                        storeAddr.getBase() != addr.getBase())
                        return mlir::failure();
                    auto storeAddrIdx = storeAddr.getIndices()[0];
                    // cannot be an address range or the same constant value
                    if (mlir::isa<mlir::Value>(storeAddrIdx) ||
                        storeAddrIdx == addrIdx)
                        return mlir::failure();
                }
            }

            prevOp = prevOp->getPrevNode();
        }

        // replace it with zero
        rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(
            op, op.getType(), rewriter.getZeroAttr(op.getType()));

        return mlir::success();
    }
};

struct UnmodifiedLoad : mlir::OpRewritePattern<mlir::LLVM::LoadOp> {
    using mlir::OpRewritePattern<mlir::LLVM::LoadOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(mlir::LLVM::LoadOp op,
                    mlir::PatternRewriter &rewriter) const override {
        auto *block = op->getBlock();
        if (!block->getPredecessors().empty())
            return mlir::failure();
        auto addr = op.getAddr().getDefiningOp<mlir::LLVM::AddressOfOp>();
        if (!addr || addr.getGlobalName() != kDataArrayName)
            return mlir::failure();

        auto prevOp = op->getPrevNode();
        // assume gep[0] is optimized away
        while (prevOp) {
            if (auto store = mlir::dyn_cast<mlir::LLVM::StoreOp>(prevOp)) {
                if (store.getAddr() == op.getAddr())
                    return mlir::failure();
            }
            prevOp = prevOp->getPrevNode();
        }
        // replace it with zero
        rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(
            op, op.getType(), rewriter.getZeroAttr(op.getType()));
        return mlir::success();
    }
};

struct OptimizeUnmodifiedLoad
    : impl::OptimizeUnModifiedLoadBase<OptimizeUnmodifiedLoad> {
    void runOnOperation() override {
        auto *context = &getContext();
        mlir::RewritePatternSet patterns(context);
        patterns.insert<UnmodifiedLoadGep, UnmodifiedLoad>(context);
        mlir::arith::CmpIOp::getCanonicalizationPatterns(patterns, context);
        mlir::arith::AddIOp::getCanonicalizationPatterns(patterns, context);
        mlir::cf::CondBranchOp::getCanonicalizationPatterns(patterns, context);

        if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
                getOperation(), std::move(patterns))))
            return signalPassFailure();
    }
};
} // namespace

namespace mlir::bf {
std::unique_ptr<::mlir::Pass> createOptimizeUnModifiedLoad() {
    return std::make_unique<OptimizeUnmodifiedLoad>();
}
} // namespace mlir::bf