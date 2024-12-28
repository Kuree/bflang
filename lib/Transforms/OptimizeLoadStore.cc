#include "Conversion/Passes.hh"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_OPTIMIZELOADSTORE
#include "Transforms/Passes.h.inc"

namespace {

struct SameGepValue : mlir::OpRewritePattern<mlir::LLVM::LoadOp> {
    using mlir::OpRewritePattern<mlir::LLVM::LoadOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(mlir::LLVM::LoadOp op,
                    mlir::PatternRewriter &rewriter) const override {
        auto addr = op.getAddr().getDefiningOp<mlir::LLVM::GEPOp>();
        if (!addr)
            return mlir::failure();
        if (addr.getIndices().size() != 1)
            return mlir::failure();
        auto addrIdx = addr.getIndices()[0];

        auto prevOp = op->getPrevNode();
        while (prevOp) {
            if (auto store = mlir::dyn_cast<mlir::LLVM::StoreOp>(prevOp)) {
                if (auto storeAddr =
                        store.getAddr().getDefiningOp<mlir::LLVM::GEPOp>()) {
                    if (storeAddr.getIndices().size() != 1)
                        return mlir::failure();
                    auto storeAddrIdx = storeAddr.getIndices()[0];

                    if (addrIdx != storeAddrIdx) {
                        if (mlir::isa<mlir::Value>(addrIdx) ||
                            mlir::isa<mlir::Value>(storeAddrIdx)) {
                            return mlir::failure();
                        }
                    } else {
                        // replace the store value
                        rewriter.replaceOp(op, store.getValue());
                        return mlir::success();
                    }
                }
            }

            prevOp = prevOp->getPrevNode();
        }
        return mlir::failure();
    }
};

struct OptimizeLoadStore : impl::OptimizeLoadStoreBase<OptimizeLoadStore> {
    void runOnOperation() override {
        auto *context = &getContext();
        mlir::RewritePatternSet patterns(context);
        patterns.insert<SameGepValue>(context);
        if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
                getOperation(), std::move(patterns))))
            return signalPassFailure();
    }
};
} // namespace

namespace mlir::bf {
std::unique_ptr<::mlir::Pass> createOptimizeLoadStore() {
    return std::make_unique<OptimizeLoadStore>();
}
} // namespace mlir::bf