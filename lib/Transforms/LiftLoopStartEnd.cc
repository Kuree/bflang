#include "Conversion/Passes.hh"
#include "IR/BFOps.hh"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_LIFTLOOPSTARTEND
#include "Transforms/Passes.h.inc"

namespace {

struct ConvertStart : mlir::OpRewritePattern<mlir::bf::LoopStart> {
    using mlir::OpRewritePattern<mlir::bf::LoopStart>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(mlir::bf::LoopStart op,
                    mlir::PatternRewriter &rewriter) const override {
        // we only focus on the innermost scope

        auto *nextOp = op->getNextNode();
        while (nextOp) {
            if (mlir::isa<mlir::bf::LoopStart>(nextOp))
                return mlir::failure();
            if (mlir::isa<mlir::bf::LoopEnd>(nextOp))
                break;
            nextOp = nextOp->getNextNode();
        }
        auto endOp = mlir::dyn_cast_or_null<mlir::bf::LoopEnd>(nextOp);
        if (!endOp)
            return mlir::failure();

        rewriter.setInsertionPoint(op);
        auto loop = rewriter.create<mlir::bf::Loop>(op->getLoc());
        auto *finalBlock = rewriter.createBlock(&loop.getRegion());
        // split and inline
        rewriter.setInsertionPoint(op);
        // bb0:
        // loop
        // bb1:
        //     ....
        rewriter.splitBlock(op->getBlock(), rewriter.getInsertionPoint());

        rewriter.setInsertionPointAfter(endOp);
        auto *loopBlock = endOp->getBlock();
        auto *dstBlock =
            rewriter.splitBlock(op->getBlock(), rewriter.getInsertionPoint());
        // bb0:
        //  loop
        // bb1:
        //  ...
        // bb2: <- dstBlock
        rewriter.inlineBlockBefore(loopBlock, finalBlock, finalBlock->end());
        rewriter.eraseOp(op);
        rewriter.eraseOp(endOp);

        rewriter.setInsertionPointAfter(loop);
        rewriter.inlineBlockBefore(dstBlock, loop->getBlock(),
                                   rewriter.getInsertionPoint());

        return mlir::success();
    }
};

struct LiftLoopStartEnd : impl::LiftLoopStartEndBase<LiftLoopStartEnd> {
    void runOnOperation() override {
        auto *context = &getContext();
        mlir::RewritePatternSet patterns(context);
        patterns.insert<ConvertStart>(context);
        if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
                getOperation(), std::move(patterns))))
            return signalPassFailure();
    }
};
} // namespace

namespace mlir::bf {
std::unique_ptr<::mlir::Pass> createLiftLoopStartEnd() {
    return std::make_unique<LiftLoopStartEnd>();
}
} // namespace mlir::bf