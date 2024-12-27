#include "Conversion/Passes.hh"
#include "IR/BFOps.hh"

#define GEN_PASS_DEF_VALIDATEBF
#include "Transforms/Passes.h.inc"

namespace {

bool isLoopStartValidate(mlir::bf::LoopStart startOp) {
    uint32_t startCount = 0;
    mlir::Operation *nextOp = startOp;
    while (nextOp) {
        if (mlir::isa<mlir::bf::LoopStart>(nextOp)) {
            ++startCount;
        } else if (mlir::isa<mlir::bf::LoopEnd>(nextOp)) {
            --startCount;
        }
        if (startCount == 0)
            break;
        nextOp = nextOp->getNextNode();
    }
    return startCount == 0;
}

struct ValidateBF : impl::ValidateBFBase<ValidateBF> {
    void runOnOperation() override {
        if (getOperation()
                ->walk([](mlir::bf::LoopStart startOp) {
                    auto valid = isLoopStartValidate(startOp);
                    if (!valid) {
                        startOp->emitError("missing matching ']'");
                        return mlir::WalkResult::interrupt();
                    }
                    return mlir::WalkResult::advance();
                })
                .wasInterrupted())
            return signalPassFailure();
    }
};
} // namespace

namespace mlir::bf {
std::unique_ptr<::mlir::Pass> createValidateBF() {
    return std::make_unique<ValidateBF>();
}
} // namespace mlir::bf
