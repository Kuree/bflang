#include "Conversion/Passes.hh"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/FormatVariadic.h"

#define GEN_PASS_DEF_ATTACHDEBUGINFO
#include "Transforms/Passes.h.inc"

#include <filesystem>

namespace {

struct AttachDebugInfo : impl::AttachDebugInfoBase<AttachDebugInfo> {
    void runOnOperation() override {
        mlir::Builder builder(&getContext());
        auto fileLoc =
            mlir::dyn_cast<mlir::FileLineColLoc>(getOperation()->getLoc());
        if (!fileLoc)
            return;
        auto fileName = std::filesystem::path(fileLoc.getFilename().str());
        auto dirname = fileName.parent_path();
        auto fileBaseName = fileName.filename();
        auto diFile = builder.getAttr<mlir::LLVM::DIFileAttr>(
            std::string{fileBaseName}, std::string{dirname});
        auto diId = mlir::DistinctAttr::create(builder.getUnitAttr());
        auto diCu = builder.getAttr<mlir::LLVM::DICompileUnitAttr>(
            diId, llvm::dwarf::DW_LANG_C11, diFile,
            builder.getStringAttr("bflang"), false,
            mlir::LLVM::DIEmissionKind::Full);

        auto i32Type = builder.getAttr<mlir::LLVM::DIBasicTypeAttr>(
            llvm::dwarf::DW_TAG_base_type, "int", 32,
            llvm::dwarf::DW_ATE_signed);
        auto subroutineType =
            builder.getAttr<mlir::LLVM::DISubroutineTypeAttr>(i32Type);

        getOperation()->walk([&](mlir::FunctionOpInterface func) {
            if (func.isExternal())
                return;
            auto name = func.getName();
            auto subProgramId =
                mlir::DistinctAttr::create(builder.getUnitAttr());
            auto subProgram = mlir::LLVM::DISubprogramAttr::get(
                subProgramId, diCu, diFile, name, "", diFile, 0, 0,
                mlir::LLVM::DISubprogramFlags::Definition, subroutineType);

            func->setLoc(builder.getFusedLoc(func->getLoc(), subProgram));
            func->walk([&](mlir::Operation *op) {
                if (op == func)
                    return;
                if (auto fileLoc =
                        mlir::dyn_cast<mlir::FileLineColLoc>(op->getLoc())) {
                    if (fileLoc.getLine() != 0) {
                        op->setLoc(
                            builder.getFusedLoc(op->getLoc(), subProgram));
                    }
                }
            });
        });

        auto getIntType = [&](uint32_t width) {
            std::string tyStr = llvm::formatv("uint{0}_t", width);
            auto ty = builder.getAttr<mlir::LLVM::DIBasicTypeAttr>(
                llvm::dwarf::DW_TAG_base_type, tyStr, width,
                llvm::dwarf::DW_ATE_unsigned);
            return ty;
        };

        getOperation()->walk([&](mlir::LLVM::GlobalOp globalOp) {
            auto name = globalOp.getName();
            auto ty = globalOp.getGlobalType();
            mlir::LLVM::DITypeAttr typeAttr;
            if (auto iTy = mlir::dyn_cast<mlir::IntegerType>(ty)) {
                typeAttr = getIntType(iTy.getIntOrFloatBitWidth());
            } else if (auto arrayType =
                           mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(ty)) {
                auto elmTy = mlir::dyn_cast<mlir::IntegerType>(
                    arrayType.getElementType());
                if (!elmTy)
                    return;
                auto diElmTy = getIntType(elmTy.getIntOrFloatBitWidth());
                auto totalBits = elmTy.getIntOrFloatBitWidth();
                totalBits *= arrayType.getNumElements();
                auto diRange = builder.getAttr<mlir::LLVM::DISubrangeAttr>(
                    builder.getI64IntegerAttr(totalBits), mlir::IntegerAttr{},
                    mlir::IntegerAttr{}, mlir::IntegerAttr{});
                typeAttr = mlir::LLVM::DICompositeTypeAttr::get(
                    builder.getContext(), llvm::dwarf::DW_TAG_array_type,
                    mlir::StringAttr{}, mlir::LLVM::DIFileAttr{}, 0,
                    mlir::LLVM::DIScopeAttr{}, diElmTy, mlir::LLVM::DIFlags(),
                    totalBits, 0, {});
            } else {
                return;
            }
            auto globalVar = mlir::LLVM::DIGlobalVariableAttr::get(
                builder.getContext(), diCu, builder.getStringAttr(name),
                mlir::StringAttr{}, diFile, 0, typeAttr, true, true, 0);
            auto globalVarExpr =
                mlir::LLVM::DIGlobalVariableExpressionAttr::get(
                    builder.getContext(), globalVar,
                    builder.getAttr<mlir::LLVM::DIExpressionAttr>());
            globalOp->setAttr("dbg_expr", globalVarExpr);
        });
    }
};
} // namespace

namespace mlir::bf {
std::unique_ptr<::mlir::Pass> createAttachDebugInfo() {
    return std::make_unique<AttachDebugInfo>();
}
} // namespace mlir::bf
