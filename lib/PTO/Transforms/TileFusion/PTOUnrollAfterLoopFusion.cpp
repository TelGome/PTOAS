// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOUnrollAfterLoopFusion.cpp ---------------------------------------===//
//
// Partial-unroll scf.for inside pto.fusion_region after PTOLowLevelLoopFusion.
//
// The unroll factor is annotated by the external cost model on the enclosing
// `pto.fusion_region` as two i64 attributes:
//   * `pto.fusion.row_unroll_factor`  (intended for the outer / row layer)
//   * `pto.fusion.col_unroll_factor`  (intended for the inner / col layer)
// Exactly which layer gets a factor > 1 is the cost model's call; this pass
// only consumes the values. Keeping both attributes (instead of collapsing to
// one) preserves the cost model's intent on the IR, so `-debug` output can be
// cross-referenced against the cost model's prediction when something misfires.
//
// Hard constraints:
//   * Only the innermost (leaf) scf.for is unrolled. Unrolling an outer loop
//     would duplicate its child scf.for, violating the "carrier body has at
//     most one child scf.for" invariant that the downstream
//     `PTOFusionLoadStoreElision::getLeafLoopBody` relies on.
//   * The constant trip count must be divisible by the factor (no epilogue
//     tail loop) -- otherwise the loop is left untouched. This keeps the same
//     single-child-loop invariant intact for the second LoadStoreElision run.
//   * Re-unroll of the same loop within one greedy sweep is prevented by
//     consuming the used factor: the attribute (col_unroll_factor /
//     row_unroll_factor) is reset to 1 on the fusion_region after a successful
//     unroll. The cost model is single-dimensional (it annotates at most one of
//     the two factors with a value > 1), so once consumed both factors are <= 1
//     and the leaf no longer matches. This is sufficient because the cost model
//     never emits both factors > 1; no extra loop-side marker is needed.
//
// Non-fusion kernels carry no `pto.fusion_region`, so this pass is a natural
// no-op on them. Failures are skipped with LLVM_DEBUG; the pass never signals
// failure.
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/Passes.h"

#include "PTO/IR/PTO.h" // FusionRegionOp
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h" // loopUnrollByFactor
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h" // getConstantIntValue
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#include <cstdint>
#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOUNROLLAFTERLOOPFUSION
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "pto-unroll-after-loop-fusion"

static constexpr llvm::StringLiteral kRowUnrollFactorAttr =
    "pto.fusion.row_unroll_factor";
static constexpr llvm::StringLiteral kColUnrollFactorAttr =
    "pto.fusion.col_unroll_factor";

namespace {

/// Read an i64 unroll factor from `region`; return the value only if it is
/// present and strictly greater than 1, else 0.
static int64_t getEffectiveFactor(pto::FusionRegionOp region,
                                  llvm::StringRef attrName) {
  if (auto attr = region->getAttrOfType<IntegerAttr>(attrName)) {
    int64_t v = attr.getInt();
    if (v > 1)
      return v;
  }
  return 0;
}

/// Whether `forOp` has a child scf.for in its body (i.e. it is NOT the
/// innermost loop of the nest).
static bool hasChildForOp(scf::ForOp forOp) {
  bool hasChild = false;
  forOp.getBody()->walk([&](scf::ForOp) {
    hasChild = true;
    return WalkResult::interrupt();
  });
  return hasChild;
}

/// Constant trip count of a half-open scf.for (`lb <= i < ub`, `step > 0`),
/// or std::nullopt if any bound/step is not a compile-time constant.
static std::optional<int64_t> getConstantTripCount(scf::ForOp forOp) {
  std::optional<int64_t> lb = getConstantIntValue(forOp.getLowerBound());
  std::optional<int64_t> ub = getConstantIntValue(forOp.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
  if (!lb || !ub || !step || *step <= 0 || *ub <= *lb)
    return std::nullopt;
  return (*ub - *lb + *step - 1) / *step; // ceilDiv
}

struct UnrollAfterFusionPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // Scope gate: only loops inside a fusion_region.
    auto region = forOp->getParentOfType<pto::FusionRegionOp>();
    if (!region) {
      LLVM_DEBUG(llvm::dbgs() << "PTOUnrollAfterLoopFusion: skip non-fusion "
                 << "scf.for at " << forOp.getLoc() << "\n");
      return failure();
    }

    // Cost model's intent (both factors), for cross-referencing in -debug.
    int64_t rowF = getEffectiveFactor(region, kRowUnrollFactorAttr);
    int64_t colF = getEffectiveFactor(region, kColUnrollFactorAttr);

    // Only unroll the innermost (leaf) loop. Unrolling an outer loop would
    // duplicate its child scf.for and break the single-child-loop invariant
    // that PTOFusionLoadStoreElision::getLeafLoopBody relies on.
    if (hasChildForOp(forOp)) {
      LLVM_DEBUG(llvm::dbgs() << "PTOUnrollAfterLoopFusion: skip non-leaf "
                 << "(outer) scf.for at " << forOp.getLoc()
                 << " -- unrolling outer breaks LoadStoreElision;"
                 << " row_f=" << rowF << " col_f=" << colF
                 << " (cost-model intent may target this layer but pass "
                 << "defers to the leaf)\n");
      return failure();
    }

    // Leaf takes whichever factor is > 1 (col preferred; in a two-layer nest
    // the leaf is the col loop). The cost model may place the > 1 value on
    // either attribute depending on which layer is the effective innermost
    // (e.g. col trip == 1 -> col gets folded away -> row becomes leaf -> the
    // > 1 value lives on row_f). Both > 1 is a legal input; col wins.
    int64_t factor;
    llvm::StringRef src;
    if (colF > 1) {
      factor = colF;
      src = "col";
    } else if (rowF > 1) {
      factor = rowF;
      src = "row";
    } else {
      LLVM_DEBUG(llvm::dbgs() << "PTOUnrollAfterLoopFusion: skip no factor>1 "
                 << "scf.for at " << forOp.getLoc()
                 << " row_f=" << rowF << " col_f=" << colF << "\n");
      return failure();
    }

    // Divisibility gate: constant trip count must be divisible by the factor
    // (no epilogue tail loop). Non-constant / non-divisible -> leave untouched.
    auto trip = getConstantTripCount(forOp);
    if (!trip) {
      LLVM_DEBUG(llvm::dbgs() << "PTOUnrollAfterLoopFusion: skip non-constant "
                 << "trip scf.for at " << forOp.getLoc()
                 << " factor(from " << src << ")=" << factor << "\n");
      return failure();
    }
    if (*trip % factor != 0) {
      LLVM_DEBUG(llvm::dbgs() << "PTOUnrollAfterLoopFusion: skip indivisible "
                 << "trip scf.for at " << forOp.getLoc()
                 << " trip=" << *trip << " factor=" << factor << "(from "
                 << src << ")\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "PTOUnrollAfterLoopFusion: unroll scf.for at "
               << forOp.getLoc() << " factor=" << factor << "(from " << src
               << ") trip=" << *trip
               << " row_f=" << rowF << " col_f=" << colF << "\n");

    // loopUnrollByFactor rewrites the loop in place (step scaled, body copied)
    // and may fully promote it away when trip == factor (the original forOp is
    // erased by promoteIfSingleIteration). It returns FailureOr<UnrolledLoopInfo>;
    // on success we never touch `forOp` again -- the divisibility gate above
    // guarantees no epilogue, so only mainLoopOp (which is nullopt when the loop
    // was fully promoted) could have survived. Consume the factor on the region
    // (which is never erased here) so the greedy driver does not re-apply it to
    // a sibling/outer loop that becomes the new leaf after this one is gone.
    auto unrolled =
        loopUnrollByFactor(forOp, static_cast<uint64_t>(factor));
    if (failed(unrolled)) {
      LLVM_DEBUG(llvm::dbgs() << "  loopUnrollByFactor failed "
                 << "(iter_args live-out?) at " << forOp.getLoc() << "\n");
      return failure();
    }
    (void)unrolled; // mainLoopOp/epilogueLoopOp unused: divisibility => no tail.

    // Consume the factor we just used: reset it to 1 on the fusion_region so a
    // sibling/outer loop that becomes the new leaf (after this loop is fully
    // unrolled and erased) does not re-use the same factor. The cost model's
    // factor is dimension-specific (col_unroll_factor / row_unroll_factor); we
    // picked it for *this* leaf's dimension, so consuming it prevents it from
    // being applied to a different dimension's leaf later in the same greedy
    // sweep. Without this, e.g. unrolling the col leaf (trip==factor, erased)
    // would leave col_unroll_factor>1 on the region, and the greedy driver
    // would then re-apply it to the row leaf that the col disappearance exposed.
    llvm::StringRef consumedAttr =
        src == "col" ? kColUnrollFactorAttr : kRowUnrollFactorAttr;
    region->setAttr(consumedAttr,
                    rewriter.getI64IntegerAttr(/*value=*/1));
    return success();
  }
};

struct PTOUnrollAfterLoopFusion
    : public pto::impl::PTOUnrollAfterLoopFusionBase<
          PTOUnrollAfterLoopFusion> {
  using pto::impl::PTOUnrollAfterLoopFusionBase<
      PTOUnrollAfterLoopFusion>::PTOUnrollAfterLoopFusionBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<UnrollAfterFusionPattern>(ctx);

    GreedyRewriteConfig config;
    config.setMaxIterations(10); // loops may nest
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);

    if (failed(applyPatternsGreedily(func, std::move(patterns), config)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOUnrollAfterLoopFusionPass() {
  return std::make_unique<PTOUnrollAfterLoopFusion>();
}
