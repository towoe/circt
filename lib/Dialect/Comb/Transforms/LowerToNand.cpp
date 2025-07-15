//===- LowerToNand.cpp - Lower And/Or ops to NAND operations --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::comb;

namespace circt {
namespace comb {
#define GEN_PASS_DEF_LOWERTONAND
#include "circt/Dialect/Comb/Passes.h.inc"
} // namespace comb
} // namespace circt

namespace {
/// Convert And operations to NAND + NOT operations
struct AndToNandPattern : public OpRewritePattern<AndOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AndOp op,
                                PatternRewriter &rewriter) const override {
    // Lower and(a, b, c) -> nand(nand(a, b, c), nand(a, b, c))
    // This implements: ~(~(a & b & c)) = a & b & c

    // Create NAND operation with same inputs
    auto nandOp = rewriter.create<NAndOp>(op.getLoc(), op.getType(),
                                          op.getInputs(), op.getTwoState());

    // Create double NAND (NAND of the result with itself = NOT)
    auto doubleNand =
        rewriter.create<NAndOp>(op.getLoc(), op.getType(),
                                ValueRange{nandOp, nandOp}, op.getTwoState());

    rewriter.replaceOp(op, doubleNand);
    return success();
  }
};

/// Convert Or operations to NAND operations using De Morgan's law
/// or(a, b) = nand(not(a), not(b))
struct OrToNandPattern : public OpRewritePattern<OrOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(OrOp op,
                                PatternRewriter &rewriter) const override {
    // Apply De Morgan's law: or(a, b, c) = nand(nand(a, a), nand(b, b), nand(c,
    // c)) This implements: ~(~a & ~b & ~c) = a | b | c

    // Create NOT of each input using NAND with itself
    SmallVector<Value> notInputs;
    for (auto input : op.getInputs()) {
      auto notInput =
          rewriter.create<NAndOp>(op.getLoc(), input.getType(),
                                  ValueRange{input, input}, op.getTwoState());
      notInputs.push_back(notInput);
    }

    // Apply NAND to all the inverted inputs
    auto nandResult = rewriter.create<NAndOp>(op.getLoc(), op.getType(),
                                              notInputs, op.getTwoState());

    rewriter.replaceOp(op, nandResult);
    return success();
  }
};
} // namespace

namespace {
class LowerToNandPass : public impl::LowerToNandBase<LowerToNandPass> {
public:
  using LowerToNandBase::LowerToNandBase;

  void runOnOperation() override;
};
} // namespace
  //

void LowerToNandPass::runOnOperation() {
  auto *context = &getContext();
  RewritePatternSet patterns(context);

  patterns.add<AndToNandPattern, OrToNandPattern>(context);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}
