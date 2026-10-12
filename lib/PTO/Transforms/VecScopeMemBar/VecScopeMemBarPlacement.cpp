// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VecScopeMemBarPlacement.cpp ------------------------------------===//
//
// Solves barrier placement from the read-only analysis result. Picks a
// provably-mandatory anchor for each hazard relation, merges same-kind
// hazards at the same anchor, collapses multiple kinds in the same family at one anchor to
// that family's ALL barrier, and recognizes existing-barrier coverage. Outputs a deterministic
// plan; does not mutate the IR.
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/VecScopeMemBar/VecScopeMemBarPlacement.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <optional>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecscopemembar;

namespace {

// Hazards are identified by their stable index in the analysis result. Keep
// the complete set on every candidate barrier: a barrier at one anchor can
// cover several producer/consumer pairs of the same kind. Losing all but the
// last id here makes later redundancy elimination unsound because it compares
// incomplete hazard sets.
struct BarrierKindHazards {
  MemBarKind kind = MemBarKind::VV_ALL;
  SmallVector<unsigned, 4> hazards;
};

// An insertion anchor keyed by the op it precedes, for de-duplication.
struct AnchorKey {
  Operation *op;
  bool operator==(const AnchorKey &o) const { return op == o.op; }
};

struct AnchorKeyInfo {
  static AnchorKey getEmptyKey() { return {reinterpret_cast<Operation *>(1)}; }
  static AnchorKey getTombstoneKey() {
    return {reinterpret_cast<Operation *>(2)};
  }
  static unsigned getHashValue(const AnchorKey &a) {
    return llvm::hash_value(a.op);
  }
  static bool isEqual(const AnchorKey &a, const AnchorKey &b) { return a == b; }
};

static void appendUniqueHazard(SmallVectorImpl<unsigned> &hazards,
                               unsigned hazardId) {
  if (!llvm::is_contained(hazards, hazardId))
    hazards.push_back(hazardId);
}

static void sortUniqueHazards(SmallVectorImpl<unsigned> &hazards) {
  llvm::sort(hazards);
  hazards.erase(std::unique(hazards.begin(), hazards.end()), hazards.end());
}

// Return true when `covering` resolves every hazard resolved by `covered`.
// The vectors are normalized before this helper is called, so this is the
// same set-subset relation as std::includes used by the LLVM implementation.
static bool coversHazardSet(ArrayRef<unsigned> covering,
                            ArrayRef<unsigned> covered) {
  return std::includes(covering.begin(), covering.end(), covered.begin(),
                       covered.end());
}

// Enum values encode four families, each with an ALL and three directed kinds.
static bool barrierCovers(MemBarKind covering, MemBarKind covered) {
  const auto family = [](MemBarKind kind) {
    return static_cast<unsigned>(kind) / 4;
  };
  return covering == covered ||
         (family(covering) == family(covered) &&
          static_cast<unsigned>(covering) % 4 == 0);
}

static bool coveredByExisting(Operation *anchor, MemBarKind kind,
                              bool isLoopLatch, scf::ForOp carryingLoop,
                              ArrayRef<ExistingBarrier> existing) {
  Operation *prev = anchor->getPrevNode();
  if (!prev)
    return false;
  for (const auto &b : existing) {
    if (b.op != prev)
      continue;
    if (!barrierCovers(b.kind, kind))
      continue;
    if (isLoopLatch) {
      // Must be on the same carrying loop's backedge.
      if (b.latchLoop && carryingLoop) {
        // Match by op identity.
        if (b.op->getParentOfType<scf::ForOp>() != carryingLoop)
          continue;
      } else if (carryingLoop) {
        if (b.op->getParentOfType<scf::ForOp>() != carryingLoop)
          continue;
      }
    }
    return true;
  }
  return false;
}

static bool isProducerAccessForBarrier(MemBarKind kind, Operation *op) {
  switch (kind) {
  case MemBarKind::VLD_VST:
  case MemBarKind::VLD_ST:
    return isUBVectorLoad(op);
  case MemBarKind::VST_VLD:
  case MemBarKind::VST_VST:
  case MemBarKind::VST_LD:
  case MemBarKind::VST_ST:
    return isUBVectorStore(op);
  case MemBarKind::LD_VST:
  case MemBarKind::LD_ST:
    return isUBScalarLoad(op);
  case MemBarKind::ST_VLD:
  case MemBarKind::ST_VST:
  case MemBarKind::ST_LD:
  case MemBarKind::ST_ST:
    return isUBScalarStore(op);
  case MemBarKind::VV_ALL:
    return isUBVectorMemoryOp(op);
  case MemBarKind::VS_ALL:
    return isUBVectorMemoryOp(op);
  case MemBarKind::SV_ALL:
  case MemBarKind::SS_ALL:
    return isUBScalarMemoryOp(op);
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Transitive WAW redundancy helpers
//
// A VST_VST hazard store#1 -> store#2 (same iteration, same UB address) is
// redundant when the order is already guaranteed transitively: some load L
// reads store#1's address (RAW store#1 -> L, covered by a barrier) and L's
// result flows through pure-value ops into store#2's stored value. The SSA
// use-def chain then forces store#1 < L < store#2, so the explicit WAW cut
// adds nothing. Below: access-shape introspection and a reverse use-def walk
// that stops at UB writes (another store) so an unrelated store can never be
// mistaken for the relay.
//===----------------------------------------------------------------------===//

// The UB memory object a vector memory op reads or writes (the effect
// value, not the data vreg). Null for ops whose effects are not UB-backed.
static Value getUBMemoryObject(Operation *op) {
  auto iface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!iface)
    return nullptr;
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
  iface.getEffects(effects);
  for (const auto &effect : effects) {
    Value v = effect.getValue();
    if (!v || !isUBBackedType(v.getType()))
      continue;
    if (isa<MemoryEffects::Read>(effect.getEffect()) ||
        isa<MemoryEffects::Write>(effect.getEffect()))
      return v;
  }
  return nullptr;
}
} // namespace

static bool reverseReaches(Value sink, Value target, DenseSet<Value> &visited,
                           unsigned depth,
                           unsigned maxDepth) {
  if (sink == target)
    return true;
  if (depth > maxDepth)
    return false;
  if (!visited.insert(sink).second)
    return false;

  if (Operation *defOp = sink.getDefiningOp()) {
    if (auto iface = dyn_cast<MemoryEffectOpInterface>(defOp)) {
      SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> eff;
      iface.getEffects(eff);
      bool hasUBWrite = llvm::any_of(eff, [](const auto &e) {
        return isa<MemoryEffects::Write>(e.getEffect()) && e.getValue() &&
               vecscopemembar::isUBBackedType(e.getValue().getType());
      });
      if (hasUBWrite)
        return false; // relay through another store: not a value relay
    }
    // `sink` may be a result of an scf.for. Result i corresponds to region
    // iter-arg i, whose value at each iteration is the i-th operand of the
    // body's scf.yield. Relay through that operand so a dependence carried out
    // of the loop (e.g. a reduction result feeding a later store) is followed
    // back into the loop body, instead of stopping at the loop's input
    // operands (lb/ub/step/init) which never see the body's loads.
    if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
      unsigned k = 0;
      for (Value r : forOp.getResults()) {
        if (r == sink)
          break;
        ++k;
      }
      if (k < forOp.getNumRegionIterArgs()) {
        Value yielded =
            forOp.getBody()->getTerminator()->getOperand(k);
        if (reverseReaches(yielded, target, visited, depth + 1, maxDepth))
          return true;
      }
    }
    for (Value op : defOp->getOperands())
      if (reverseReaches(op, target, visited, depth + 1, maxDepth))
        return true;
    return false;
  }

  // Block argument: only scf.for iter_args are relayed, via the yield operand.
  auto ba = dyn_cast<BlockArgument>(sink);
  if (!ba)
    return false;
  auto forOp = ba.getOwner() ? dyn_cast<scf::ForOp>(ba.getOwner()->getParentOp())
                             : scf::ForOp();
  if (!forOp)
    return false;
  unsigned arg = ba.getArgNumber();
  // arg 0 is the loop IV; iter_args start at 1.
  if (arg == 0 || arg > forOp.getRegion().getNumArguments() - 1)
    return false;
  Value yielded = forOp.getBody()->getTerminator()->getOperand(arg - 1);
  return reverseReaches(yielded, target, visited, depth + 1, maxDepth);
}

bool vecscopemembar::valueDependsOn(Value sink, Value target) {
  DenseSet<Value> visited;
  return reverseReaches(sink, target, visited, 0, 32);
}

FailureOr<VecScopeMemBarPlan> vecscopemembar::solveVecScopeMemBarPlacement(
    const VecScopeMemBarAnalysisResult &result) {
  VecScopeMemBarPlan plan;
  const auto &A = result.accesses;

  //===--------------------------------------------------------------------===//
  // Transitive WAW redundancy elimination (same-iteration only).
  //
  // A VST_VST hazard store#1 -> store#2 is redundant when the order is
  // already enforced by an intermediate RAW store#1 -> load whose result
  // flows (through pure-value ops) into store#2's stored value, AND that
  // RAW is itself covered — either by an existing barrier between store#1
  // and the load, or by a scheduled VST_VLD hazard on the same pair. The
  // SSA use-def chain then fixes store#1 < load < store#2, so the explicit
  // WAW cut adds nothing. Loop-carried WAW is left untouched.
  //===--------------------------------------------------------------------===//
  DenseSet<unsigned> redundantHazards;
  for (const auto &h : result.hazards) {
    if (h.scope != VecScopeHazardScope::SameIteration)
      continue;
    if (h.kind != MemBarKind::VST_VST)
      continue;

    Operation *store1 = A[h.producer].op;
    Operation *store2 = A[h.consumer].op;
    Value obj1 = getUBMemoryObject(store1);
    SmallVector<Value, 2> storedVals2 = getStoredValues(store2);
    if (!obj1 || storedVals2.empty())
      continue; // unanalysable shapes: keep the barrier (conservative)

    unsigned lo = A[h.producer].lexicalOrder;
    unsigned hi = A[h.consumer].lexicalOrder;

    for (const auto &occ : A) {
      if (occ.kind != VecScopeAccessKind::Load)
        continue;
      if (occ.lexicalOrder <= lo || occ.lexicalOrder >= hi)
        continue; // load must sit strictly between the two stores

      // (a) load reads an address aliasing store#1's write.
      if (aliasSameIteration(A[h.producer].footprint, occ.footprint) ==
          VecScopeAliasResult::NoAlias)
        continue;

      // (b) store#2's stored value reaches back to this load's result.
      SmallVector<Value, 2> loadedVals = getLoadedValues(occ.op);
      if (loadedVals.empty())
        continue;
      bool reached = false;
      for (Value sv : storedVals2) {
        for (Value lv : loadedVals) {
          if (valueDependsOn(sv, lv)) {
            reached = true;
            break;
          }
        }
        if (reached)
          break;
      }
      if (!reached)
        continue;

      // (c) the RAW store#1 -> load must be covered, otherwise the relay
      // is not ordered. Covered = an existing VST_VLD/VV_ALL barrier at the
      // load's anchor, OR a scheduled same-iteration VST_VLD hazard on the
      // same (producer, consumer) pair (it will place a barrier itself).
      Operation *rawAnchor =
          findSameIterationAnchor(store1, occ.op, result.scope);
      if (!rawAnchor)
        rawAnchor = occ.op;
      bool covered = coveredByExisting(rawAnchor, MemBarKind::VST_VLD, false,
                                       nullptr, result.existingBarriers);
      if (!covered) {
        for (const auto &rh : result.hazards) {
          if (rh.scope == VecScopeHazardScope::SameIteration &&
              rh.kind == MemBarKind::VST_VLD &&
              rh.producer == h.producer && rh.consumer == occ.lexicalOrder) {
            covered = true;
            break;
          }
        }
      }
      if (!covered)
        continue;

      redundantHazards.insert(h.id);
      break;
    }
  }

  // Generalize the WAW proof to multi-hop scalar/vector relays. A store can
  // be ordered before a later store through store-to-load barriers and SSA
  // value edges, even when the relay contains scalar stores and loads.
  for (const auto &candidate : result.hazards) {
    if (candidate.scope != VecScopeHazardScope::SameIteration ||
        candidate.kind != MemBarKind::VST_VST ||
        redundantHazards.contains(candidate.id))
      continue;
    const unsigned count = A.size();
    SmallVector<SmallVector<unsigned, 4>, 16> edges(count);
    auto addEdge = [&](unsigned from, unsigned to) {
      if (!llvm::is_contained(edges[from], to))
        edges[from].push_back(to);
    };
    for (const auto &h : result.hazards) {
      if (h.id != candidate.id &&
          h.scope == VecScopeHazardScope::SameIteration)
        addEdge(h.producer, h.consumer);
    }
    for (unsigned load = 0; load < count; ++load) {
      if (A[load].kind != VecScopeAccessKind::Load)
        continue;
      auto loaded = getLoadedValues(A[load].op);
      for (unsigned store = load + 1; store < count; ++store) {
        if (A[store].kind != VecScopeAccessKind::Store)
          continue;
        auto stored = getStoredValues(A[store].op);
        bool depends = false;
        for (Value sv : stored)
          for (Value lv : loaded)
            depends |= valueDependsOn(sv, lv);
        if (depends)
          addEdge(load, store);
      }
    }
    SmallVector<bool, 16> reachable(count, false);
    SmallVector<unsigned, 16> worklist;
    reachable[candidate.producer] = true;
    worklist.push_back(candidate.producer);
    while (!worklist.empty()) {
      unsigned current = worklist.pop_back_val();
      for (unsigned next : edges[current]) {
        if (!reachable[next]) {
          reachable[next] = true;
          worklist.push_back(next);
        }
      }
    }
    if (reachable[candidate.consumer])
      redundantHazards.insert(candidate.id);
  }

  // Merge kinds within each family; retain separate cross-family barriers.
  DenseMap<AnchorKey, SmallVector<BarrierKindHazards>, AnchorKeyInfo>
      anchorKinds;
  // Anchors that are genuine loop-latch cuts (carrying-loop terminator),
  // distinguished from a plain last-op-in-vecscope anchor: vecscope blocks are
  // NoTerminator, so their last op is `getBlock()->getTerminator()` without
  // being a latch. Latch semantics differ (per-iteration backedge), so the
  // redundant-barrier pass must not treat such anchors as latches.
  DenseSet<AnchorKey, AnchorKeyInfo> latchAnchors;
  SmallVector<Operation *, 8> anchorOrder;

  auto addPlacement = [&](Operation *anchor, MemBarKind kind, unsigned hazardId,
                          bool isLoopLatch = false,
                          scf::ForOp carryingLoop = nullptr) {
    if (coveredByExisting(anchor, kind, isLoopLatch, carryingLoop,
                          result.existingBarriers))
      return;
    auto &kinds = anchorKinds[{anchor}];
    if (kinds.empty()) {
      anchorOrder.push_back(anchor);
      if (isLoopLatch)
        latchAnchors.insert({anchor});
    }

    // A different directed barrier already immediately before this anchor is
    // part of the required cut as well. Normalize it together with the newly
    // discovered kind within its family.
    Operation *prev = anchor->getPrevNode();
    for (const auto &existing : result.existingBarriers) {
      if (existing.op != prev || existing.kind == kind ||
          barrierCovers(existing.kind, kind))
        continue;
      if (isLoopLatch && carryingLoop &&
          existing.op->getParentOfType<scf::ForOp>() != carryingLoop)
        continue;
      bool foundExisting = false;
      for (const auto &existingKind : kinds)
        if (existingKind.kind == existing.kind)
          foundExisting = true;
      if (!foundExisting)
        kinds.push_back({existing.kind, {hazardId}});
      break;
    }
    for (auto &entry : kinds)
      if (entry.kind == kind) {
        appendUniqueHazard(entry.hazards, hazardId);
        return;
      }
    kinds.push_back({kind, {hazardId}});
  };

  for (const auto &h : result.hazards) {
    if (redundantHazards.count(h.id))
      continue; // transitive WAW redundancy: order already guaranteed
    Operation *anchor = nullptr;
    bool isLatch = false;
    scf::ForOp carrying = nullptr;
    if (h.scope == VecScopeHazardScope::SameIteration) {
      // Insert at the mandatory consumer-side cut. For cross-hierarchy pairs
      // this is the enclosing consumer loop, not an op inside its body.
      anchor = h.sameIterationAnchor ? h.sameIterationAnchor : A[h.consumer].op;
    } else {
      // Loop-carried: carrying loop terminator.
      if (!h.carryingLoop)
        continue;
      // Resolve the carrying loop op.
      for (const auto &l : result.loops)
        if (l.id == *h.carryingLoop) {
          carrying = l.op;
          break;
        }
      if (!carrying)
        continue;
      anchor = carrying.getBody()->getTerminator();
      isLatch = true;
    }
    if (!anchor)
      continue;
    addPlacement(anchor, h.kind, h.id, isLatch, carrying);
  }

  // DenseMap iteration is intentionally not used for scheduling: anchors can
  // belong to different blocks, so isBeforeInBlock would assert. The first
  // hazard that discovers an anchor is already deterministic lexical order.
  for (Operation *anchor : anchorOrder) {
    auto it = anchorKinds.find({anchor});
    if (it == anchorKinds.end())
      continue;
    auto &kinds = it->second;
    for (unsigned family = 0; family < 4; ++family) {
      BarrierPlacement bp;
      bp.anchor = anchor;
      unsigned kindCount = 0;
      for (const auto &entry : kinds) {
        if (static_cast<unsigned>(entry.kind) / 4 != family) {
          continue;
        }
        bp.kind = entry.kind;
        ++kindCount;
        for (unsigned hazardId : entry.hazards) {
          appendUniqueHazard(bp.resolvedHazards, hazardId);
        }
      }
      if (kindCount == 0) {
        continue;
      }
      if (kindCount > 1) {
        bp.kind = static_cast<MemBarKind>(family * 4);
      }
      bp.anchorKind = latchAnchors.count({anchor})
                          ? BarrierAnchorKind::BeforeLoopTerminator
                          : BarrierAnchorKind::BeforeOperation;
      sortUniqueHazards(bp.resolvedHazards);
      plan.barriers.push_back(bp);
    }
  }

  // Normalize each candidate's hazard set before comparing candidates. The
  // set contains only hazards from the candidate barrier's family.
  for (auto &barrier : plan.barriers)
    sortUniqueHazards(barrier.resolvedHazards);

  // Redundant-barrier elimination by hazard coverage
  SmallVector<bool> live(plan.barriers.size(), true);
  auto canCompareCoverage = [&](unsigned covering, unsigned covered) {
    if (!live[covering] || !live[covered] || covering == covered)
      return false;
    const auto &lhs = plan.barriers[covering];
    const auto &rhs = plan.barriers[covered];
    if (lhs.anchorKind == BarrierAnchorKind::BeforeLoopTerminator ||
        rhs.anchorKind == BarrierAnchorKind::BeforeLoopTerminator)
      return false;
    // An ALL barrier covers directed hazards only in its own family.
    // A directed barrier covers the same kind; comparing the
    // resolved hazard sets alone is not sufficient to establish this.
    if (!barrierCovers(lhs.kind, rhs.kind))
      return false;
    if (!lhs.anchor || !rhs.anchor ||
        lhs.anchor->getBlock() != rhs.anchor->getBlock())
      return false;
    return lhs.anchor->isBeforeInBlock(rhs.anchor);
  };

  // First implement the requested subset rule. Since candidates are ordered
  // lexically, only an earlier covering cut can subsume a later cut; an
  // earlier cut also has the required execution ordering semantics.
  for (unsigned i = 0; i < plan.barriers.size(); ++i) {
    if (!live[i])
      continue;
    for (unsigned j = 0; j < plan.barriers.size(); ++j) {
      if (!canCompareCoverage(j, i))
        continue;
      if (coversHazardSet(plan.barriers[j].resolvedHazards,
                          plan.barriers[i].resolvedHazards)) {
        live[i] = false;
        break;
      }
    }
  }

  // Then implement the requested shared-hazard rule. Count only live
  // candidates, and remove a candidate only when every hazard it resolves has
  // another live resolver. Recompute until stable so the count is exact after
  // each removal.
  bool changed = true;
  while (changed) {
    changed = false;
    DenseMap<unsigned, unsigned> pairMemBarNum;
    for (unsigned i = 0; i < plan.barriers.size(); ++i) {
      if (!live[i])
        continue;
      for (unsigned hazardId : plan.barriers[i].resolvedHazards)
        ++pairMemBarNum[hazardId];
    }
    for (unsigned i = 0; i < plan.barriers.size(); ++i) {
      if (!live[i] || plan.barriers[i].resolvedHazards.empty())
        continue;
      bool shared = true;
      for (unsigned hazardId : plan.barriers[i].resolvedHazards) {
        auto count = pairMemBarNum.find(hazardId);
        if (count == pairMemBarNum.end() || count->second <= 1) {
          shared = false;
          break;
        }
      }
      if (!shared)
        continue;
      live[i] = false;
      changed = true;
      for (unsigned hazardId : plan.barriers[i].resolvedHazards) {
        auto count = pairMemBarNum.find(hazardId);
        if (count != pairMemBarNum.end())
          --count->second;
      }
    }
  }

  {
    SmallVector<BarrierPlacement, 8> kept;
    kept.reserve(plan.barriers.size());
    for (unsigned i = 0; i < plan.barriers.size(); ++i)
      if (live[i])
        kept.push_back(plan.barriers[i]);
    plan.barriers = std::move(kept);
  }

  DenseSet<unsigned> redundantBarriers;
  for (unsigned i = 0; i < plan.barriers.size(); ++i) {
    const auto &b2 = plan.barriers[i];
    if (b2.anchorKind == BarrierAnchorKind::BeforeLoopTerminator)
      continue;
    for (unsigned j = 0; j < i; ++j) {
      if (redundantBarriers.count(j))
        continue;
      const auto &b1 = plan.barriers[j];
      if (b1.kind != b2.kind)
        continue;
      if (b1.anchorKind == BarrierAnchorKind::BeforeLoopTerminator)
        continue;
      Block *block = b1.anchor->getBlock();
      if (block != b2.anchor->getBlock())
        continue;
      if (!b1.anchor->isBeforeInBlock(b2.anchor))
        continue;
      bool breaksCoverage = false;
      for (Operation *cur = b1.anchor;
           cur && cur != b2.anchor && cur->getBlock() == block;
           cur = cur->getNextNode()) {
        cur->walk([&](Operation *o) {
          if (breaksCoverage)
            return;
          if (isProducerAccessForBarrier(b1.kind, o))
            breaksCoverage = true;
        });
        if (breaksCoverage)
          break;
      }
      if (!breaksCoverage) {
        redundantBarriers.insert(i);
        break;
      }
    }
  }
  if (!redundantBarriers.empty()) {
    SmallVector<BarrierPlacement, 8> kept;
    kept.reserve(plan.barriers.size() - redundantBarriers.size());
    for (unsigned i = 0; i < plan.barriers.size(); ++i)
      if (!redundantBarriers.count(i))
        kept.push_back(plan.barriers[i]);
    plan.barriers = std::move(kept);
  }

  return plan;
}
