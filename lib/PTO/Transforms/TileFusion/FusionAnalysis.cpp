// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/TileFusion/FusionAnalysis.h"

#include "PTO/IR/PTO.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace pto {

namespace {

static int64_t getConstantIndexOrDynamic(Value value) {
  if (!value)
    return ShapedType::kDynamic;
  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  if (auto cst = value.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  return ShapedType::kDynamic;
}

static SmallVector<int64_t, 4> getValidShapeVec(Type type) {
  if (auto tileType = dyn_cast<pto::TileBufType>(type)) {
    return SmallVector<int64_t, 4>(tileType.getValidShape().begin(),
                                   tileType.getValidShape().end());
  }
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return SmallVector<int64_t, 4>(shapedType.getShape().begin(),
                                   shapedType.getShape().end());
  }
  return {};
}

static constexpr unsigned kInvalidShapeDim = ~0u;

struct ShapeValueDims {
  unsigned rows = kInvalidShapeDim;
  unsigned cols = kInvalidShapeDim;

  bool isValid() const {
    return rows != kInvalidShapeDim && cols != kInvalidShapeDim;
  }
};

class ShapeConstraintSolver {
public:
  unsigned createDim() {
    unsigned id = parent.size();
    parent.push_back(id);
    rank.push_back(0);
    constants.push_back(std::nullopt);
    conflicts.push_back(false);
    return id;
  }

  unsigned find(unsigned dim) {
    assert(dim < parent.size() && "shape dim out of range");
    if (parent[dim] == dim)
      return dim;
    parent[dim] = find(parent[dim]);
    return parent[dim];
  }

  void merge(unsigned lhs, unsigned rhs) {
    if (lhs == kInvalidShapeDim || rhs == kInvalidShapeDim)
      return;

    unsigned lhsRoot = find(lhs);
    unsigned rhsRoot = find(rhs);
    if (lhsRoot == rhsRoot)
      return;

    if (rank[lhsRoot] < rank[rhsRoot])
      std::swap(lhsRoot, rhsRoot);
    parent[rhsRoot] = lhsRoot;
    if (rank[lhsRoot] == rank[rhsRoot])
      ++rank[lhsRoot];

    conflicts[lhsRoot] = conflicts[lhsRoot] || conflicts[rhsRoot];
    if (constants[lhsRoot] && constants[rhsRoot] &&
        *constants[lhsRoot] != *constants[rhsRoot])
      conflicts[lhsRoot] = true;
    else if (!constants[lhsRoot])
      constants[lhsRoot] = constants[rhsRoot];
  }

  void bindConstant(unsigned dim, int64_t value) {
    if (dim == kInvalidShapeDim || value == ShapedType::kDynamic)
      return;

    unsigned root = find(dim);
    if (constants[root] && *constants[root] != value)
      conflicts[root] = true;
    else
      constants[root] = value;
  }

  bool hasConflict(unsigned dim) {
    if (dim == kInvalidShapeDim)
      return true;
    return conflicts[find(dim)];
  }

  std::optional<int64_t> getConstant(unsigned dim) {
    if (dim == kInvalidShapeDim)
      return std::nullopt;
    return constants[find(dim)];
  }

private:
  SmallVector<unsigned, 32> parent;
  SmallVector<unsigned, 32> rank;
  SmallVector<std::optional<int64_t>, 32> constants;
  SmallVector<bool, 32> conflicts;
};

static void bindDimToValue(ShapeConstraintSolver &solver,
                           DenseMap<Value, unsigned> &symbolDimByValue,
                           unsigned dim, Value value) {
  if (!value || dim == kInvalidShapeDim)
    return;

  int64_t constant = getConstantIndexOrDynamic(value);
  if (constant != ShapedType::kDynamic) {
    solver.bindConstant(dim, constant);
    return;
  }

  auto [it, inserted] = symbolDimByValue.try_emplace(value, kInvalidShapeDim);
  if (inserted)
    it->second = solver.createDim();
  solver.merge(dim, it->second);
}

static void bindExplicitValidDims(ShapeConstraintSolver &solver,
                                  DenseMap<Value, unsigned> &symbolDimByValue,
                                  Value value, ShapeValueDims dims) {
  if (auto alloc = value.getDefiningOp<pto::AllocTileOp>()) {
    bindDimToValue(solver, symbolDimByValue, dims.rows, alloc.getValidRow());
    bindDimToValue(solver, symbolDimByValue, dims.cols, alloc.getValidCol());
    return;
  }
  if (auto bind = value.getDefiningOp<pto::BindTileOp>()) {
    bindDimToValue(solver, symbolDimByValue, dims.rows, bind.getValidRow());
    bindDimToValue(solver, symbolDimByValue, dims.cols, bind.getValidCol());
    return;
  }
  if (auto materialize = value.getDefiningOp<pto::MaterializeTileOp>()) {
    bindDimToValue(solver, symbolDimByValue, dims.rows,
                   materialize.getValidRow());
    bindDimToValue(solver, symbolDimByValue, dims.cols,
                   materialize.getValidCol());
    return;
  }
  if (auto subview = value.getDefiningOp<pto::SubViewOp>()) {
    bindDimToValue(solver, symbolDimByValue, dims.rows, subview.getValidRow());
    bindDimToValue(solver, symbolDimByValue, dims.cols, subview.getValidCol());
    return;
  }
}

static ShapeValueDims getOrCreateValueDims(
    ShapeConstraintSolver &solver, DenseMap<Value, ShapeValueDims> &dimsByValue,
    DenseMap<Value, unsigned> &symbolDimByValue, Value value) {
  auto existing = dimsByValue.find(value);
  if (existing != dimsByValue.end())
    return existing->second;

  ShapeValueDims dims;
  SmallVector<int64_t, 4> validShape = getValidShapeVec(value.getType());
  if (validShape.size() >= 2) {
    dims.rows = solver.createDim();
    dims.cols = solver.createDim();
    if (!ShapedType::isDynamic(validShape[0]))
      solver.bindConstant(dims.rows, validShape[0]);
    if (!ShapedType::isDynamic(validShape[1]))
      solver.bindConstant(dims.cols, validShape[1]);
    bindExplicitValidDims(solver, symbolDimByValue, value, dims);
  }

  dimsByValue.try_emplace(value, dims);
  return dims;
}

static ShapeValueDims getValueDims(
    ShapeConstraintSolver &solver, DenseMap<Value, ShapeValueDims> &dimsByValue,
    DenseMap<Value, unsigned> &symbolDimByValue, Value value) {
  return getOrCreateValueDims(solver, dimsByValue, symbolDimByValue, value);
}

static void mergeRows(ShapeConstraintSolver &solver, ShapeValueDims lhs,
                      ShapeValueDims rhs) {
  solver.merge(lhs.rows, rhs.rows);
}

static void mergeCols(ShapeConstraintSolver &solver, ShapeValueDims lhs,
                      ShapeValueDims rhs) {
  solver.merge(lhs.cols, rhs.cols);
}

static void mergeShapes(ShapeConstraintSolver &solver, ShapeValueDims lhs,
                        ShapeValueDims rhs) {
  mergeRows(solver, lhs, rhs);
  mergeCols(solver, lhs, rhs);
}

static void mergeAllShapes(
    ShapeConstraintSolver &solver, DenseMap<Value, ShapeValueDims> &dimsByValue,
    DenseMap<Value, unsigned> &symbolDimByValue, ArrayRef<Value> values) {
  if (values.empty())
    return;
  ShapeValueDims anchor =
      getValueDims(solver, dimsByValue, symbolDimByValue, values.front());
  for (Value value : values.drop_front())
    mergeShapes(solver, anchor,
                getValueDims(solver, dimsByValue, symbolDimByValue, value));
}

static void applyShapeConstraintsForNode(
    ShapeConstraintSolver &solver, DenseMap<Value, ShapeValueDims> &dimsByValue,
    DenseMap<Value, unsigned> &symbolDimByValue,
    const FusionComputeNode &node) {
  const FusionOpSemantics &semantics = node.semantics;
  switch (semantics.computeFamily) {
  case FusionComputeFamily::Elementwise: {
    SmallVector<Value, 6> values;
    values.append(semantics.tileInputs.begin(), semantics.tileInputs.end());
    values.append(semantics.tileOutputs.begin(), semantics.tileOutputs.end());
    mergeAllShapes(solver, dimsByValue, symbolDimByValue, values);
    return;
  }
  case FusionComputeFamily::ScalarExpand:
    mergeAllShapes(solver, dimsByValue, symbolDimByValue,
                   semantics.tileOutputs);
    return;
  case FusionComputeFamily::RowBroadcastBinary: {
    if (semantics.tileOutputs.empty())
      return;
    ShapeValueDims output = getValueDims(solver, dimsByValue, symbolDimByValue,
                                         semantics.tileOutputs.front());
    if (!semantics.tileInputs.empty())
      mergeShapes(solver,
                  getValueDims(solver, dimsByValue, symbolDimByValue,
                               semantics.tileInputs[0]),
                  output);
    if (semantics.tileInputs.size() >= 2) {
      ShapeValueDims rowInput = getValueDims(
          solver, dimsByValue, symbolDimByValue, semantics.tileInputs[1]);
      mergeRows(solver, rowInput, output);
      solver.bindConstant(rowInput.cols, 1);
    }
    for (Value extraOutput : ArrayRef<Value>(semantics.tileOutputs).drop_front())
      mergeShapes(solver, output,
                  getValueDims(solver, dimsByValue, symbolDimByValue,
                               extraOutput));
    return;
  }
  case FusionComputeFamily::ReduceRow:
  case FusionComputeFamily::ReduceCol: {
    mergeAllShapes(solver, dimsByValue, symbolDimByValue,
                   semantics.tileInputs);
    if (semantics.tileInputs.empty() || semantics.tileOutputs.empty())
      return;
    ShapeValueDims input = getValueDims(solver, dimsByValue, symbolDimByValue,
                                        semantics.tileInputs.front());
    ShapeValueDims output = getValueDims(solver, dimsByValue, symbolDimByValue,
                                         semantics.tileOutputs.front());
    if (semantics.computeFamily == FusionComputeFamily::ReduceRow) {
      mergeRows(solver, input, output);
      solver.bindConstant(output.cols, 1);
    } else {
      solver.bindConstant(output.rows, 1);
      mergeCols(solver, input, output);
    }
    for (Value extraOutput : ArrayRef<Value>(semantics.tileOutputs).drop_front())
      mergeShapes(solver, output,
                  getValueDims(solver, dimsByValue, symbolDimByValue,
                               extraOutput));
    return;
  }
  case FusionComputeFamily::Unknown:
    return;
  }
}

static ShapeValueDims getIterationDomainDimsForNode(
    ShapeConstraintSolver &solver, DenseMap<Value, ShapeValueDims> &dimsByValue,
    DenseMap<Value, unsigned> &symbolDimByValue,
    const FusionComputeNode &node) {
  const FusionOpSemantics &semantics = node.semantics;
  switch (semantics.computeFamily) {
  case FusionComputeFamily::Elementwise:
  case FusionComputeFamily::ScalarExpand:
  case FusionComputeFamily::RowBroadcastBinary:
    if (!semantics.tileOutputs.empty())
      return getValueDims(solver, dimsByValue, symbolDimByValue,
                          semantics.tileOutputs.front());
    if (!semantics.tileInputs.empty())
      return getValueDims(solver, dimsByValue, symbolDimByValue,
                          semantics.tileInputs.front());
    break;
  case FusionComputeFamily::ReduceRow:
  case FusionComputeFamily::ReduceCol:
    if (!semantics.tileInputs.empty())
      return getValueDims(solver, dimsByValue, symbolDimByValue,
                          semantics.tileInputs.front());
    break;
  case FusionComputeFamily::Unknown:
    break;
  }
  return ShapeValueDims();
}

static IterationDomainInfo
buildIterationDomainInfo(ShapeConstraintSolver &solver, ShapeValueDims dims) {
  IterationDomainInfo info;
  if (!dims.isValid())
    return info;
  if (solver.hasConflict(dims.rows) || solver.hasConflict(dims.cols)) {
    info.unprovenReason = IterationDomainUnprovenReason::InconsistentShape;
    return info;
  }

  info.proof = IterationDomainProof::Proven;
  info.unprovenReason = IterationDomainUnprovenReason::None;
  if (std::optional<int64_t> row = solver.getConstant(dims.rows))
    info.vRow = *row;
  if (std::optional<int64_t> col = solver.getConstant(dims.cols))
    info.vCol = *col;
  return info;
}

static unsigned assignShapeInferredDomainClass(
    ShapeConstraintSolver &solver, SmallVectorImpl<IterationDomainClass> &classes,
    DenseMap<std::pair<unsigned, unsigned>, unsigned> &provenClassByRoot,
    ShapeValueDims dims, const IterationDomainInfo &info, unsigned nodeId) {
  if (info.proof == IterationDomainProof::Proven) {
    std::pair<unsigned, unsigned> key{solver.find(dims.rows),
                                     solver.find(dims.cols)};
    auto it = provenClassByRoot.find(key);
    if (it != provenClassByRoot.end()) {
      classes[it->second].members.push_back(nodeId);
      return it->second;
    }

    unsigned classId = classes.size();
    IterationDomainClass klass;
    klass.id = classId;
    klass.info = info;
    klass.members.push_back(nodeId);
    classes.push_back(std::move(klass));
    provenClassByRoot.try_emplace(key, classId);
    return classId;
  }

  unsigned classId = classes.size();
  IterationDomainClass klass;
  klass.id = classId;
  klass.info = info;
  klass.members.push_back(nodeId);
  classes.push_back(std::move(klass));
  return classId;
}

static LogicalResult inferShapeConstraints(FusionBlockAnalysis &analysis) {
  ShapeConstraintSolver solver;
  DenseMap<Value, ShapeValueDims> dimsByValue;
  DenseMap<Value, unsigned> symbolDimByValue;

  for (const FusionComputeNode &node : analysis.computeNodes) {
    for (Value input : node.semantics.tileInputs)
      (void)getValueDims(solver, dimsByValue, symbolDimByValue, input);
    for (Value output : node.semantics.tileOutputs)
      (void)getValueDims(solver, dimsByValue, symbolDimByValue, output);
  }

  for (const FusionComputeNode &node : analysis.computeNodes)
    applyShapeConstraintsForNode(solver, dimsByValue, symbolDimByValue, node);

  analysis.iterationDomainClasses.clear();
  DenseMap<std::pair<unsigned, unsigned>, unsigned> provenClassByRoot;
  for (FusionComputeNode &node : analysis.computeNodes) {
    ShapeValueDims domainDims = getIterationDomainDimsForNode(
        solver, dimsByValue, symbolDimByValue, node);
    IterationDomainInfo info = buildIterationDomainInfo(solver, domainDims);
    node.iterationDomainClass = assignShapeInferredDomainClass(
        solver, analysis.iterationDomainClasses, provenClassByRoot, domainDims,
        info, node.id);
  }
  return success();
}

struct MutableLiveness {
  FusionValueLiveness live;
};

struct MutableWriteInstance {
  FusionWriteInstanceLiveness live;
  unsigned producerBlockOrder = 0;
};

static FusionWriteInstanceEscapeClass classifyEscapeClass(
    const FusionWriteInstanceLiveness &live) {
  if (live.hasExternalUsers || live.escapesBlock ||
      live.hasLocalHardBoundaryUsers) {
    return FusionWriteInstanceEscapeClass::HardExternal;
  }
  if (live.hasLocalBoundaryUsers)
    return FusionWriteInstanceEscapeClass::LocalBoundaryExternal;
  return FusionWriteInstanceEscapeClass::Internal;
}

static Value getWriteInstanceStorageValue(Operation *op, unsigned outputIndex,
                                          Value output) {
  if (auto dpsIface = dyn_cast<pto::PTO_DpsInitOpInterface>(op)) {
    unsigned tileOutputIndex = 0;
    for (Value init : dpsIface.getDpsInits()) {
      if (!isa<pto::TileBufType>(init.getType()))
        continue;
      if (tileOutputIndex == outputIndex)
        return init;
      ++tileOutputIndex;
    }
  }
  return output;
}

static unsigned getOrCreateLivenessSlot(DenseMap<Value, unsigned> &slotByValue,
                                        SmallVectorImpl<MutableLiveness> &slots,
                                        Value value) {
  auto [it, inserted] = slotByValue.try_emplace(value, slots.size());
  if (inserted) {
    MutableLiveness state;
    state.live.value = value;
    slots.push_back(std::move(state));
  }
  return it->second;
}

static void appendUniqueNode(SmallVectorImpl<unsigned> &nodes, unsigned nodeId) {
  if (!llvm::is_contained(nodes, nodeId))
    nodes.push_back(nodeId);
}

static void finalizeBlockLiveness(
    Block &block, DenseMap<Operation *, FusionOpKind> &kindByOp,
    DenseMap<Operation *, unsigned> &computeNodeByOp,
    SmallVectorImpl<MutableLiveness> &mutableLiveness) {
  for (MutableLiveness &state : mutableLiveness) {
    for (OpOperand &use : state.live.value.getUses()) {
      Operation *user = use.getOwner();
      if (user->getBlock() != &block) {
        state.live.hasExternalUsers = true;
        state.live.escapesBlock = true;
        continue;
      }

      auto kindIt = kindByOp.find(user);
      if (kindIt == kindByOp.end())
        continue;

      if (user->hasTrait<OpTrait::IsTerminator>())
        state.live.escapesBlock = true;

      switch (kindIt->second) {
      case FusionOpKind::Compute: {
        auto nodeIt = computeNodeByOp.find(user);
        if (nodeIt == computeNodeByOp.end())
          continue;
        appendUniqueNode(state.live.consumerNodes, nodeIt->second);
        state.live.lastLocalConsumer = nodeIt->second;
        break;
      }
      case FusionOpKind::LocalBoundary:
        state.live.hasLocalBoundaryUsers = true;
        break;
      case FusionOpKind::HardBoundary:
        state.live.hasLocalHardBoundaryUsers = true;
        break;
      }
    }
  }
}

static std::optional<unsigned> findReachingWriteInstance(
    ArrayRef<unsigned> writeInstanceIds,
    ArrayRef<MutableWriteInstance> mutableWriteInstances,
    std::optional<unsigned> userBlockOrder) {
  if (writeInstanceIds.empty())
    return std::nullopt;

  if (!userBlockOrder)
    return writeInstanceIds.back();

  for (unsigned writeInstanceId : llvm::reverse(writeInstanceIds)) {
    if (mutableWriteInstances[writeInstanceId].producerBlockOrder <
        *userBlockOrder)
      return writeInstanceId;
  }
  return std::nullopt;
}

static bool isDpsInitOperandUse(OpOperand &use) {
  auto dpsIface = dyn_cast<pto::PTO_DpsInitOpInterface>(use.getOwner());
  if (!dpsIface)
    return false;

  for (OpOperand &dpsInit : dpsIface.getDpsInitsMutable())
    if (&dpsInit == &use)
      return true;
  return false;
}

static void finalizeWriteInstances(
    Block &block, DenseMap<Operation *, FusionOpKind> &kindByOp,
    DenseMap<Operation *, unsigned> &computeNodeByOp,
    DenseMap<Operation *, unsigned> &blockOrderByOp,
    ArrayRef<MutableLiveness> mutableLiveness,
    SmallVectorImpl<MutableWriteInstance> &mutableWriteInstances) {
  for (const MutableLiveness &storageState : mutableLiveness) {
    if (storageState.live.writeInstances.empty())
      continue;

    for (OpOperand &use : storageState.live.value.getUses()) {
      if (isDpsInitOperandUse(use))
        continue;

      Operation *user = use.getOwner();
      bool isInBlock = user->getBlock() == &block;
      std::optional<unsigned> userBlockOrder;
      if (isInBlock) {
        auto orderIt = blockOrderByOp.find(user);
        if (orderIt != blockOrderByOp.end())
          userBlockOrder = orderIt->second;
      }

      std::optional<unsigned> writeInstanceId = findReachingWriteInstance(
          storageState.live.writeInstances, mutableWriteInstances,
          userBlockOrder);
      if (!writeInstanceId)
        continue;

      FusionWriteInstanceLiveness &writeLive =
          mutableWriteInstances[*writeInstanceId].live;

      if (!isInBlock) {
        writeLive.hasExternalUsers = true;
        writeLive.escapesBlock = true;
        continue;
      }

      auto kindIt = kindByOp.find(user);
      if (kindIt == kindByOp.end())
        continue;

      if (user->hasTrait<OpTrait::IsTerminator>())
        writeLive.escapesBlock = true;

      switch (kindIt->second) {
      case FusionOpKind::Compute: {
        auto nodeIt = computeNodeByOp.find(user);
        if (nodeIt == computeNodeByOp.end())
          continue;
        appendUniqueNode(writeLive.consumerNodes, nodeIt->second);
        writeLive.lastLocalConsumer = nodeIt->second;
        break;
      }
      case FusionOpKind::LocalBoundary:
        writeLive.hasLocalBoundaryUsers = true;
        break;
      case FusionOpKind::HardBoundary:
        writeLive.hasLocalHardBoundaryUsers = true;
        break;
      }
    }
  }

  for (MutableWriteInstance &state : mutableWriteInstances)
    state.live.escapeClass = classifyEscapeClass(state.live);
}

static FailureOr<FusionBlockAnalysis> analyzeBlock(Block &block) {
  FusionBlockAnalysis analysis;
  analysis.block = &block;

  DenseMap<Value, unsigned> producerByValue;
  DenseMap<Value, unsigned> livenessSlotByValue;
  SmallVector<MutableLiveness, 8> mutableLiveness;
  SmallVector<MutableWriteInstance, 8> mutableWriteInstances;
  DenseMap<Operation *, FusionOpKind> kindByOp;
  DenseMap<Operation *, unsigned> computeNodeByOp;
  DenseMap<Operation *, unsigned> blockOrderByOp;

  unsigned blockOrder = 0;
  for (Operation &op : block) {
    FailureOr<FusionOpSemantics> semanticsOr = getFusionOpSemantics(&op);
    if (failed(semanticsOr)) {
      op.emitError("failed to normalize fusion op semantics");
      return failure();
    }
    blockOrderByOp[&op] = blockOrder;
    kindByOp[&op] = semanticsOr->kind;

    if (semanticsOr->kind == FusionOpKind::LocalBoundary) {
      for (Value input : semanticsOr->tileInputs)
        getOrCreateLivenessSlot(livenessSlotByValue, mutableLiveness, input);
      for (Value output : semanticsOr->tileOutputs)
        getOrCreateLivenessSlot(livenessSlotByValue, mutableLiveness, output);
      ++blockOrder;
      continue;
    }

    if (semanticsOr->kind != FusionOpKind::Compute) {
      ++blockOrder;
      continue;
    }

    FusionComputeNode node;
    node.id = analysis.computeNodes.size();
    node.blockOrder = blockOrder;
    node.op = &op;
    node.semantics = *semanticsOr;
    computeNodeByOp[&op] = node.id;

    for (auto [outputIdx, output] : llvm::enumerate(node.semantics.tileOutputs)) {
      producerByValue[output] = node.id;
      unsigned liveSlot =
          getOrCreateLivenessSlot(livenessSlotByValue, mutableLiveness, output);
      mutableLiveness[liveSlot].live.producerNode = node.id;

      MutableWriteInstance writeInstance;
      writeInstance.live.id = mutableWriteInstances.size();
      writeInstance.live.value = output;
      writeInstance.live.storageValue =
          getWriteInstanceStorageValue(&op, outputIdx, output);
      writeInstance.live.producerNode = node.id;
      writeInstance.producerBlockOrder = blockOrder;
      mutableLiveness[liveSlot].live.writeInstances.push_back(
          writeInstance.live.id);
      mutableWriteInstances.push_back(std::move(writeInstance));
    }

    for (Value input : node.semantics.tileInputs) {
      unsigned liveSlot =
          getOrCreateLivenessSlot(livenessSlotByValue, mutableLiveness, input);
      appendUniqueNode(mutableLiveness[liveSlot].live.consumerNodes, node.id);
      mutableLiveness[liveSlot].live.lastLocalConsumer = node.id;

      auto producerIt = producerByValue.find(input);
      if (producerIt == producerByValue.end())
        continue;

      FusionDFGEdge edge;
      edge.producerNode = producerIt->second;
      edge.consumerNode = node.id;
      edge.value = input;

      unsigned edgeId = analysis.edges.size();
      analysis.edges.push_back(edge);
      node.incomingEdges.push_back(edgeId);
      if (edge.producerNode < analysis.computeNodes.size())
        analysis.computeNodes[edge.producerNode].outgoingEdges.push_back(edgeId);
    }

    analysis.computeNodes.push_back(std::move(node));
    ++blockOrder;
  }

  finalizeBlockLiveness(block, kindByOp, computeNodeByOp, mutableLiveness);
  finalizeWriteInstances(block, kindByOp, computeNodeByOp, blockOrderByOp,
                         mutableLiveness, mutableWriteInstances);

  analysis.liveness.reserve(mutableLiveness.size());
  for (MutableLiveness &state : mutableLiveness)
    analysis.liveness.push_back(std::move(state.live));
  analysis.writeInstances.reserve(mutableWriteInstances.size());
  for (MutableWriteInstance &state : mutableWriteInstances)
    analysis.writeInstances.push_back(std::move(state.live));

  if (failed(inferShapeConstraints(analysis)))
    return failure();

  return std::move(analysis);
}

static LogicalResult analyzeRegion(Region &region,
                                   SmallVectorImpl<FusionBlockAnalysis> &blocks) {
  for (Block &block : region.getBlocks()) {
    FailureOr<FusionBlockAnalysis> blockAnalysis = analyzeBlock(block);
    if (failed(blockAnalysis))
      return failure();
    blocks.push_back(std::move(*blockAnalysis));
    for (Operation &op : block)
      for (Region &nested : op.getRegions())
        if (failed(analyzeRegion(nested, blocks)))
          return failure();
  }
  return success();
}

} // namespace

FailureOr<PreFusionAnalysisResult> buildPreFusionAnalysis(func::FuncOp func) {
  PreFusionAnalysisResult result;
  if (failed(analyzeRegion(func.getRegion(), result.blocks)))
    return failure();
  return std::move(result);
}

} // namespace pto
} // namespace mlir
