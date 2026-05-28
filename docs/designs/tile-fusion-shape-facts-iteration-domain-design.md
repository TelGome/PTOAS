# ShapeFacts / IterationDomain 实现附录

本文是 [shape inference.md](./shape%20inference.md) 的实现附录，只保留与任务 7.2 直接相关的代码组织建议。

不再重复：

- 7.2 的目标与非目标
- 各算子族的标准化语义规则
- 动态 shape 下的证明/拒绝边界

这些内容统一以主文档为准。

## 1. 模块落点

建议新增独立子目录：

- `include/PTO/Transforms/ShapeAnalysis/ShapeFacts.h`
- `include/PTO/Transforms/ShapeAnalysis/IterationDomainAnalysis.h`
- `lib/PTO/Transforms/ShapeAnalysis/ShapeFacts.cpp`
- `lib/PTO/Transforms/ShapeAnalysis/IterationDomainAnalysis.cpp`

理由：

- 7.2 首先服务 tile fusion，但本身不应耦合在某一个 fusion pass 文件里
- `ShapeFactsAnalysis` 和 `IterationDomainAnalysis` 后续还会被版本选择复用
- 当前仓库更接近“Transforms 下独立子模块”的组织方式

## 2. 推荐数据结构

主文档已经定义了 `DimExpr`、`TileShapeFacts`、`IterationDomain` 与 `ProofResult`。实现上建议保持最小化：

```c++
class ShapeFactsAnalysis {
public:
  explicit ShapeFactsAnalysis(func::FuncOp func);

  FailureOr<TileShapeFacts> getFacts(Value value);

  ProofResult proveSameValidShape(Value lhs, Value rhs);
  ProofResult proveDimEq(Value lhs, unsigned lhsDim, Value rhs, unsigned rhsDim);
  ProofResult proveDimLE(Value lhs, unsigned lhsDim, Value rhs, unsigned rhsDim);

  std::optional<int64_t> getStaticValidDim(Value value, unsigned dim);
  bool isFullTile(Value value);
  bool hasDynamicOrUnknownValid(Value value);

private:
  DenseMap<Value, FailureOr<TileShapeFacts>> factCache;
};

class IterationDomainAnalysis {
public:
  explicit IterationDomainAnalysis(func::FuncOp func,
                                   ShapeFactsAnalysis &shapeFacts);

  FailureOr<IterationDomain> getDomain(Operation *op);
  ProofResult proveSameDomain(Operation *lhs, Operation *rhs);
};
```

设计约束：

- 第一版按 `func::FuncOp` 生命周期构造，不强依赖 `AnalysisManager`
- `ShapeFactsAnalysis` 负责 facts 传播与维度证明
- `IterationDomainAnalysis` 只负责“某个 op 的循环域来自哪里”与同域比较

## 3. Producer 适配范围

`ShapeFactsAnalysis` 首版建议只覆盖下面几类 producer：

- `alloc_tile / declare_tile / bind_tile / materialize_tile / pointer_cast`
- `set_validshape`
- `subview`
- `bitcast`
- `treshape`
- `scf.if` result
- `scf.for` iter_arg/result

对应实现原则：

- `subview` 跟随当前 result 端语义，不做 parent partial valid 继承
- `bitcast` 直接透传
- 动态 `treshape` 返回保守 `Unknown`
- region result 只在两侧 facts 完全一致时合并

## 4. Op Family 适配范围

`IterationDomainAnalysis` 首版只接入四类 tile 计算族：

- Element-wise
- RowExpand
- RowReduce
- RowExpandBinary

域来源表建议固定为：

| Op family | 迭代域来源 |
| --- | --- |
| Element-wise | `dst.valid_shape` |
| RowExpand | `dst.valid_shape` |
| RowReduce | `src.valid_shape` |
| RowExpandBinary | `dst.valid_shape` |

注意：

- 这里比较的是主文档定义的**标准化语义域**
- verifier/layout 的特殊编码约束应放在独立 legality helper，而不是塞进 `proveSameDomain`

## 5. 失败原因与观测

建议定义统一 reason code，避免融合日志、support matrix 和版本选择各自发散：

```c++
enum class ShapeAnalysisFailureReason {
  UnsupportedDynamicTReshape,
  UnknownIterationDomain,
  MismatchedIterationDomain,
  UnsupportedOpFamily,
  DivergentRegionYieldFacts,
};
```

首版至少提供一种稳定观测方式：

- lit 测试中的文本化 dump
- 或隐藏 debug pass
- 或 pass debug log 中的统一 reason code 输出

## 6. 落地顺序

### Phase 1：基础 facts

- 落地 `ShapeFactsAnalysis`
- 打通基础 producer、`set_validshape`、`subview`、`bitcast`
- 提供 `proveSameValidShape` / `proveDimEq`

### Phase 2：region 与保守拒绝

- 支持 `scf.if` result
- 支持 `scf.for` 纯传递迭代参数
- 把动态 `treshape` 明确收敛到 `Unknown`

### Phase 3：迭代域分析

- 落地 `IterationDomainAnalysis`
- 接入四类 op family
- 提供 `proveSameDomain`

### Phase 4：测试与接口收口

- 补齐正负例 lit
- 给 `FusionPlan / RegionGen / 版本选择` 暴露统一查询接口
- 检查所有 fallback 是否都带失败原因

## 7. 首版完成标准

完成 7.2 时，至少应满足：

1. 主文档中的四类算子语义已经在实现中有唯一对应规则。
2. `subview`、`treshape`、region yield/result 的传播边界已经稳定，不靠隐式约定。
3. 同域证明只在 `ProvedTrue` 时放行，其余稳定回退。
4. 失败原因可观测，测试覆盖正负例。

如果实现需要补充新的算子语义，应先改主文档，再扩展本附录。
