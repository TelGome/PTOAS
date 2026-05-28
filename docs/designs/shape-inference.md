# Shape Facts 与 Iteration-Domain 设计

## 1. 定位

本文是任务 7.2 的**唯一主文档**，只回答下面五件事：

- 如何建立 tile 级 `valid_row / valid_col` 推导能力
- 如何判定两个 tile op 是否工作在同一迭代域
- 如何支持 `treshape`、view-like op、region yield/result 的 shape 传播
- 动态 shape 下哪些场景允许保守证明，哪些场景必须拒绝
- 如何为 `FusionPlan` / `RegionGen` / 版本选择提供统一 shape facts 查询接口

Discussion 382 中 7.2 的交付物是：

- 独立 `shape facts / iteration-domain` 分析模块
- 正负例测试
- 面向融合 pass 的统一查询接口

实现附录见 [tile-fusion-shape-facts-iteration-domain-design.md](./tile-fusion-shape-facts-iteration-domain-design.md)。

## 2. 设计原则

### 2.1 只做编译期 facts，不做运行时求值

分析模块在编译期只追踪“可证明的维度事实”，不计算运行时 shape 数值。

第一版只支持四类维度表达式：

- `Unknown`
- `Const(int64_t)`
- `Value(Value)`
- `Min(Value, Const)`

不支持一般算术归一化，例如：

- `%a + %b`
- `%x * 32`
- `%x / 4`

### 2.2 不证明，不融合

所有融合相关查询都返回三值结果：

```c++
enum class ProofResult {
  ProvedTrue,
  ProvedFalse,
  Unknown
};
```

只有 `ProvedTrue` 允许继续融合；`Unknown` 必须保守拒绝。

### 2.3 区分语义域与物理编码

7.2 分析只在**逻辑 valid region** 上推理，不把下面三层混在一起：

- `physicalShape`：tile 的物理形状
- `valid_shape`：当前 op 真实读写的逻辑区域
- verifier / 模板编码约束：当前实现接受的具体布局与编码形式

例如 `RowExpandBinary` 的 broadcaster 在语义上只要求“每行提供一个逻辑标量”，不要求其物理 `valid_col` 必然等于 `1`。

### 2.4 统一查询接口，服务多个消费者

`FusionPlan`、`RegionGen` 和版本选择都不应各自回溯 SSA。7.2 必须产出统一的查询入口。

## 3. 核心抽象

### 3.1 语义角色

```c++
enum class TileSemanticRole {
  Dense2D,
  LogicalScalarPerRow,
  Unknown
};
```

- `Dense2D`：逻辑上按二维区域访问
- `LogicalScalarPerRow`：每行提供一个逻辑标量 carrier
- `Unknown`：当前无法稳定分类

### 3.2 `TileShapeFacts`

```c++
struct DimExpr {
  enum Kind { Unknown, Const, Value, MinValueConst } kind;
  int64_t constVal = 0;
  Value value;
  int64_t upperBound = 0;
};

struct TileShapeFacts {
  SmallVector<int64_t, 2> physicalShape;
  SmallVector<DimExpr, 2> validExprs;
  TileSemanticRole role = TileSemanticRole::Unknown;
  bool validIsExact = false;
  bool hasUnsupportedViewTransform = false;
};
```

说明：

- `physicalShape` 来自结果 `tile_buf` type
- `validExprs[0/1]` 分别表示 `valid_row / valid_col`
- `role` 表示该 value 的逻辑 shape 角色
- `validIsExact = false` 表示 facts 仅保守可用
- `hasUnsupportedViewTransform = true` 表示经过了当前首版不敢精确传播的 view 变换

### 3.3 `IterationDomain`

```c++
enum class IterationDomainKind {
  DstValidShape,
  Src0ValidShape,
  Src1ValidShape,
  Unsupported
};

struct IterationDomain {
  IterationDomainKind kind;
  SmallVector<DimExpr, 2> dims;
  bool exact = false;
};
```

这里的“迭代域”指 op 实际按谁的 `valid_shape` 进行循环，而不是“输入输出看起来是否同形”。

## 4. Shape Facts 传播规则

### 4.1 基础 producer

对 `alloc_tile / declare_tile / bind_tile / materialize_tile / pointer_cast`：

- `physicalShape` 取结果 type
- 静态 `validShape` 记为 `Const`
- 动态 `validShape` 记为 `Value`
- `role` 初始为 `Dense2D`，除非 type 或语义已明确更窄

### 4.2 `set_validshape`

- 继承 source 的 `physicalShape`
- 用新 operands 覆盖 `valid_row / valid_col`
- `validIsExact = true`

### 4.3 `subview` 与 view-like op

第一版只承认**结果端显式可见**的 valid 事实，不偷偷从 parent 反推。

规则：

- `physicalShape = result type shape`
- 若无显式 valid，结果 `validExprs = Const(sizeR), Const(sizeC)`
- 若有显式常量 valid，结果记为 `Const(min(cst, size))`
- 若有显式动态 valid，结果记为 `Min(Value(valid), Const(size))`

`bitcast` 直接透传 source facts，因为 verifier 已要求 shape / valid / config 一致。

### 4.4 `treshape`

这是 7.2 里最需要保守化的部分。

第一版规则：

- 若结果 `validShape` 两维都是静态常量，直接采用结果端静态 facts
- 若 source/result 的 valid 都是静态满 tile，也可采用结果端静态 facts
- 其他带动态 valid 的 `treshape`，统一标记为：
  - `validExprs = Unknown`
  - `validIsExact = false`
  - `hasUnsupportedViewTransform = true`

原因是当前主线还没有一套对动态 `treshape` 足够稳妥的几何证明规则。

### 4.5 region yield/result

首版需要支持 region 边界的 facts 合并。

对 `scf.if` result：

- then / else yield facts 完全一致时，result 继承该 facts
- 否则 result 的 `validExprs = Unknown`

对 `scf.for` iter_arg/result：

- `initArg` 与 `yield` facts 完全一致时，result 继承该 facts
- body 若直接 `yield iterArg`，result 继承 `initArg`
- 其他情况统一 `Unknown`

这等价于“支持纯传递，不支持循环中逐轮改变 valid shape 的证明”。

## 5. 算子语义归一化

7.2 第一版只支持四类 tile 计算族：

- Element-wise
- RowExpand
- RowReduce
- RowExpandBinary

### 5.1 Element-wise

逻辑语义：

```text
dst(r, c) = F(src0(r, c), src1(r, c), ...)
```

分析事实：

- 所有 full-tile 输入与输出共享相同 `valid_row / valid_col`
- 所有 value 的 `role = Dense2D`
- `IterationDomain = dst.valid_shape`

### 5.2 RowExpand

逻辑语义：

```text
dst(r, c) = src(r, 0)
```

分析事实：

- `src.role = LogicalScalarPerRow`
- `dst.role = Dense2D`
- `src.valid_row == dst.valid_row`
- `IterationDomain = dst.valid_shape`

注意：这里说的是**逻辑 broadcaster**，不是强制 `src.valid_col == 1` 的唯一物理编码。

### 5.3 RowReduce

逻辑语义：

```text
dst(r, 0) = reduce_c(src(r, c))
```

分析事实：

- `src.role = Dense2D`
- `dst.role = LogicalScalarPerRow`
- `src.valid_row == dst.valid_row`
- `IterationDomain = src.valid_shape`

这里必须显式区分：

- `src.valid_col` 是 reduction extent
- `dst.valid_col` 是归约结果宽度

因此不能把 `RowReduce` 错误建模成“输入输出同 `valid_col`”。

### 5.4 RowExpandBinary

逻辑语义：

```text
dst(r, c) = F(full(r, c), scalar(r))
```

分析事实：

- 一个输入扮演 `Dense2D`
- 一个输入扮演 `LogicalScalarPerRow`
- `dst.role = Dense2D`
- `full.valid_shape == dst.valid_shape`
- `scalar.valid_row == dst.valid_row`
- `IterationDomain = dst.valid_shape`

同域证明比较的是**标准化后的语义域**，不是 broadcaster 的物理编码。

## 6. 统一查询接口

建议对外提供两层接口：

```c++
class ShapeFactsAnalysis {
public:
  FailureOr<TileShapeFacts> getFacts(Value value);

  ProofResult proveSameValidShape(Value lhs, Value rhs);
  ProofResult proveDimEq(Value lhs, unsigned lhsDim, Value rhs, unsigned rhsDim);
  ProofResult proveDimLE(Value lhs, unsigned lhsDim, Value rhs, unsigned rhsDim);

  std::optional<int64_t> getStaticValidDim(Value value, unsigned dim);
  bool isFullTile(Value value);
  bool hasDynamicOrUnknownValid(Value value);
};

class IterationDomainAnalysis {
public:
  FailureOr<IterationDomain> getDomain(Operation *op);
  ProofResult proveSameDomain(Operation *lhs, Operation *rhs);
};
```

接口职责：

- `FusionPlan / RegionGen` 主要依赖 `getDomain` 与 `proveSameDomain`
- 版本选择主要依赖 `getFacts`、`getStaticValidDim`、`proveDimLE`
- legality adapter 如需判断“当前 verifier / 模板编码是否接受”，应作为独立层追加，不污染 7.2 的语义证明

## 7. 同一迭代域判定规则

`proveSameDomain(lhs, rhs)` 只有在下面条件都满足时返回 `ProvedTrue`：

1. 两个 op 都属于当前支持的算子族
2. 两侧迭代域都能提取为受支持的 `DimExpr`
3. `row` 与 `col` 都能分别证明相等
4. 中间没有 `hasUnsupportedViewTransform` 污染

返回 `ProvedFalse` 的场景只限于“常量层面直接可证不等”。

其余一律返回 `Unknown`。例如：

```mlir
%row0 = memref.load %a[%c0]
%row1 = memref.load %b[%c0]
```

即使运行时 `%row0 == %row1`，当前也只能返回 `Unknown`。

## 8. 动态 Shape 下的保守策略

### 8.1 允许证明

- 同一 SSA `valid_row / valid_col` 直接传递
- 常量 valid 维度
- `subview` 形成的 `Min(Value, Const)`
- region 两侧 yield 完全相同

### 8.2 必须拒绝

- 动态 `treshape`
- 需要复杂代数化简才能证明相等
- loop-carried facts 每轮可能变化
- region 两侧 yield facts 不一致
- 经过 view-like 链路后 facts 已退化为 `Unknown`

### 8.3 失败原因

建议统一 failure reason，供融合日志和版本选择复用：

- `UnsupportedDynamicTReshape`
- `UnknownIterationDomain`
- `MismatchedIterationDomain`
- `UnsupportedOpFamily`
- `DivergentRegionYieldFacts`

## 9. 测试与首版验收

### 9.1 正例

- `alloc_tile` 动态 valid 直传
- `set_validshape` 后 facts 被正确覆盖
- `subview` 生成 `Const` 或 `Min(Value, Const)` facts
- `scf.if` 两支 yield 相同 facts
- `scf.for` 纯传递 iter_arg
- element-wise 共域证明成功
- `RowExpandBinary` 在 broadcaster 物理编码不同的情况下仍能证明语义同域
- `RowReduce` 能正确使用源操作数作为迭代域

### 9.2 负例

- 动态 `treshape` 返回 `Unknown`
- 两个逐元素 op 的 `valid_row` 来自不同 SSA
- `subview` 结果不能错误继承 parent partial valid
- 不能把 `RowReduce` 误判成“输入输出同 `valid_col`”
- 不能要求 `RowExpandBinary` 的 broadcaster 与 `dst` 拥有相同 `valid_col`

## 10. 非目标

本文不负责：

- 重新定义 `subview / treshape` 的 IR 语义
- 直接实现 cost model
- 直接实现版本选择规则
- 一次性覆盖 `ColExpand / ColReduce / ColExpandBinary / Partial elementwise`

首版重点只有一句话：

```text
不证明，不融合
```
