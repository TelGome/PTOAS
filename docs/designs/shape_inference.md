# Tile Fusion Shape Inference Design

## 1. 目标

完成 tile fusion 的 shape inference 约束推导。

输入：

- `/PTOAS/lib/PTO/Transforms/TileFusion/PTOPreFusionAnalysis.cpp` 之后得到的 DAG 图，即 `PreFusionAnalysisResult` / `FusionBlockAnalysis`
- 每个 block 内的 `FusionComputeNode`
- 节点之间的 `FusionDFGEdge`
- 每个节点已有的 `FusionOpSemantics`

输出：

- 给每个 DAG node 推导出的迭代域
- 哪些 node 的迭代域可证明相同
- 给 `/PTOAS/lib/PTO/Transforms/TileFusion/PTOFusionPlan.cpp`
  使用的 `FusionComputeNode::iterationDomainClass` /
  `FusionBlockAnalysis::iterationDomainClasses` 等价信息

这里的 dynamic shape 不是由编译器在编译期计算出具体运行时数值，而是在编译时
分析依赖图，建立 shape 约束关系。solver 只证明“哪些维度必须相同”或“哪些维度
被约束为静态常量”，再据此判断 compute node 的 iteration domain 是否相同。

核心问题：

```text
在 PreFusionAnalysis 产出的 DAG 上建立 shape 约束系统，
判断哪些 compute node 的 iteration domain 可证明相同。
```

## 2. 位置

当前实现可以继续放在 tile fusion analysis 层，消费
`PreFusionAnalysisResult` / `FusionBlockAnalysis` 中已经建好的 DAG，不直接修改
IR。底层 shape inference 逻辑由两部分组成：

```text
Shape constraint solver:
  用并查集合并维度等价关系，记录等价类上的常量约束。

Shape propagation rules:
  按 op 内 operand/result 关系和 producer / consumer 关系解释 shape 关系，把规则转成 solver 约束，
  并从 solver 查询 node 的 iteration domain。
```

如果后续规则继续膨胀，可以迁移到独立文件：

```text
lib/PTO/Transforms/TileFusion/PTOShapeInference.cpp
include/PTO/Transforms/TileFusion/ShapeInference.h
```

推荐接线方式：

```text
PTOPreFusionAnalysis.cpp / FusionAnalysis.cpp
  生成 DAG: PreFusionAnalysisResult

Shape inference
  读取 DAG
  建立并查集维度等价类
  记录常量维度约束
  运行 producer / consumer 传播规则
  回填 domain class

PTOFusionPlan.cpp
  读取回填后的 iterationDomainClass / iterationDomainClasses
  根据 domain class + DAG 连接关系做融合分组
```

`shape_inference` 不改 IR，只修改分析结果。当前推荐继续回填现有字段，减少
`PTOFusionPlan.cpp` 改动：

```text
FusionComputeNode::iterationDomainClass
FusionBlockAnalysis::iterationDomainClasses
```

## 3. 底层约束模型

### 3.1 维度变量

shape inference 不直接把一个 tile 的 shape 当成不可分割对象，而是把每个
row/col 维度拆成独立维度变量：

```c++
struct ShapeDimVar {
  unsigned id;
};
```

每个需要参与推导的维度都会映射到一个 `ShapeDimVar`，例如：

```text
%a.valid_rows -> dim var A_r
%a.valid_cols -> dim var A_c
%b.valid_rows -> dim var B_r
%b.valid_cols -> dim var B_c
```

维度变量来源包括：

- tile type 中的静态 shape / valid-shape
- `pto.bind_tile` 或同类 op 上显式给出的 valid row / valid col
- 算子 result 的 shape 约束信息
- DAG 边上 producer / consumer 之间需要传播的 shape 约束信息

未知动态维度也会创建变量。变量未知不等于推导失败；solver 不尝试计算动态
维度的具体值，只传播约束。只有当 FusionPlan 需要证明两个 node 同域而 solver
无法证明它们落在同一个等价类时，才标记为 `Unproven`。

### 3.2 并查集等价类

底层 solver 使用并查集维护“哪些维度必须相同”：

```c++
struct ShapeDimClass {
  unsigned root;
  std::optional<int64_t> constant;
  SmallVector<ShapeDimVar, 4> members;
};
```

约束形式保持很小：

```text
merge(lhsDim, rhsDim)
bindConstant(dim, c)
```

`merge(lhsDim, rhsDim)` 表示两个维度必须相同。`bindConstant(dim, c)` 表示该
维度等价类必须等于静态常量 `c`。

当两个等价类合并时，常量约束一起合并：

```text
classA.constant = 32
classB.constant = none
merge(A, B) => merged.constant = 32

classA.constant = 32
classB.constant = 64
merge(A, B) => conflict，相关 domain 标记为 Unproven
```

因此 solver 能回答两类问题：

```text
1. 两个维度是否可证明相同
2. 一个维度是否被绑定到某个常量约束
```

### 3.3 Value 级 shape 约束信息

```c++
struct ShapeValueDims {
  ShapeDimVar validRows;
  ShapeDimVar validCols;
};
```

`ShapeValueDims` 描述一个 rank-2 tile value 的 valid shape 维度变量。它只
提供约束信息，不直接决定融合。

### 3.4 Node 级迭代域

```c++
struct IterationDomainInfo {
  ShapeDimVar rows;
  ShapeDimVar cols;
  IterationDomainProof proof;
  IterationDomainUnprovenReason unprovenReason;
};
```

node 的 iteration domain 是一组 `(rows, cols)` 维度变量。producer /
consumer 传播规则决定这组变量来自哪些 operand/result。

例如：

```text
elementwise:
  domain = output.valid_shape
  input.valid_shape == output.valid_shape

reduce:
  domain = input.valid_shape
  output.valid_shape 不和 input.valid_shape 合并
```

### 3.5 Domain class

domain class 是 solver 查询结果，不是另一个独立推导系统：

```text
两个 node 的 rows 在同一个并查集等价类，
并且 cols 在同一个并查集等价类，
并且相关等价类没有常量冲突，
=> 两个 node 属于同一个 proven domain class。
```

`PTOFusionPlan.cpp` 只消费：

- 某个 node 是否 `Proven`
- 某个 node 的 `domainClass`
- 两个 node 是否属于同一个 `domainClass`

## 4. 约束传播规则

`shape_inference` 的规则包含两层：

```text
intra-op rule:
  一个 op 内部 tile inputs / tile outputs 的 shape 约束。

inter-op rule:
  producer output 被 consumer input 使用时的 shape 约束。
```

每条规则最终都会落到以下三种基础动作：

```text
merge(lhsDim, rhsDim):
  两个维度必须相同，合并到同一个并查集等价类。

mergeConst1(dim):
  该维度与常量 1 合并，等价类绑定常量 1。

keep(lhsDim, rhsDim):
  两个维度不建立约束，各自保持独立。
```

实验阶段按以下逻辑 family 建立规则：

- `FusionComputeFamily::Elementwise`
- `RowExpand` / `ColExpand` 单输入广播
- `FusionComputeFamily::RowBroadcastBinary` 及后续 col-broadcast binary 变体
- `FusionComputeFamily::ReduceRow`
- `FusionComputeFamily::ReduceCol`
- producer output 和 consumer input 是同一个 SSA value 时共享同一组维度变量

`FusionComputeFamily::ScalarExpand` 当前对应 `texpands`，不纳入本实验阶段；
其它 unsupported op family 当前一律保持 `Unproven`，不参与可融合 domain class。

### 4.1 OP 支持清单

当前 `FusionOpSemantics.cpp` 已支持的 OP 只是后续目标的子集：

| Family | 当前已支持 OP |
| --- | --- |
| `Elementwise` | `tadd`, `tsub`, `tmul`, `tdiv`, `tmax`, `tmin`, `tadds`, `tsubs`, `tmuls`, `tdivs`, `tmaxs`, `tmins`, `texp` |
| `RowBroadcastBinary` | `trowexpandmul`, `trowexpanddiv` |
| `ReduceRow` / `ReduceCol` | `trowsum`, `trowmax`, `trowmin`, `tcolsum`, `tcolmax`, `tcolmin` |
| 暂不纳入实验阶段 | `texpands` |

实验阶段 shape inference 需要按以下目标列表逐步补齐。每个 OP 打开前都要确认
`FusionOpSemantics` 的 tile input / output 归一化顺序与该 OP 的 TileOp 语义一致。

| Family | 后续目标 OP |
| --- | --- |
| `Elementwise` 二元 tile | `tadd`, `tsub`, `tmul`, `tdiv`, `tmax`, `tmin`, `trem`, `tfmod`, `tand`, `tor`, `txor`, `tshl`, `tshr`, `tcmp` |
| `Elementwise` tile-scalar | `tadds`, `tsubs`, `tmuls`, `tdivs`, `tmaxs`, `tmins`, `trems`, `tfmods`, `tands`, `tors`, `txors`, `tshls`, `tshrs`, `tcmps` |
| `Elementwise` 单输入 | `tabs`, `tneg`, `tnot`, `texp`, `tlog`, `tsqrt`, `trsqrt`, `trecip`, `trelu`, `tlrelu`, `tcvt`, `tquant`, `tdequant` |
| `Elementwise` 多输入 / fused | `taddc`, `tsubc`, `taddsc`, `tsubsc`, `taxpy`, `tprelu`, `tsel`, `tsels` |
| `RowExpand` / `ColExpand` | `trowexpand`, `tcolexpand` |
| `RowExpandBinary` | `trowexpandadd`, `trowexpandsub`, `trowexpandmul`, `trowexpanddiv`, `trowexpandexpdif`, `trowexpandmax`, `trowexpandmin` |
| `ColExpandBinary` | `tcolexpandadd`, `tcolexpandsub`, `tcolexpandmul`, `tcolexpanddiv`, `tcolexpandexpdif`, `tcolexpandmax`, `tcolexpandmin` |
| `RowReduce` | `trowsum`, `trowprod`, `trowmax`, `trowmin`, `trowargmax`, `trowargmin` |
| `ColReduce` | `tcolsum`, `tcolprod`, `tcolmax`, `tcolmin`, `tcolargmax`, `tcolargmin` |

以下 OP 不属于本实验阶段四类规则，首版保持 `UnsupportedOpFamily` 或作为
boundary 处理：

- DMA / GM-UB：`tload`, `tstore`, `tprefetch`, `tstore_fp`
- Cube / matrix：`tmatmul*`, `tgemv*`
- pipe / sync / communication：`tpush`, `tpop`, `tfree`, `talloc`, `tassign`,
  `tsync`, `comm.*`
- view / movement boundary：`treshape`, `tmov`, `tmov.fp`, `ttrans`
- scalar element access：`tgetval`, `tsetval`
- partial / irregular / index / sorting / fill / debug：
  `tpartadd`, `tpartmul`, `tpartmax`, `tpartmin`, `tpartargmax`,
  `tpartargmin`, `tgather`, `tgatherb`, `tscatter`, `tsort32`, `tmrgsort`,
  `tconcat`, `tconcatidx`, `textract`, `textract_fp`, `tinsert`,
  `tinsert_fp`, `tfillpad`, `tfillpad_expand`, `tfillpad_inplace`,
  `thistogram`, `trandom`, `tci`, `ttri`, `tprint`

### 4.2 基础动作

#### merge

```text
merge(A.rows, B.rows)
merge(A.cols, B.cols)
```

表示 `A` 和 `B` 在对应维度上必须相同。合并后，如果任意一边已经有常量
约束，常量约束会传播到合并后的等价类。

#### mergeConst1

```text
mergeConst1(A.cols)
```

表示该维度被约束为规约或广播后的常量 `1`。solver 在该维度等价类上记录
常量 `1`。如果同一等价类此前已经绑定到其它常量，则该 domain 冲突并标记为
`Unproven`。

#### keep

```text
keep(A.cols, B.cols)
```

表示两个维度之间没有可证明的相等关系，也不应该被合并。典型场景是 broadcast
或 expand 中新扩展出的维度。

### 4.3 Elementwise

element-wise op 的 tile inputs 和 tile outputs 的 row / col 完全相同：

```text
merge(input0.rows, output.rows)
merge(input0.cols, output.cols)
merge(input1.rows, output.rows)
merge(input1.cols, output.cols)
```

因此 element-wise 链上的所有 tile input / output 的 valid rows、valid cols
都会落到同一组维度等价类。

### 4.4 RowExpand

`RowExpand` / `ColExpand` 对应单输入广播，例如 `trowexpand` / `tcolexpand`。
它们的 iteration domain 来自 tile output：

```text
domain = output.rows x output.cols
```

广播输入只参与合法性检查，不和 output 的完整矩形直接合并。

### 4.5 RowExpandBinary

`RowExpandBinary` 当前对应 `trowexpandmul` / `trowexpanddiv`，后续扩展到
row/col broadcast binary 全量 OP。full tile input 与 output 同矩形，
row/col broadcast input 的 broadcast 维度约束为 `1`：

`tileInputs[0]`：

```text
merge(tile0.rows, output.rows)
merge(tile0.cols, output.cols)
```

`tileInputs[1]`：

```text
merge(tile1.rows, output.rows)
mergeConst1(tile1.cols)
keep(tile1.cols, output.cols)
```

这里的 `tileInputs[0]` / `tileInputs[1]` 依赖 `getFusionOpSemantics()` 保持
稳定 operand 顺序。

### 4.6 ReduceRow

`ReduceRow` 当前对应 `trowsum` / `trowmax` / `trowmin`。它的 iteration domain
来自 reduce input。输出 shape 与输入 shape 不是同一个完整矩形：

```text
domain = input.rows x input.cols
merge(input.rows, output.rows)
mergeConst1(output.cols)
keep(input.cols, output.cols)
```

如果 `output` 继续进入 element-wise consumer，consumer 会继承 `output.cols =
1` 的约束；但 reduce node 自己的 iteration domain 仍然是 input domain。
因此当前 FusionPlan 的 `sameDomainClass` 模型不会把 `reduce_row -> elementwise`
自动视为同域融合。

### 4.7 ReduceCol

`ReduceCol` 当前对应 `tcolsum` / `tcolmax` / `tcolmin`。它与 `ReduceRow`
对称：iteration domain 来自 reduce input，输出 row 维度被规约为常量 `1`：

```text
domain = input.rows x input.cols
merge(input.cols, output.cols)
mergeConst1(output.rows)
keep(input.rows, output.rows)
```

规约输出继续流向下游时，例如：

```text
tadd -> trowsum -> tadd
```

下游 consumer 只能看到规约后的输出 shape，因此下游 col 会传播为常量 `1`；
规约前输入 col 保持为 reduce 的遍历域，不与输出 col 合并。

## 5. DAG 上的推导流程

对每个 `FusionBlockAnalysis` 单独处理：

### Step 1：初始化维度变量和 value 级约束信息

从 DAG 中所有 node 的 `tileInputs` / `tileOutputs` 收集 value。

对每个 value 尝试恢复：

```text
validRows / validCols
```

来源优先级：

```text
1. tile type 中的静态 valid-shape
2. 显式 valid_row / valid_col operand，例如 alloc_tile / bind_tile / materialize_tile / subview
3. 同一个 dynamic SSA valid_row / valid_col value
4. Unknown
```

每个 rank-2 tile value 建立两个 valid-shape 维度变量：

```text
validRows, validCols
```

如果某个维度来自静态常量，则调用 `bindConstant(dim, c)`。如果某个维度来自
同一个 SSA value 或等价的 cast 链，则把它们映射到同一个维度变量，或通过
`merge` 合并。

### Step 2：收集 producer / consumer shape 约束

producer output 和 consumer input 如果是同一个 SSA value，会通过
`dimsByValue` 共享同一组维度变量：

```text
producer output value == consumer input value
  => same ShapeValueDims
```

显式 shape-changing 关系由对应 op family 的 intra-op 规则表达。

### Step 3：按 blockOrder 应用传播规则

对每个 `FusionComputeNode`：

```text
1. 根据 FusionOpSemantics.opName / family 分类
2. 查当前 op family 的 intra-op 规则
3. 对 operand/result 的 shape 维度增加 merge / mergeConst1 / keep 动作
4. 记录该 node 的 iteration domain 维度变量
```

如果缺少必要约束信息、遇到 unsupported family，或规则本身无法表达该 op 的
shape 关系：

```text
proof = Unproven
reason = 对应失败原因
```

但这类失败只影响对应 node 是否可作为可融合 domain，不应导致 pass failure。

### Step 4：求解并查集并构造 domain class

传播规则应用结束后，solver 已经知道所有维度等价类和常量约束。对所有
`FusionComputeNode`：

```text
node 有合法 iteration domain
并且 rows/cols 所在等价类没有常量冲突
并且 rows/cols 可 canonicalize
  => 放进同一个 ShapeDomainClass

proof == Unproven
  => 不进入可融合 domain class
```

domain class 表示：

```text
这些 DAG node 的实际遍历范围可证明相同。
```

### Step 5：输出给 FusionPlan

`PTOFusionPlan.cpp` 中：

```text
evaluateSeed(node):
  只有 nodeDomain.proof == Proven 才能作为 seed

evaluateAppend(group, candidate):
  candidate.domainClass == groupAnchor.domainClass
  并且 candidate 与 group 有 DAG 连接
  再进入 cost model
```

shape inference 回填：

```text
FusionComputeNode::iterationDomainClass
FusionBlockAnalysis::iterationDomainClasses
```

这样 `PTOFusionPlan.cpp` 的改动最小。

## 6. 动态 shape 约束证明

动态 shape 的处理目标不是算出动态维度的具体值，而是在编译期依赖图上证明
维度约束关系。当前可证明“必须相同”的动态 shape：

- 同一个 SSA value
- 去掉 `index_cast/ext/trunc` 后是同一个 SSA value
- 已经通过并查集 `merge` 到同一个等价类的维度
- 由 producer / consumer 规则强制要求的同矩形关系

当前可证明“被约束为常量”的 shape：

- tile type / shaped type 中的静态维度
- 显式常量 valid row / valid col
- 通过等价类传播得到的常量约束

当前不可证明：

- `%vrow0` 与 `%vrow1` 没有已知相等关系
- `%vrow` 与 `min(%vrow, 32)` 缺少上界证明
- `treshape` 后无法恢复精确矩形
- value 级约束信息缺失
- unsupported op family
- 同一个等价类绑定到两个不同常量

不可证明只影响 shape inference 的 `proof`，不应导致 pass failure。

## 7. FusionPlan 使用方式

`PTOFusionPlan.cpp` 不再自己判断 dynamic shape 是否可融合，也不计算动态
shape 值，而是只消费 `shape_inference` 的约束证明结果。

现有逻辑仍然可保留：

```text
node.iterationDomainClass
blockAnalysis.iterationDomainClasses[classId].info.proof
```

可以继续保留，但这些字段应由 `shape_inference` 统一产生。

新的职责划分：

```text
PTOPreFusionAnalysis / FusionAnalysis:
  建 DAG、DFG edge、liveness、write instance

Shape inference:
  在 DAG 上跑 shape 约束收集、并查集合并、常量传播、domain class 构造
  标记每个 node 的 domainClass

PTOFusionPlan:
  只根据 domainClass、DAG 连接、cost model 分组
```

## 8. 验收用例

### 正例

1. `tadd -> tmul`

```text
Elementwise intra-op 规则把每个 op 的 inputs/outputs rows/cols 合并。
两个 op 通过同一个 SSA value 共享维度变量，domain class 相同。
```

2. `tadd -> rowreduce`

```text
ReduceRow 的 iteration domain 来自 reduce input。
上游 element-wise output 和 reduce input 是同一个 SSA value，因此二者
domain class 相同。
reduce output col 绑定常量 1，但不与 reduce input col 合并。
```

3. `trowexpand -> tadd`

```text
RowExpand 的 domain 来自 output。
输出进入 element-wise 后，通过同一个 SSA value 共享 output rows/cols。
```

4. `trowexpandmul -> tadd`

```text
RowBroadcastBinary 的 tile0 与 output 同矩形，tile1 col 绑定为 const1 并和
output col keep。
输出进入 element-wise 后，consumer 使用输出 shape。
```

### 负例

1. 两个动态 row 值不同且无可证关系

```text
%vrow0 != %vrow1 无法证明，domain Unproven。
```

2. reduce input/output 被错误当成同矩形

```text
应拒绝把 reduce input col 和 reduce output col merge。
```

3. unsupported op family

```text
proof = Unproven
reason = missing_tile_domain
```

4. 常量冲突

```text
同一个等价类先 mergeConst1，后续又绑定到其它常量，应标记为 Unproven。
```

## 9. 实施顺序

1. 抽出 shape constraint solver：维度变量、并查集、常量约束、冲突标记。
2. 从 `analyzeBlock()` 中移出现有静态 `(vRow, vCol)` 直接比较逻辑，改为先
   收集 value 级约束信息、op 内约束和 producer / consumer 共享值约束。
3. 实现 Elementwise、RowExpand、RowBroadcastBinary、ReduceRow、
   ReduceCol 的 op 内传播规则。
4. 根据 solver 查询结果回填 `FusionComputeNode::iterationDomainClass` 和
   `FusionBlockAnalysis::iterationDomainClasses`，保证 `PTOFusionPlan.cpp`
   改动最小。
5. 增加 `tadd -> rowreduce`、动态 SSA 等价、常量冲突等测试。
6. 后续如有需要，再把 solver 结果拆成独立 sidecar 数据结构。

## 10. 总结

shape inference 的核心不是 FusionPlan 中的局部判断，而是一个独立的底层
约束求解过程：

```text
1. 为 tile value 的 rows/cols 建立维度变量
2. 用并查集合并 producer / consumer 规则要求相同的维度
3. 在等价类上记录常量约束并检测冲突
4. 根据传播规则确定每个 compute node 的 iteration domain
5. 把可证明相同的 iteration domain 分配到同一个 domain class
```

`PTOFusionPlan.cpp` 不再关心 shape 细节，只判断：

```text
node 是否 Proven
candidate 是否和 group 起点同 domain class
candidate 是否与 group 有 DAG 连接
cost model 是否接受
```

这样可以用最小结构变化支持更清晰的融合判断，包括 `rowreduce -> tadd` 这类
规约后 shape 会继续传播到下游的场景。
