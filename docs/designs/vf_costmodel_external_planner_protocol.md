# PTOAS + VfSim Costmodel 接口设计

本文描述 PTOAS 以源码级 submodule 方式接入 VfSimulator costmodel 的当前接口形式。
当前实现面向 A5 tile fusion 路径：PTOAS 负责生成合法 fusion group，VfSim 基于
已选 group 做 costmodel 优化决策，并把结果写回同一份 MLIR IR。

## 接入形式

VfSimulator 作为 PTOAS 的 git submodule 引入：

```text
PTOAS/
  3rdparty/
    VfSimulator/
```

PTOAS 仓库只记录 submodule commit；VfSimulator 的源码历史仍由 VfSimulator 仓库维护。
当前 `PTO_ENABLE_VFSIM_COSTMODEL=ON` 只支持 build-tree/source-tree 开发使用，
尚不支持 install/export。

## 控制选项

| 选项 | 类型 | 作用 |
|---|---|---|
| `-DPTO_ENABLE_VFSIM_COSTMODEL=ON` | CMake 编译期选项 | 编译并链接 VfSim native planner。默认 `OFF`。 |
| `--enable-vfsim-costmodel-optimization` | `ptoas` 运行期选项 | 启用 VfSim costmodel 优化。VPTO 路径下会同时启用 unroll attrs 的后端消费。 |
| `--dump-vfsim-unroll-test` | `ptoas` 运行期调试选项 | 只打印 VfSim 对各个 unroll candidate 的预测 cycle，不控制优化是否启用。 |

用户侧常用命令形态：

```bash
ptoas \
  --pto-backend=vpto \
  --pto-arch=a5 \
  --pto-level=level2 \
  --enable-op-fusion \
  --enable-vfsim-costmodel-optimization \
  input.pto
```

## 构建接入

| 文件 | 作用 |
|---|---|
| `CMakeLists.txt` | 引入 `cmake/VfSimulator.cmake`。 |
| `cmake/VfSimulator.cmake` | 定义 `PTO_ENABLE_VFSIM_COSTMODEL`，校验 submodule，加入 `3rdparty/VfSimulator/native`。 |
| `lib/PTO/Transforms/CMakeLists.txt` | 将 `vfsim::native_core` 和 `vfsim::ir_planner` 链接进 `PTOTransforms`。 |

VfSim native 侧主要 target：

```text
vfsim::native_core
vfsim::ir_planner
```

## 编译链路

```text
PreFusionAnalysis
  -> FusionPlan
       - PTOAS 生成合法 fusion group
       - PTOAS 写入 pto.fusion.group_id / pto.fusion.order
       - 启用 --enable-vfsim-costmodel-optimization 时调用 VfSim planner
       - VfSim 写回 pto.fusion.row_unroll_factor / pto.fusion.col_unroll_factor
  -> OpScheduling
  -> FusionRegionGen
       - 将 tileop group 包成 pto.fusion_region
       - 将一致的 row/col unroll attrs 提升到 pto.fusion_region
  -> ExpandTileOp / Inline / shape-only fold
  -> PTOLowLevelLoopFusion
  -> PTOUnrollAfterLoopFusion
       - VPTO 路径下消费 row/col unroll attrs
       - 成功消费后将对应 factor 复位为 1
  -> FlattenFusionRegion
```

EmitC 路径可以运行 FusionPlan 和 VfSim planner，但当前 unroll attrs 只由 VPTO
后端的 `PTOUnrollAfterLoopFusion` 消费。

## PTOAS 调用点

| 文件 | 符号 | 作用 |
|---|---|---|
| `include/PTO/Transforms/Passes.td` | `FusionPlan` options | 定义 `enableVfSimCostmodelOptimization` 和 `dumpVfSimUnrollTest` pass options。 |
| `tools/ptoas/ptoas.cpp` | `enableVfSimCostmodelOptimization` | 定义用户侧 `--enable-vfsim-costmodel-optimization`。 |
| `tools/ptoas/ptoas.cpp` | `compilePTOASModule` | 将用户选项传入 `FusionPlanOptions`，并在 VPTO 路径下插入 unroll consumer。 |
| `lib/PTO/Transforms/TileFusion/PTOFusionPlan.cpp` | `runVfSimFusionPlanner` | 调用 `vfsim::planTileFusionIR`。 |
| `lib/PTO/Transforms/TileFusion/PTOFusionRegionGen.cpp` | `getCommonSpanI64Attr` | 校验并提升 row/col unroll attrs 到 `pto.fusion_region`。 |
| `lib/PTO/Transforms/TileFusion/PTOUnrollAfterLoopFusion.cpp` | `PTOUnrollAfterLoopFusion` | 消费 `pto.fusion_region` 上的 row/col unroll attrs。 |

## VfSim C++ API

VfSim 向 PTOAS 暴露源码级 C++ API：

```cpp
namespace vfsim {

struct PlannerOptions {
  bool dumpCandidates = false;
  unsigned maxUnroll = 8;
};

mlir::LogicalResult planTileFusionIR(
    mlir::Operation *candidateIR,
    const PlannerOptions &options = {});

} // namespace vfsim
```

接口约定：

| 项 | 约定 |
|---|---|
| 输入 IR | FusionPlan 后的 MLIR operation；当前自动 tileop fusion 路线传入 `func::FuncOp`。 |
| 输入内容 | PTOAS 已写好 `pto.fusion.group_id` 和 `pto.fusion.order` 的 tileop-level IR。 |
| 输出方式 | VfSim 原地写回 `pto.fusion.row_unroll_factor` / `pto.fusion.col_unroll_factor`。 |
| 返回值 | `success()` 表示 planner 完成、没有可处理 group，或某些 group 被 warning 降级跳过；`failure()` 表示接口级错误。 |
| 修改范围 | Planner 只能写 attrs，不允许增删、替换、移动 op，也不允许修改 operand/result/type。 |

## 输入 IR

PTOAS 传给 VfSim 的 IR 必须已经包含：

| 属性 | 含义 |
|---|---|
| `pto.fusion.group_id` | PTOAS 已选择的 fusion group ID。 |
| `pto.fusion.order` | group 内 tileop 的执行顺序。 |

VfSim 从 IR 中读取：

| 信息 | 来源 |
|---|---|
| group 边界 | `pto.fusion.group_id` |
| group 内顺序 | `pto.fusion.order` |
| tileop 类型 | MLIR op name，例如 `pto.tadd` |
| 数据依赖 | SSA use-def |
| 输入输出 value | tileop operands |
| dtype | operand/result type |
| shape | tile buffer type |
| 模板参数 | tileop attrs |

示例：

```mlir
pto.tadd ins(%b, %c : !pto.tile_buf<vec, 32x128xf32>,
                      !pto.tile_buf<vec, 32x128xf32>)
         outs(%a : !pto.tile_buf<vec, 32x128xf32>)
         {pto.fusion.group_id = 0 : i64,
          pto.fusion.order = 0 : i64}

pto.tmul ins(%a, %b : !pto.tile_buf<vec, 32x128xf32>,
                      !pto.tile_buf<vec, 32x128xf32>)
         outs(%d : !pto.tile_buf<vec, 32x128xf32>)
         {pto.fusion.group_id = 0 : i64,
          pto.fusion.order = 1 : i64}
```

VfSim 不重新判断这组 tileop 是否可以融合，只基于 PTOAS 已选 group 生成优化决策。

## 输出 IR

VfSim 输出仍然是同一份 tileop-level IR，通过 attrs 表示优化计划：

| 属性 | 含义 |
|---|---|
| `pto.fusion.row_unroll_factor` | 当 row loop 是实际最内层 loop 时使用的 unroll factor。 |
| `pto.fusion.col_unroll_factor` | 当 col loop 是实际最内层 loop 时使用的 unroll factor。 |

当前 unroll 语义：

| 情况 | VfSim 输出 |
|---|---|
| col trip count 为 1 | `row_unroll_factor > 1`，`col_unroll_factor = 1` |
| col trip count 大于 1 | `row_unroll_factor = 1`，`col_unroll_factor > 1` |

示例：

```mlir
pto.tadd ... {
  pto.fusion.group_id = 0 : i64,
  pto.fusion.order = 0 : i64,
  pto.fusion.row_unroll_factor = 1 : i64,
  pto.fusion.col_unroll_factor = 2 : i64
}

pto.tmul ... {
  pto.fusion.group_id = 0 : i64,
  pto.fusion.order = 1 : i64,
  pto.fusion.row_unroll_factor = 1 : i64,
  pto.fusion.col_unroll_factor = 2 : i64
}
```

## RegionGen 属性提升

`FusionRegionGen` 会把同一个 group 包成 `pto.fusion_region`，并将一致的
row/col unroll attrs 从 tileop 提升到 region：

```mlir
%0 = pto.fusion_region {
  pto.tadd ...
  pto.tmul ...
  pto.yield(%d) : (!pto.tile_buf<vec, 32x128xf32>) -> ()
} {
  pto.fusion.group_id = 0 : i64,
  pto.fusion.row_unroll_factor = 1 : i64,
  pto.fusion.col_unroll_factor = 2 : i64
} : !pto.tile_buf<vec, 32x128xf32>
```

提升规则：

- 同一个 group 内，某个 unroll attr 要么所有成员都没有，要么所有成员都有。
- 如果所有成员都有，值必须一致。
- factor 必须是正整数。
- 部分成员有、部分成员没有，或者值不一致，会报错。

## VPTO Unroll 消费

`PTOUnrollAfterLoopFusion` 在 VPTO low-level loop fusion 后运行。它读取
`pto.fusion_region` 上的：

```text
pto.fusion.row_unroll_factor
pto.fusion.col_unroll_factor
```

消费规则：

- 只展开当前最内层 `scf.for`。
- 只处理常量 trip count，且 trip count 必须能被 factor 整除。
- 当前约定下，col loop 存在时消费 `col_unroll_factor`；col loop 已被折叠后，
  row loop 成为最内层时消费 `row_unroll_factor`。
- 成功消费某个 factor 后，将该 region 上对应 attr 复位为 `1`，避免同一 factor
  在后续 greedy/walk 过程中被重复应用。

示意：

```mlir
// Before PTOUnrollAfterLoopFusion
pto.fusion_region {
  scf.for %row = %c0 to %c32 step %c1 {
    scf.for %col = %c0 to %c2 step %c1 {
      pto.vadd ...
      pto.vmul ...
    }
  }
} {
  pto.fusion.row_unroll_factor = 1 : i64,
  pto.fusion.col_unroll_factor = 2 : i64
}

// After PTOUnrollAfterLoopFusion
pto.fusion_region {
  scf.for %row = %c0 to %c32 step %c1 {
    pto.vadd ...
    pto.vmul ...
    pto.vadd ...
    pto.vmul ...
  }
} {
  pto.fusion.row_unroll_factor = 1 : i64,
  pto.fusion.col_unroll_factor = 1 : i64
}
```

## VfSim Planner 行为

VfSim native planner 位于 `3rdparty/VfSimulator/native`。

| 文件 | 作用 |
|---|---|
| `IRPlanner.h` | 定义 `PlannerOptions` 和 `planTileFusionIR`。 |
| `IRPlanner.cpp` | 收集 fusion group，枚举 unroll candidate，调用 native VfInfo simulator，写回 attrs。 |
| `TileOpTemplates.h/cpp` | 将 PTOAS tileop group lower 成 VfSim `VfInfo` micro-op program。 |
| `ParamDB.h/cpp` | 加载 ISA/uarch/forwarding/initiation-interval 参数。 |

候选枚举规则：

- 搜索范围是 `1..maxUnroll`。
- 当前默认 `maxUnroll = 8`。
- 只考虑能够整除目标 loop trip count 的 factor。

降级诊断：

| 场景 | 行为 |
|---|---|
| 参数表加载失败 | 打印 warning，planner 对本次编译降级，不写 unroll attrs，PTOAS 继续编译。 |
| group 中存在不支持的 tileop/template | 打印 warning，跳过该 group。 |
| micro-op/dtype 参数缺失或所有 candidate 仿真失败 | 打印 warning，跳过该 group。 |

`--dump-vfsim-unroll-test` 只额外打印候选值预测结果，例如：

```text
unroll=1 trip=2 dtype=fp32 cycles=278
unroll=2 trip=2 dtype=fp32 cycles=131
```

## 当前模板覆盖范围

当前源码级 planner 主要覆盖 elementwise 风格 tileop。

| TileOp | Micro-op |
|---|---|
| `tadd` | `VADD` |
| `tsub` | `VSUB` |
| `tmul` | `VMUL` |
| `tdiv` | `VDIV` |
| `tmax` | `VMAX` |
| `tmin` | `VMIN` |
| `tabs` | `VABS` |
| `texp` | `VEXP` |
| `tadds` | `VADDS` |
| `tsubs` | `VSUB`，标量通过 `VBR` 转成向量 |
| `tmuls` | `VMULS` |
| `tdivs` | `VDIV`，标量通过 `VBR` 转成向量 |
| `tmaxs` | `VMAXS` |
| `tmins` | `VMINS` |
| `tcvt` | `VCVT_F16_TO_F32` 或 `VCVT_F32_TO_F16` |

模板选择原则：

- 根据 op name 选择 tileop 模板族。
- 根据 dtype 选择 micro-op 参数和 vector lanes。
- 根据 shape / valid shape 推导 row/col loop 结构。
- 模板语义需要与 VPTO 后端 tileop template 保持一致。

## 手写 VF IR 扩展方向

除自动 tileop fusion 路线外，后续也可以支持开发者直接提供 VF/micro-op IR。
该路线仍可复用源码级 submodule 接入方式，但输入不再是 tileop group，而是开发者
已经写好的 VF IR。

```text
developer VF IR
  -> VfSim analyzer
  -> VF IR with cost/advice attrs or diagnostic report
```

该路线不决定 fusion group，而是评估已有 VF 实现，并给出预测和优化建议。
