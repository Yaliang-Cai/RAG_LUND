# 查询期同义边硬过滤开关（2026-04-20）

## 目标
- 在一份包含 `FACTUAL + SYNONYM` 的图谱上，通过查询参数切换“是否使用同义边”。
- 不重建图谱即可得到两种检索视图：
  - 包含同义边传播；
  - 排除同义边（近似 `v0_v1`）。

## 开关定义
- `QueryParam.exclude_synonym_edges: bool | None`
  - `True`：总是过滤 `SYNONYM` 边。
  - `False`：总是不过滤 `SYNONYM` 边。
  - `None`：自动模式（推荐默认）。
- `RAGANYTHING_EXCLUDE_SYNONYM_EDGES`（可选环境变量）
  - 未设置：保持 `None`（自动模式）。
  - 设置为 `true/false`：强制覆盖自动模式。

## 自动模式默认规则（None）
- PPR 查询（`mode=ppr` / `mode=ppr_local`，或 legacy `enable_multi_hop=true`）：默认 `False`（不排除同义边）。
- 非 PPR 查询（`local/global/hybrid/mix/rrf` 等）：默认 `True`（排除同义边）。

## 生效范围
- `local`：实体邻接关系候选中可过滤 `SYNONYM` 边。
- `global/hybrid`：关系候选中可过滤 `SYNONYM` 边。
- `ppr_local`：子图边可过滤 `SYNONYM` 边。
- `ppr`（global PPR）：邻接矩阵构建可过滤 `SYNONYM` 边。

## 兼容性
- 该开关是纯查询期逻辑，不修改已有图库与向量库内容。
- 显式传 `True/False` 时，始终优先于自动模式。
