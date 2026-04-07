# /query 与 /query/stream 设计分析

## 概述

系统一（`rag-anything/server/app.py`）有两条查询路径，由路由决定，不可在单个 endpoint 内切换。

---

## 两条路径对比

| | `/query` | `/query/stream` |
|---|---|---|
| 调用链 | `aquery_data()` + `service.query()` | `stream_query()` → `aquery_llm(stream=True)` |
| 响应格式 | JSON `{answer, data, metadata, graph}` | SSE (`text/event-stream`)，meta/chunk/done 事件 |
| VLM 增强 | ✅ 支持 | ❌ 不支持 |
| 结构化检索数据 | 独立 `aquery_data` 调用 | 来自 `aquery_llm` meta 事件 |
| 子图可视化 | ✅ 支持 | ✅ 支持（done 事件） |
| inline citation | ✅ 修复后支持（2026-04-07） | ✅ 支持 |
| PPR/multi-hop 透传 | ✅ 支持 | ✅ 修复后支持（2026-04-07） |

---

## 为什么没有 stream 参数开关

两条路径**不是同一操作的两种交付方式**，响应格式完全不同，因此无法合并为带 `stream: bool` 参数的单一 endpoint。

- `/query` 返回一次性完整 JSON，包含 `aquery_data` 的独立结构化检索数据
- `/query/stream` 返回 SSE 事件流，检索数据通过 meta 事件推送

这两条路径是**分别演进**出来的，没有统一维护约定，导致功能漂移。

---

## 历史问题与修复记录

### 问题 1：inline citation 仅在 stream 路径生效（已修复）

`_INLINE_CITATION_INSTRUCTION` 只在 `stream_query()`（`local_rag.py:1554`）注入，
`service.query()` 没有传 `user_prompt`，导致 `/query` 路径 LLM 不带 citation 指令。

**修复**：在 `service.query()` 的 `normalized_kwargs` 中补加 `user_prompt` setdefault。

### 问题 2：PPR/multi-hop 参数在 stream 路径不透传（已修复）

`stream_query()` 签名缺少 `enable_multi_hop`、`multi_hop_depth`、`ppr_damping`、`ppr_top_k`，
`QueryParam` 硬读 `self.settings`，payload 中的值被静默忽略。

**修复**：
1. `stream_query()` 签名增加 4 个参数（`None` 为默认，fallback 到 `self.settings`）
2. `app.py` stream endpoint 调用处补传 payload 字段

### 问题 3：VLM 增强在 stream 路径缺失（未修复）

`/query` 支持 `vlm_enhanced=payload.vlm_enhanced`，`/query/stream` 不支持。
`stream_query()` 使用 `aquery_llm()` 直接调用，没有 VLM 图像注入路径。
这是较大的功能补齐，需要单独实现。

### 问题 4：`/query` 双重检索开销（设计取舍，暂不修复）

`/query` 先调 `aquery_data()`（仅检索），再调 `service.query()`（检索 + LLM），
同一 query 的检索执行两次。
`aquery_data` 提供的独立结构化数据格式是有意保留的，改动需评估影响。

---

## 维护约定

每次在任一路径新增功能时，需同步检查另一条路径是否需要对齐：

- [ ] 新的 QueryParam 字段 → 检查 `stream_query()` 签名和 `app.py` 两个 endpoint 的调用
- [ ] 新的 LLM 指令（`user_prompt` 等）→ 检查 `service.query()` 和 `stream_query()` 是否都注入
- [ ] 新的检索参数 → 检查两条路径的 `QueryParam` 构造
