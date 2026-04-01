# 参数默认值流向说明

本文档解释 `raganything/constants.py` 与 `lightrag/lightrag/constants.py` 中的同名常量
各自在什么时候生效，是否存在互相覆盖。

---

## 核心结论（先读这个）

> **两个 constants.py 是完全独立的 Python 文件，不会自动互相覆盖。**
>
> 但 LightRAG 对象有自己的一批字段默认值，这些默认值来自 `lightrag/constants.py`。
> 当你创建 LightRAG 对象时，如果某个参数**没有显式传入**，就会用 LightRAG 自己的默认值。

---

## 参数如何一步步传递

```
你运行 local_rag.py / server
        │
        ▼
LocalRagService.__init__()
  读取 raganything/constants.py 中的值 → 存入 self.settings
        │
        ▼
LocalRagService._build_rag()
  用 self.settings 构造 lightrag_kwargs 字典
  例如: {"min_rerank_score": 0.3, "max_parallel_insert": 4, ...}
        │
        ▼
RAGAnything(lightrag_kwargs=...)     ← __post_init__ 只做基础设置，不初始化 LightRAG
        │
        ▼  (第一次调用 process_document 或 query 时触发)
RAGAnything._ensure_lightrag_initialized()
        │
        ▼
LightRAG(**lightrag_params)          ← 在这里 LightRAG 对象才被创建
  传入的参数 → 用 raganything/constants.py 的值  ✅
  没传的参数 → 用 lightrag/constants.py 的值    ⚠️
```

---

## 哪些参数"传了"，哪些"没传"

### 传了的参数（raganything 的值生效）

`_build_rag()` 在 `lightrag_kwargs` 里显式传入了这些，所以 LightRAG 的默认值被覆盖：

| 参数 | raganything 的值 | lightrag 默认值 | 结果 |
|---|---|---|---|
| `min_rerank_score` | **0.3** | 0.0 | 用 0.3 ✅ |
| `max_parallel_insert` | **4** | 2 | 用 4 ✅ |
| `embedding_batch_num` | **32** | 10 | 用 32 ✅ |
| `embedding_func_max_async` | **8** | 8 | 相同，无影响 |
| `llm_model_max_async` | **16** | 4 | 用 16 ✅ |
| `entity_extract_max_gleaning` | **1** | 1 | 相同，无影响 |
| `chunk_token_size` | **1200** | (base.py) | 用 1200 ✅ |
| `chunk_overlap_token_size` | **100** | (base.py) | 用 100 ✅ |

### 没传的参数（lightrag 的默认值生效）

这些参数 `lightrag_kwargs` 里没有，所以 LightRAG 用自己的默认值：

| 参数 | lightrag 默认值 | 说明 |
|---|---|---|
| `top_k` | **40** | 初始化时用 40，但查询时 QueryParam 会显式传入，所以不影响查询结果（见下方说明） |
| `chunk_top_k` | **20** | 同上 |
| `cosine_threshold` | **0.2** | 向量检索相似度阈值，lightrag 默认值生效 |
| `max_entity_tokens` | **6000** | context 中实体的 token 上限 |
| `max_relation_tokens` | **8000** | context 中关系的 token 上限 |
| `max_total_tokens` | **30000** | context 总 token 上限 |
| `related_chunk_number` | **5** | 每个实体/关系关联的 chunk 数 |

---

## `top_k` 和 `chunk_top_k` 为什么不影响查询结果

这两个参数在两个 constants.py 里值不同（raganything=20/10，lightrag=40/20），
但实际查询时走的是另一条路：

```
app.py 收到 /query 请求
   payload.top_k = 20 (来自 raganything 的 DEFAULT_TOP_K)
        │
        ▼
QueryParam(top_k=20, chunk_top_k=10, ...)   ← 每次查询显式传入
        │
        ▼
rag.lightrag.aquery(param=QueryParam)       ← LightRAG 用 QueryParam 里的值，忽略构造时的默认值
```

所以 LightRAG 对象初始化时 `top_k=40` 这件事没有影响，查询时永远用的是 QueryParam 传入的值。

---

## `temperature` 去哪了

`raganything/constants.py` 中 `DEFAULT_TEMPERATURE = 0.0`，
`lightrag/constants.py` 中 `DEFAULT_TEMPERATURE = 1.0`。

这两个完全不冲突，因为 temperature 根本不是 LightRAG 的构造参数。
它被注入到 `llm_model_func` 的函数闭包里：

```python
# local_rag.py
async def llm_func(...):
    response = await client.chat.completions.create(
        temperature=self.settings.temperature,  # 始终是 0.0
        ...
    )
```

LightRAG 的 `DEFAULT_TEMPERATURE=1.0` 是给 LightRAG 官方 server 模式用的，
走 `local_rag.py` 时完全不会碰到它。

---

## 一句话总结

| 情况 | 用哪个 constants.py |
|---|---|
| `lightrag_kwargs` 显式传入的参数 | `raganything/constants.py` ✅ |
| `lightrag_kwargs` 没传、但查询时 QueryParam 显式传的 | `raganything/constants.py` ✅ |
| `lightrag_kwargs` 没传、也没在 QueryParam 传的 | `lightrag/constants.py` ⚠️ |
| `llm_model_func` 闭包内部的参数（如 temperature） | `raganything/constants.py` ✅ |
| 直接使用 LightRAG（不经过 RAGAnything） | `lightrag/constants.py` |

当前项目中"没传、也没在 QueryParam 传"的参数（`cosine_threshold`、`max_entity_tokens` 等）
在 `raganything/constants.py` 里根本没有对应定义，说明这些参数**有意**使用 LightRAG 的默认值，不是遗漏。
