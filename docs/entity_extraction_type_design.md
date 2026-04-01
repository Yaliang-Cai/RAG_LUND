# 实体类型体系设计分析

## 背景

LightRAG 默认使用 11 个通用实体类型，在知识图谱构建中存在类型粒度不均、分类不一致等问题，导致实体消歧效果差。本文档分析现有问题并给出改进方案。

---

## 一、当前 11 个类型的设计问题

当前类型分布严重不均衡：

```
社会维度（2个）: Person, Organization          → 清晰
时空维度（2个）: Location, Event               → 清晰
物理维度（2个）: Creature, NaturalObject       → 过于 nature-domain
信息维度（5个）: Concept, Method, Content, Data, Artifact  → 互相重叠，全是垃圾桶
```

`Concept / Method / Content / Data / Artifact` 这 5 个类型对 LLM 来说几乎不可区分：

- "Machine Learning" → Concept? Method?
- "BERT model weights" → Data? Artifact?
- "Research paper" → Content? Artifact?

**根本原因**：类型没有按照清晰的**本体论维度**划分，导致 LLM 分类不一致。

---

## 二、好的类型系统的设计原则

1. **正交性**：每个类型占据不同的语义空间，LLM 容易做出一致选择
2. **完备性**：所有实体都能找到合适的类型，极少需要 fallback 到 `Other`
3. **对称性**：各维度类型数量均衡，不要让某一维度堆积过多类型

---

## 三、推荐的通用类型表（12 个）

```python
ENTITY_TYPES = [
    # 能动主体（WHO does things）
    "Person",           # 自然人
    "Organization",     # 公司、机构、团队、政府

    # 时空（WHERE / WHEN）
    "Location",         # 地理/空间实体
    "Event",            # 有时间边界的事件、活动、事故

    # 人造物（human-made THINGS）
    "Artifact",         # 物理制品：设备、产品、建筑
    "Work",             # 知识/创意产出：论文、软件、数据集、标准、法规

    # 自然物（natural THINGS）
    "NaturalEntity",    # 自然现象、生物、元素、天体、物质

    # 抽象（IDEAS）
    "Concept",          # 理论、原则、现象、抽象思想
    "Process",          # 方法、流程、算法、技术（有步骤的"怎么做"）

    # 量化（QUANTITIES）
    "Measure",          # 指标、度量、统计量、数值+单位

    # 兜底（2个，而非5个）
    "Role",             # 职位、头衔、身份（不是 Person，但代表人的角色）
    "Other",            # 真正无法归类时才用
]
```

---

## 四、与当前类型的对比映射

| 当前类型 | 问题 | 替换为 |
|----------|------|--------|
| Concept | 吞掉所有抽象词 | Concept（理论/原则）+ Process（方法/流程）拆分 |
| Method | 和 Concept 难区分 | → Process |
| Content | 和 Artifact/Data 难区分 | → Work |
| Data | 和 Content 难区分 | → Work（数据集）或 Measure（数值） |
| Artifact | 物理 vs 知识产物混用 | → Artifact（物理）+ Work（知识） |
| Creature | 太 nature-specific | → NaturalEntity（合并动植物、元素、天体） |
| NaturalObject | 和 Creature 难区分 | → NaturalEntity |

---

## 五、关键区分说明

### `Concept` vs `Process`（最重要的改进）

| 类型 | 定义 | 示例 |
|------|------|------|
| `Concept` | 是什么（理论、原则、现象） | "Attention Mechanism", "Capitalism", "Entropy" |
| `Process` | 怎么做（步骤、算法、流程） | "Gradient Descent", "PCR Protocol", "Agile" |

这个区分让 LLM 有清晰依据，不再全部塞进模糊的 `Method` 或 `Concept`。

### `Artifact` vs `Work`

| 类型 | 定义 | 示例 |
|------|------|------|
| `Artifact` | 物理制品 | "GPU A100", "iPhone", "Eiffel Tower" |
| `Work` | 知识/创意产出 | "BERT paper", "ImageNet dataset", "ISO 9001 standard", "Linux kernel" |

统一了原来 `Content / Data / Artifact（知识层面）` 三个混乱的类型。

### `NaturalEntity`（合并两个 nature 类型）

- "Python snake"、"Carbon element"、"Jupiter planet" 全是 `NaturalEntity`
- 和 `Process`（Python 语言 → `Process`）区分清晰
- 同时解决了原来 `Creature` vs `NaturalObject` 难以区分的问题

---

## 六、需要同步修改的地方

仅替换类型名称不够，还需要以下配套改动：

### 6.1 在 Prompt 中为每个类型添加一行定义

当前 `entity_extraction_system_prompt` 只列出类型名，没有说明。需要在 `<Entity_types>` 部分改为：

```
- Person: individual humans
- Organization: companies, institutions, teams, governments
- Location: geographic or spatial entities
- Event: time-bounded occurrences, incidents, proceedings
- Artifact: physical human-made objects (devices, products, buildings)
- Work: intellectual/creative outputs (papers, software, datasets, standards, regulations)
- NaturalEntity: natural phenomena, organisms, chemical elements, celestial bodies, materials
- Concept: abstract ideas, theories, principles (WHAT something is)
- Process: methods, algorithms, procedures, workflows (HOW things are done)
- Measure: metrics, statistics, quantities with units or context
- Role: job titles, positions, identities (not a person, but represents a person's role)
- Other: only when truly no other type applies
```

### 6.2 更新 few-shot examples

`prompt.py` 中的 3 个 few-shot 示例使用旧类型，会对 LLM 产生误导。需要用新类型重新标注示例。

### 6.3 限制 `Other` 的使用描述

将 prompt 中：
```
If none of the provided entity types apply, do not add new entity type and classify it as `Other`.
```
改为：
```
If none of the provided entity types precisely apply, select the CLOSEST matching type.
Do NOT use `Other` unless the entity is truly ambiguous — in which case, consider
whether it should be extracted at all.
```

### 6.4 配置方式

通过环境变量或初始化参数覆盖默认类型（无需改代码）：

```python
# 环境变量
ENTITY_TYPES=Person,Organization,Location,Event,Artifact,Work,NaturalEntity,Concept,Process,Measure,Role,Other

# 或在 LightRAG 初始化时
rag = LightRAG(
    addon_params={"entity_types": [
        "Person", "Organization", "Location", "Event",
        "Artifact", "Work", "NaturalEntity",
        "Concept", "Process", "Measure",
        "Role", "Other",
    ]}
)
```

---

## 七、对 V1 实体消歧的影响

改进类型体系后，V1 的 `name|type` 复合键能区分更多 homonym 场景：

| 实体名 | 旧类型（无法区分） | 新类型（可区分） |
|--------|-------------------|-----------------|
| Python | Method / Concept | Process（语言）vs NaturalEntity（蛇） |
| Transformer | Method / Artifact | Process（NLP模型）vs Artifact（变压器） |
| Mercury | NaturalObject / Concept | NaturalEntity（元素/行星）区分更清晰 |
| Apollo | Event / Concept | Event（登月计划）vs Concept（神话人物→Person） |

类型粒度的提升是 V1 消歧效果的基础，类型越准确，`name|type` 键的区分度越高。
