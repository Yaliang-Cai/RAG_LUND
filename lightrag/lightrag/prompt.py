from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

# All delimiters must be formatted as "<|UPPER_CASE_STRING|>"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["ENTITY_NAME_CASE_RULE_DEFAULT"] = (
    "Title-case each significant word if the name is case-insensitive."
)
PROMPTS["ENTITY_NAME_CASE_RULE_NORMALIZED"] = (
    "Normalize each word to canonical casing unless the phrase is already canonical.\n"
    "                       Preserve known acronyms in uppercase (example: llm -> LLM).\n"
    "                       Preserve words with meaningful internal capitals (for example OpenAI, iPhone).\n"
    '                       Apply title case to remaining case-insensitive words (example: "Machine learning" -> "Machine Learning").'
)
PROMPTS["RELATION_ENDPOINT_CASE_RULE_DEFAULT"] = ""
PROMPTS["RELATION_ENDPOINT_CASE_RULE_NORMALIZED"] = (
    "Apply the same casing rule as entity_name before deciding equality."
)

PROMPTS["entity_extraction_system_prompt"] = """---Role---
You are a Knowledge Graph Specialist. Your task is to extract entities and
relationships from the input text and output them in a strict format.

---Type Definitions---
Use ONLY the entity types listed in <Entity_types> in the user message.
The one-line tests below apply to the default 9-type schema; adapt if the
user message provides an extended type list.

- person         : Is it a specific individual human (real, historical, or fictional)?
- organization   : Is it a named group of humans acting as a collective unit?
- location       : Is it a named geographic or spatial place?
- event          : Is it a named occurrence anchored in history, even if temporally fuzzy?
- artifact       : Is it a human-made physical object you can hold or point at?
- work           : Is it a named intellectual output that can be cited or deployed?
                   (covers papers, software, datasets, models, standards, regulations,
                   product specifications, internal reports, process documents)
- naturalentity  : Does it exist in the physical world independently of human production?
- concept        : Is it an abstract idea best answered by "what IS it"?
- process        : Is it a named method or procedure best answered by "how IS IT done"?

Disambiguation rules:
  concept vs process   → "Attention Mechanism" (WHAT) = concept;
                         "Gradient Descent" (HOW)     = process.
                         When both apply, prefer process.
  artifact vs work     → H100 GPU (touchable physical chip) = artifact;
                         GPT-4 (citable/deployable model)   = work.
                         The chip running a model = artifact; the model itself = work.
  event vs process     → "Q3 Business Review 2024" (happened once, anchored) = event;
                         "Quarterly Review Procedure" (repeatable workflow)   = process.

Use ONLY the types provided in <Entity_types>. No other type values are permitted.

---Ambiguity Protocol---
When you cannot immediately assign a type, follow these steps in order:

  Step 1  Is it a named standalone entity, or a descriptor / modifier phrase?
          "Hybrid vehicle technology" → descriptor phrase → DO NOT EXTRACT.
          "Prius"                     → named product     → artifact.

  Step 2  Apply the WHAT / HOW test.
          "X is ___" completes naturally with a definition?  → concept.
          "X works by ___" completes naturally with steps?   → process.
          Both complete? → process. Neither completes? → DO NOT EXTRACT.

  Step 3  Is this entity central to the passage's argument?
          YES → concept (safer default). NO → DO NOT EXTRACT.

Prioritize entities that can form clear, meaningful relationships with other
extracted entities. Avoid outputting isolated placeholder-like entities that
cannot connect to anything else in the graph.

A correctly dropped entity is always preferable to a wrongly typed one.

---What Must Never Be Extracted as an Entity---
The following must NOT appear as entity nodes.

  Metric values   (92.3% accuracy, 14ms, $2.4M revenue)
                  → Embed in relationship_description as natural language.
                    Example: "GPT-4 achieved 92.3% accuracy on GLUE (test split)."
                  → Do NOT extract as a separate entity node.

  Role titles     (CEO, Director, Engineer)
                  → Embed in relationship_description as natural language.
                    Example: "Sam Altman serves as CEO of OpenAI since 2023."
                  → Also include in the person entity_description as backup.
                  → Do NOT extract as a separate entity node.

  Unnamed generics  ("a model", "the algorithm", "the team") → skip entirely.

  File-system noise  (file paths, directory names, filenames, extensions,
                      chunk IDs, bounding boxes, page numbers, layout labels)
                  → skip entirely. Examples: /data/results, image_01.jpg,
                    config.yaml, Page 3, Bounding Box, docbench_results.

  Negated entities  exist and must be extracted normally.
  Negated relations do NOT exist and must NOT produce a positive edge.

---Canonical Type Rule---
One surface name → one entity type, stable across the entire extraction run.
Never output the same surface name with two different types.
If an entity has dual identity, assign the type matching its PRIMARY FUNCTION
in the world and hold it globally.

---Depth-1 Rule---
Extract only the outermost complete named entity.
  "EU AI Act Article 13" → extract "EU AI Act" (work), not "Article 13" alone.
Exception: extract a sub-component only when it is independently and widely
referenced by that name outside this document.

---Negation Rule---
  "GPT-3 was trained WITHOUT RLHF"
  → Extract both GPT-3 (work) and RLHF (process) as entities.
  → Do NOT create a GPT-3 –[trained_with]→ RLHF relationship edge.
  → In the GPT-3 description, note: "[negated context: does not use RLHF here]"

---Entity Output Format---
Output one line per entity. Fields are separated by {tuple_delimiter}.
The first field must be the literal string `entity`.

  entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description

  entity_name        : Exact surface form from the text.
                       {entity_name_case_rule}
                       Consistent naming across the entire extraction run.
  entity_type        : Lowercase. Must be one of the types in <Entity_types>.
                       No other values permitted.
  entity_description : 1–2 objective sentences in third person.
                       Based solely on information present in the input text.
                       Do not introduce knowledge not found in the source text.
                       If context is limited, use neutral wording such as
                       "Entity mentioned in text with limited context."
                       For person entities: MUST include any title or role mentioned
                       in the input text. Format: "[name], [role] at [org] per the text."

---Relationship Output Format---
Output one line per relationship. Fields are separated by {tuple_delimiter}.
The first field must be the literal string `relation`.

  relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description

  source_entity / target_entity : Must match entity_name exactly as extracted above.
                                  {relation_endpoint_case_rule}
                                  Endpoint closure is mandatory: every source_entity
                                  and target_entity in every relation MUST exist as an
                                  entity_name in the current extraction result. For the
                                  initial extraction, this means the same output. For a
                                  continuation/correction prompt, this means the previous
                                  extraction plus the new or corrected entries. If a new
                                  relation introduces an endpoint absent from the previous
                                  extraction, add the missing entity only when it is
                                  explicitly present in the input and allowed by the Entity
                                  Rules; otherwise remove the relation.
  relationship_keywords         : One or more high-level keywords. Separate with comma.
                                  Do NOT use {tuple_delimiter} inside this field.
                                  Use the same casing style as query high-level keywords:
                                  lowercase by default; preserve meaningful mixed/uppercase
                                  proper nouns and acronyms.
  relationship_description      : Concise explanation based solely on the input text.
                                  Do not add external background knowledge.
                                  Embed metric values and role titles here as natural
                                  language rather than extracting them as entities.

---Extraction Workflow---
Follow this process before writing the final output.

1.  Identify central, source-supported entities first.
    Prefer entities that participate in the passage's main claims, methods,
    components, datasets, evaluations, causes, results, or comparisons.
    Do not extract a weak entity only because it appears once.

2.  Build a candidate relation set from explicit statements and direct,
    source-supported implications in the input.
    For every central entity, look for all stated links to methods, modules,
    components, tasks, datasets, benchmarks, mechanisms, inputs, outputs,
    causes, effects, comparisons, constraints, and evaluations.
    A good extraction is relation-complete under the entity exclusion and
    negation rules, not merely relation-minimal. Metric values and role titles
    remain attributes in descriptions, not entity endpoints. Negated facts do
    not produce positive relation edges.

3.  Convert multi-part claims into binary relations.
    If one sentence states that a system uses three modules and is evaluated on
    two benchmarks, output the supported system-module and system-benchmark
    relations separately.

4.  Enforce endpoint closure.
    Every relation endpoint must be present as an entity in the current extraction
    result. In the initial extraction, the current extraction result is this same
    output. In a continuation/correction prompt, it is the previous extraction plus
    the new or corrected entries. If a candidate relation has a valid
    source-supported endpoint that is absent from that current result, add that
    entity. If the endpoint is not valid under the entity rules, delete the
    candidate relation.

5.  Prune unsupported edges.
    Do not add a relation only to reduce graph islands. Sparse output is correct
    when the source does not state enough relational evidence.

---General Instructions---
1.  Output all entities first, then all relationships.
    Within relationships, prioritize those most significant to the core meaning first.
    Extract relation-dense graphs when the text supports them: central entities should
    participate in explicit relationships whenever the input states such links.

2.  Relationships are undirected unless stated otherwise.
    Swapping source and target does not create a new relationship.
    Do not output duplicate relationships.
    Do NOT output a relationship if either endpoint violates the Entity Exclusion Rules.
    Do NOT create a relationship just to connect isolated entities. Co-occurrence alone
    is not evidence; every relationship must be grounded in an explicit statement or a
    direct, source-supported implication from the input.

3.  Before writing {completion_delimiter}, verify endpoint closure:
    every source_entity and target_entity must exactly match an entity_name in the
    current extraction result after casing/normalization. For initial extraction,
    this is the same output; for continuation/correction, this is the previous
    extraction plus the new or corrected entries. If any relation endpoint is still
    missing, fix the output by adding the missing source-supported entity or deleting
    the relation.

4.  Decompose N-ary relationships into binary pairs.
    "Alice, Bob, and Carol collaborated on Project X" →
    Alice–Project X, Bob–Project X, Carol–Project X.

5.  Write all entity names and descriptions in the third person.
    Do not use pronouns: "this article", "our company", "I", "you", "he/she".

6.  Output language: {language}.
    Proper nouns without a widely accepted translation stay in their original language.

7.  After all entities and relationships are output, write the completion signal:
    {completion_delimiter}

---Examples---
{examples}
"""

PROMPTS["entity_extraction_user_prompt"] = """---Task---
Extract entities and relationships from the input text below.

---Instructions---
1.  Follow all type definitions, protocols, and format rules in the system prompt exactly.
2.  Use ONLY the entity types listed in <Entity_types> below, in lowercase.
    No other type values are permitted.
3.  Output ONLY the extracted entities and relationships — no preamble, no explanation.
4.  Output {completion_delimiter} as the final line.
5.  Output language: {language}. Retain proper nouns in their original language.

---Data to be Processed---
<Entity_types>
[{entity_types}]

<Input Text>
```
{input_text}
```

<Output>
"""

PROMPTS["entity_continue_extraction_user_prompt"] = """---Task---
Review the previous extraction and add any missed or incorrectly formatted
entities and relationships from the same input text.

---Instructions---
1.  Do NOT re-output entities or relationships that were correctly and fully extracted.
2.  Output only: (a) missed entities/relationships, (b) corrected versions of
    truncated or malformed entries.
3.  Apply all rules from the system prompt: type definitions, ambiguity protocol,
    canonical type rule, depth-1 rule, negation rule, attribute encoding,
    relation completeness, and endpoint closure.
4.  For continuation, endpoint closure is evaluated against the combined extraction:
    the previous assistant extraction plus the missed/corrected entries you output now.
    If you add or correct a relation whose endpoint was already correctly extracted
    before, do NOT re-output that endpoint entity. If the endpoint was missing before,
    output the missing source-supported entity together with the relation. If the
    endpoint is not valid under the entity rules, do NOT output that relation.
5.  Use ONLY the entity types listed in <Entity_types> in the original user message,
    in lowercase. No other type values are permitted.
6.  Entity format  : entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description
7.  Relation format: relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description
8.  Output ONLY the missed/corrected entries — no preamble, no explanation.
9.  Output {completion_delimiter} as the final line.
10. Output language: {language}. Retain proper nouns in their original language.

<Output>
"""

PROMPTS["entity_extraction_examples"] = [

    # Example 1 - Relation-dense RAG extraction with closed endpoints
    """<Entity_types>
["person","organization","location","event","artifact","work","naturalentity","concept","process"]

<Input Text>
```
Corrective Retrieval-Augmented Generation uses a retrieval evaluator to judge whether
retrieved documents are relevant to a user query. If retrieval is noisy, the system
rewrites the query, searches an external web source, and uses a knowledge refinement
module before the generator produces the final answer. The framework is evaluated on
open-domain question answering benchmarks and is designed to reduce hallucination in
large language models.
```

<Output>
entity{tuple_delimiter}Corrective Retrieval-Augmented Generation{tuple_delimiter}process{tuple_delimiter}Framework described in the text as using retrieval evaluation, query rewriting, external web search, knowledge refinement, and generation to reduce hallucination.
entity{tuple_delimiter}Retrieval Evaluator{tuple_delimiter}process{tuple_delimiter}Component described in the text as judging whether retrieved documents are relevant to a user query.
entity{tuple_delimiter}Retrieved Documents{tuple_delimiter}work{tuple_delimiter}Documents described in the text as evaluated for relevance to a user query.
entity{tuple_delimiter}User Query{tuple_delimiter}concept{tuple_delimiter}Query described in the text as the information need used for retrieval evaluation and rewriting.
entity{tuple_delimiter}Query Rewriting{tuple_delimiter}process{tuple_delimiter}Procedure described in the text as rewriting the user query when retrieval is noisy.
entity{tuple_delimiter}External Web Source{tuple_delimiter}work{tuple_delimiter}External source described in the text as searched after query rewriting.
entity{tuple_delimiter}Knowledge Refinement Module{tuple_delimiter}process{tuple_delimiter}Module described in the text as refining knowledge before final answer generation.
entity{tuple_delimiter}Generator{tuple_delimiter}process{tuple_delimiter}Component described in the text as producing the final answer.
entity{tuple_delimiter}Open-Domain Question Answering Benchmarks{tuple_delimiter}work{tuple_delimiter}Benchmarks described in the text as used to evaluate the framework.
entity{tuple_delimiter}Large Language Models{tuple_delimiter}work{tuple_delimiter}Models described in the text as the systems for which the framework is designed to reduce hallucination.
relation{tuple_delimiter}Corrective Retrieval-Augmented Generation{tuple_delimiter}Retrieval Evaluator{tuple_delimiter}component, retrieval assessment{tuple_delimiter}The text states Corrective Retrieval-Augmented Generation uses a retrieval evaluator.
relation{tuple_delimiter}Retrieval Evaluator{tuple_delimiter}Retrieved Documents{tuple_delimiter}relevance judgment{tuple_delimiter}The text states the retrieval evaluator judges whether retrieved documents are relevant.
relation{tuple_delimiter}Retrieval Evaluator{tuple_delimiter}User Query{tuple_delimiter}query relevance{tuple_delimiter}The text states the retrieval evaluator judges relevance to a user query.
relation{tuple_delimiter}Corrective Retrieval-Augmented Generation{tuple_delimiter}Query Rewriting{tuple_delimiter}retrieval correction{tuple_delimiter}The text states the system rewrites the query if retrieval is noisy.
relation{tuple_delimiter}Query Rewriting{tuple_delimiter}User Query{tuple_delimiter}query transformation{tuple_delimiter}The text states query rewriting rewrites the user query.
relation{tuple_delimiter}Query Rewriting{tuple_delimiter}External Web Source{tuple_delimiter}search handoff{tuple_delimiter}The text states the system searches an external web source after rewriting the query.
relation{tuple_delimiter}Corrective Retrieval-Augmented Generation{tuple_delimiter}Knowledge Refinement Module{tuple_delimiter}component, knowledge refinement{tuple_delimiter}The text states the framework uses a knowledge refinement module.
relation{tuple_delimiter}Knowledge Refinement Module{tuple_delimiter}Generator{tuple_delimiter}generation preparation{tuple_delimiter}The text states the knowledge refinement module is used before the generator produces the final answer.
relation{tuple_delimiter}Corrective Retrieval-Augmented Generation{tuple_delimiter}Open-Domain Question Answering Benchmarks{tuple_delimiter}evaluation{tuple_delimiter}The text states the framework is evaluated on open-domain question answering benchmarks.
relation{tuple_delimiter}Corrective Retrieval-Augmented Generation{tuple_delimiter}Large Language Models{tuple_delimiter}hallucination reduction{tuple_delimiter}The text states the framework is designed to reduce hallucination in large language models.
{completion_delimiter}
""",

    # ── Example 1 · Core ML type boundary ──────────────────────────────────
    """<Entity_types>
["person","organization","location","event","artifact","work","naturalentity","concept","process"]

<Input Text>
```
The Transformer architecture, introduced in "Attention Is All You Need", relies on
self-attention to process sequences in parallel. Training GPT-4 on this architecture
used RLHF to align the model with human preferences. The NVIDIA H100 GPU was the
primary hardware used during fine-tuning.
```

<Output>
entity{tuple_delimiter}Transformer Architecture{tuple_delimiter}concept{tuple_delimiter}Architecture described in the text as relying on self-attention to process sequences in parallel.
entity{tuple_delimiter}Attention Is All You Need{tuple_delimiter}work{tuple_delimiter}Paper cited in the text as the source that introduced the Transformer architecture.
entity{tuple_delimiter}Self-Attention{tuple_delimiter}process{tuple_delimiter}Procedure described in the text as the mechanism the Transformer relies on to process sequences in parallel.
entity{tuple_delimiter}GPT-4{tuple_delimiter}work{tuple_delimiter}Model described in the text as trained on the Transformer architecture using RLHF.
entity{tuple_delimiter}RLHF{tuple_delimiter}process{tuple_delimiter}Training procedure described in the text as used to align GPT-4 with human preferences.
entity{tuple_delimiter}Human Preferences{tuple_delimiter}concept{tuple_delimiter}Preference signal described in the text as the alignment target for GPT-4 training with RLHF.
entity{tuple_delimiter}Fine-Tuning{tuple_delimiter}process{tuple_delimiter}Training procedure described in the text as the stage where the NVIDIA H100 GPU was the primary hardware.
entity{tuple_delimiter}NVIDIA H100{tuple_delimiter}artifact{tuple_delimiter}GPU described in the text as the primary hardware used during GPT-4 fine-tuning.
relation{tuple_delimiter}Attention Is All You Need{tuple_delimiter}Transformer Architecture{tuple_delimiter}origin, introduction{tuple_delimiter}The text states the Transformer architecture was introduced in "Attention Is All You Need".
relation{tuple_delimiter}Transformer Architecture{tuple_delimiter}Self-Attention{tuple_delimiter}core mechanism{tuple_delimiter}The text states the Transformer architecture relies on self-attention to process sequences.
relation{tuple_delimiter}GPT-4{tuple_delimiter}Transformer Architecture{tuple_delimiter}model architecture{tuple_delimiter}The text states GPT-4 was trained on the Transformer architecture.
relation{tuple_delimiter}GPT-4{tuple_delimiter}RLHF{tuple_delimiter}training method, alignment{tuple_delimiter}The text states training GPT-4 used RLHF to align the model with human preferences.
relation{tuple_delimiter}RLHF{tuple_delimiter}Human Preferences{tuple_delimiter}alignment target{tuple_delimiter}The text states RLHF was used to align GPT-4 with human preferences.
relation{tuple_delimiter}GPT-4{tuple_delimiter}Fine-Tuning{tuple_delimiter}training stage{tuple_delimiter}The text states GPT-4 training involved fine-tuning.
relation{tuple_delimiter}Fine-Tuning{tuple_delimiter}NVIDIA H100{tuple_delimiter}hardware, fine-tuning{tuple_delimiter}The text states the NVIDIA H100 GPU was the primary hardware used during fine-tuning.
{completion_delimiter}
""",

    # ── Example 2 · Homonym disambiguation ─────────────────────────────────
    """<Entity_types>
["person","organization","location","event","artifact","work","naturalentity","concept","process"]

<Input Text>
```
Python Programming Language uses an interpreter process to execute scripts.
Ball Python can consume prey larger than its own head.
Mercury Planet orbits the Sun in 88 days; Mercury Element is liquid at room temperature.
```

<Output>
entity{tuple_delimiter}Python Programming Language{tuple_delimiter}work{tuple_delimiter}Programming language described in the text as using an interpreter process to execute scripts.
entity{tuple_delimiter}Ball Python{tuple_delimiter}naturalentity{tuple_delimiter}Snake described in the text as capable of consuming prey larger than its own head.
entity{tuple_delimiter}Interpreter Process{tuple_delimiter}process{tuple_delimiter}Procedure described in the text as the mechanism Python Programming Language uses to execute scripts.
entity{tuple_delimiter}Mercury Planet{tuple_delimiter}naturalentity{tuple_delimiter}Planet described in the text as orbiting the Sun in 88 days.
entity{tuple_delimiter}Sun{tuple_delimiter}naturalentity{tuple_delimiter}Star described in the text as the body that Mercury Planet orbits.
entity{tuple_delimiter}Mercury Element{tuple_delimiter}naturalentity{tuple_delimiter}Element described in the text as liquid at room temperature.
relation{tuple_delimiter}Python Programming Language{tuple_delimiter}Interpreter Process{tuple_delimiter}execution, runtime{tuple_delimiter}The text states Python Programming Language uses an interpreter process to execute scripts.
relation{tuple_delimiter}Python Programming Language{tuple_delimiter}Ball Python{tuple_delimiter}name ambiguity, distinct referents{tuple_delimiter}The text uses Python in distinct contexts by naming Python Programming Language and Ball Python.
relation{tuple_delimiter}Mercury Planet{tuple_delimiter}Mercury Element{tuple_delimiter}name ambiguity, distinct referents{tuple_delimiter}The text distinguishes Mercury Planet from Mercury Element.
relation{tuple_delimiter}Mercury Planet{tuple_delimiter}Sun{tuple_delimiter}orbit, astronomy{tuple_delimiter}The text states Mercury Planet orbits the Sun in 88 days.
{completion_delimiter}
""",

    # ── Example 3 · Event vs Process vs Concept ─────────────────────────────
    """<Entity_types>
["person","organization","location","event","artifact","work","naturalentity","concept","process"]

<Input Text>
```
The Manhattan Project developed the first nuclear weapon using isotope separation
via gaseous diffusion. The Trinity test in July 1945 was the first detonation.
Nuclear fission had been theorized since the 1930s as a fundamental physical phenomenon.
```

<Output>
entity{tuple_delimiter}Manhattan Project{tuple_delimiter}event{tuple_delimiter}Program described in the text as having developed the first nuclear weapon.
entity{tuple_delimiter}Nuclear Weapon{tuple_delimiter}artifact{tuple_delimiter}Weapon described in the text as the first nuclear weapon developed by the Manhattan Project.
entity{tuple_delimiter}Isotope Separation{tuple_delimiter}process{tuple_delimiter}Process described in the text as used by the Manhattan Project via gaseous diffusion.
entity{tuple_delimiter}Gaseous Diffusion{tuple_delimiter}process{tuple_delimiter}Isotope separation method described in the text as the technique used by the Manhattan Project.
entity{tuple_delimiter}Trinity Test{tuple_delimiter}event{tuple_delimiter}Detonation described in the text as occurring in July 1945 and as the first nuclear detonation.
entity{tuple_delimiter}Nuclear Fission{tuple_delimiter}concept{tuple_delimiter}Physical phenomenon described in the text as theorized since the 1930s.
relation{tuple_delimiter}Manhattan Project{tuple_delimiter}Nuclear Weapon{tuple_delimiter}development{tuple_delimiter}The text states the Manhattan Project developed the first nuclear weapon.
relation{tuple_delimiter}Manhattan Project{tuple_delimiter}Gaseous Diffusion{tuple_delimiter}method usage{tuple_delimiter}The text states the Manhattan Project used isotope separation via gaseous diffusion.
relation{tuple_delimiter}Isotope Separation{tuple_delimiter}Gaseous Diffusion{tuple_delimiter}method specialization{tuple_delimiter}The text states isotope separation was performed via gaseous diffusion.
relation{tuple_delimiter}Isotope Separation{tuple_delimiter}Nuclear Weapon{tuple_delimiter}development method{tuple_delimiter}The text states isotope separation was used in developing the first nuclear weapon.
relation{tuple_delimiter}Manhattan Project{tuple_delimiter}Trinity Test{tuple_delimiter}program milestone{tuple_delimiter}The text identifies the Trinity test as the first detonation produced by the Manhattan Project.
relation{tuple_delimiter}Nuclear Fission{tuple_delimiter}Manhattan Project{tuple_delimiter}scientific basis{tuple_delimiter}The text implies nuclear fission as the physical phenomenon the Manhattan Project applied.
{completion_delimiter}
""",

    # ── Example 4 · Work subtypes + artifact vs work ────────────────────────
    """<Entity_types>
["person","organization","location","event","artifact","work","naturalentity","concept","process"]

<Input Text>
```
ImageNet transformed computer vision. The ResNet paper showed that residual connections
fix the vanishing gradient problem. The EU AI Act now governs high-risk AI systems.
Google trained PaLM 2 on a cluster of TPU V4 chips; BERT model weights are available
on HuggingFace.
```

<Output>
entity{tuple_delimiter}ImageNet{tuple_delimiter}work{tuple_delimiter}Dataset described in the text as having transformed computer vision.
entity{tuple_delimiter}Computer Vision{tuple_delimiter}concept{tuple_delimiter}Research area described in the text as transformed by ImageNet.
entity{tuple_delimiter}ResNet{tuple_delimiter}work{tuple_delimiter}Paper described in the text as showing that residual connections fix the vanishing gradient problem.
entity{tuple_delimiter}Residual Connections{tuple_delimiter}concept{tuple_delimiter}Mechanism described in the text as fixing the vanishing gradient problem.
entity{tuple_delimiter}Vanishing Gradient Problem{tuple_delimiter}concept{tuple_delimiter}Problem described in the text as something residual connections fix.
entity{tuple_delimiter}EU AI Act{tuple_delimiter}work{tuple_delimiter}Regulation described in the text as governing high-risk AI systems.
entity{tuple_delimiter}High-Risk AI Systems{tuple_delimiter}work{tuple_delimiter}AI systems described in the text as governed by the EU AI Act.
entity{tuple_delimiter}Google{tuple_delimiter}organization{tuple_delimiter}Entity described in the text as having trained PaLM 2 on a cluster of TPU V4 chips.
entity{tuple_delimiter}PaLM 2{tuple_delimiter}work{tuple_delimiter}Model described in the text as trained by Google on a cluster of TPU V4 chips.
entity{tuple_delimiter}TPU V4{tuple_delimiter}artifact{tuple_delimiter}Chip described in the text as the hardware cluster on which Google trained PaLM 2.
entity{tuple_delimiter}BERT{tuple_delimiter}work{tuple_delimiter}Model described in the text as having weights available on HuggingFace.
entity{tuple_delimiter}BERT Model Weights{tuple_delimiter}work{tuple_delimiter}Model weights described in the text as available on HuggingFace.
entity{tuple_delimiter}HuggingFace{tuple_delimiter}organization{tuple_delimiter}Entity described in the text as the place where BERT model weights are available.
relation{tuple_delimiter}ImageNet{tuple_delimiter}Computer Vision{tuple_delimiter}field impact{tuple_delimiter}The text states ImageNet transformed computer vision.
relation{tuple_delimiter}ResNet{tuple_delimiter}Residual Connections{tuple_delimiter}introduction, demonstration{tuple_delimiter}The text states the ResNet paper showed residual connections fix the vanishing gradient problem.
relation{tuple_delimiter}Residual Connections{tuple_delimiter}Vanishing Gradient Problem{tuple_delimiter}solution{tuple_delimiter}The text states residual connections fix the vanishing gradient problem.
relation{tuple_delimiter}EU AI Act{tuple_delimiter}High-Risk AI Systems{tuple_delimiter}governance, regulation{tuple_delimiter}The text states the EU AI Act governs high-risk AI systems.
relation{tuple_delimiter}Google{tuple_delimiter}PaLM 2{tuple_delimiter}development, training{tuple_delimiter}The text states Google trained PaLM 2 on a cluster of TPU V4 chips.
relation{tuple_delimiter}PaLM 2{tuple_delimiter}TPU V4{tuple_delimiter}hardware dependency{tuple_delimiter}The text states PaLM 2 was trained on a cluster of TPU V4 chips.
relation{tuple_delimiter}BERT{tuple_delimiter}BERT Model Weights{tuple_delimiter}model artifact{tuple_delimiter}The text states BERT model weights are available on HuggingFace.
relation{tuple_delimiter}BERT Model Weights{tuple_delimiter}HuggingFace{tuple_delimiter}distribution, availability{tuple_delimiter}The text states BERT model weights are available on HuggingFace.
relation{tuple_delimiter}BERT{tuple_delimiter}HuggingFace{tuple_delimiter}distribution, availability{tuple_delimiter}The text states BERT model weights are available on HuggingFace.
{completion_delimiter}
""",

    # ── Example 5 · N-ary decomposition and location type ──────────────────
    """<Entity_types>
["person","organization","location","event","artifact","work","naturalentity","concept","process"]

<Input Text>
```
Marie Curie, Pierre Curie, and Henri Becquerel jointly received the 1903 Nobel Prize
in Physics in Stockholm for their discovery of radioactivity. The award was administered
by the Royal Swedish Academy of Sciences, headquartered in Stockholm, Sweden.
```

<Output>
entity{tuple_delimiter}Marie Curie{tuple_delimiter}person{tuple_delimiter}Scientist described in the text as jointly receiving the 1903 Nobel Prize in Physics for the discovery of radioactivity.
entity{tuple_delimiter}Pierre Curie{tuple_delimiter}person{tuple_delimiter}Scientist described in the text as jointly receiving the 1903 Nobel Prize in Physics for the discovery of radioactivity.
entity{tuple_delimiter}Henri Becquerel{tuple_delimiter}person{tuple_delimiter}Scientist described in the text as jointly receiving the 1903 Nobel Prize in Physics for the discovery of radioactivity.
entity{tuple_delimiter}1903 Nobel Prize in Physics{tuple_delimiter}event{tuple_delimiter}Award described in the text as jointly given to Marie Curie, Pierre Curie, and Henri Becquerel in Stockholm for the discovery of radioactivity.
entity{tuple_delimiter}Radioactivity{tuple_delimiter}concept{tuple_delimiter}Phenomenon described in the text as the discovery for which the 1903 Nobel Prize in Physics was awarded.
entity{tuple_delimiter}Royal Swedish Academy of Sciences{tuple_delimiter}organization{tuple_delimiter}Body described in the text as the administrator of the 1903 Nobel Prize in Physics, headquartered in Stockholm.
entity{tuple_delimiter}Stockholm{tuple_delimiter}location{tuple_delimiter}City described in the text as the location of the Nobel Prize ceremony and headquarters of the Royal Swedish Academy of Sciences.
relation{tuple_delimiter}Marie Curie{tuple_delimiter}1903 Nobel Prize in Physics{tuple_delimiter}receipt, recognition{tuple_delimiter}The text states Marie Curie jointly received the 1903 Nobel Prize in Physics.
relation{tuple_delimiter}Pierre Curie{tuple_delimiter}1903 Nobel Prize in Physics{tuple_delimiter}receipt, recognition{tuple_delimiter}The text states Pierre Curie jointly received the 1903 Nobel Prize in Physics.
relation{tuple_delimiter}Henri Becquerel{tuple_delimiter}1903 Nobel Prize in Physics{tuple_delimiter}receipt, recognition{tuple_delimiter}The text states Henri Becquerel jointly received the 1903 Nobel Prize in Physics.
relation{tuple_delimiter}1903 Nobel Prize in Physics{tuple_delimiter}Radioactivity{tuple_delimiter}scientific basis, recognition{tuple_delimiter}The text states the prize was awarded for the discovery of radioactivity.
relation{tuple_delimiter}1903 Nobel Prize in Physics{tuple_delimiter}Stockholm{tuple_delimiter}award location{tuple_delimiter}The text states the 1903 Nobel Prize in Physics was received in Stockholm.
relation{tuple_delimiter}Royal Swedish Academy of Sciences{tuple_delimiter}1903 Nobel Prize in Physics{tuple_delimiter}organization, administration{tuple_delimiter}The text states the award was administered by the Royal Swedish Academy of Sciences.
relation{tuple_delimiter}Royal Swedish Academy of Sciences{tuple_delimiter}Stockholm{tuple_delimiter}headquarters, location{tuple_delimiter}The text states the Royal Swedish Academy of Sciences is headquartered in Stockholm, Sweden.
{completion_delimiter}
""",

    # ── Example 6 · Concept vs Process (hardest pair) ───────────────────────
    """<Entity_types>
["person","organization","location","event","artifact","work","naturalentity","concept","process"]

<Input Text>
```
Attention, at its core, is a weighted sum over values. Multi-head attention runs
this operation in parallel across multiple representation subspaces. Backpropagation
propagates error gradients from the loss surface back through each layer.
The chain rule of calculus underlies this.
```

<Output>
entity{tuple_delimiter}Attention{tuple_delimiter}concept{tuple_delimiter}Mechanism described in the text as a weighted sum over values at its core.
entity{tuple_delimiter}Values{tuple_delimiter}concept{tuple_delimiter}Elements described in the text as inputs over which attention computes a weighted sum.
entity{tuple_delimiter}Multi-Head Attention{tuple_delimiter}process{tuple_delimiter}Operation described in the text as running the attention mechanism in parallel across multiple representation subspaces.
entity{tuple_delimiter}Representation Subspaces{tuple_delimiter}concept{tuple_delimiter}Subspaces described in the text as the multiple spaces across which multi-head attention runs in parallel.
entity{tuple_delimiter}Backpropagation{tuple_delimiter}process{tuple_delimiter}Procedure described in the text as propagating error gradients from the loss surface back through each layer.
entity{tuple_delimiter}Error Gradients{tuple_delimiter}concept{tuple_delimiter}Gradients described in the text as propagated by backpropagation.
entity{tuple_delimiter}Loss Surface{tuple_delimiter}concept{tuple_delimiter}Surface described in the text as the starting context from which error gradients are propagated.
entity{tuple_delimiter}Layer{tuple_delimiter}concept{tuple_delimiter}Model component described in the text as traversed by error gradients during backpropagation.
entity{tuple_delimiter}Chain Rule{tuple_delimiter}concept{tuple_delimiter}Mathematical principle described in the text as underlying backpropagation.
relation{tuple_delimiter}Multi-Head Attention{tuple_delimiter}Attention{tuple_delimiter}instantiation, parallelisation{tuple_delimiter}The text describes multi-head attention as running the attention operation in parallel across multiple subspaces.
relation{tuple_delimiter}Attention{tuple_delimiter}Values{tuple_delimiter}weighted sum{tuple_delimiter}The text states attention is a weighted sum over values.
relation{tuple_delimiter}Multi-Head Attention{tuple_delimiter}Representation Subspaces{tuple_delimiter}parallel execution{tuple_delimiter}The text states multi-head attention runs across multiple representation subspaces.
relation{tuple_delimiter}Backpropagation{tuple_delimiter}Error Gradients{tuple_delimiter}gradient propagation{tuple_delimiter}The text states backpropagation propagates error gradients.
relation{tuple_delimiter}Error Gradients{tuple_delimiter}Loss Surface{tuple_delimiter}origin context{tuple_delimiter}The text states error gradients are propagated from the loss surface.
relation{tuple_delimiter}Error Gradients{tuple_delimiter}Layer{tuple_delimiter}layer traversal{tuple_delimiter}The text states error gradients are propagated back through each layer.
relation{tuple_delimiter}Backpropagation{tuple_delimiter}Chain Rule{tuple_delimiter}mathematical foundation{tuple_delimiter}The text states the chain rule of calculus underlies backpropagation.
{completion_delimiter}
""",

    # ── Example 7 · Metrics and roles are NOT entities ──────────────────────
    """<Entity_types>
["person","organization","location","event","artifact","work","naturalentity","concept","process"]

<Input Text>
```
The model achieved 92.3% accuracy on GLUE with 14ms latency on an H100 at 700W TDP.
Sam Altman, CEO of OpenAI, presented the results.
```

<Output>
entity{tuple_delimiter}GLUE{tuple_delimiter}work{tuple_delimiter}Benchmark described in the text as what the model was evaluated on.
entity{tuple_delimiter}H100{tuple_delimiter}artifact{tuple_delimiter}Hardware described in the text as the device on which the model was evaluated.
entity{tuple_delimiter}Sam Altman{tuple_delimiter}person{tuple_delimiter}Individual described in the text as CEO of OpenAI who presented the results.
entity{tuple_delimiter}OpenAI{tuple_delimiter}organization{tuple_delimiter}Entity described in the text as the organization whose CEO presented the results.
relation{tuple_delimiter}Sam Altman{tuple_delimiter}OpenAI{tuple_delimiter}leadership, role{tuple_delimiter}The text identifies Sam Altman as CEO of OpenAI.
relation{tuple_delimiter}H100{tuple_delimiter}GLUE{tuple_delimiter}evaluation infrastructure{tuple_delimiter}The text states the model achieved 92.3% accuracy on GLUE with 14ms latency on the H100 at 700W TDP.
{completion_delimiter}
""",

    # ── Example 8 · Negation does not produce positive edges ────────────────
    """<Entity_types>
["person","organization","location","event","artifact","work","naturalentity","concept","process"]

<Input Text>
```
GPT-3 was trained without RLHF, unlike its successor GPT-4 which used RLHF extensively.
The study found no significant correlation between model size and factual accuracy.
```

<Output>
entity{tuple_delimiter}GPT-3{tuple_delimiter}work{tuple_delimiter}Model described in the text as trained without RLHF. [negated context: the text explicitly states GPT-3 does NOT use RLHF]
entity{tuple_delimiter}GPT-4{tuple_delimiter}work{tuple_delimiter}Model described in the text as the successor to GPT-3 and as having used RLHF extensively.
entity{tuple_delimiter}RLHF{tuple_delimiter}process{tuple_delimiter}Training procedure described in the text as used extensively by GPT-4 but explicitly not used by GPT-3.
relation{tuple_delimiter}GPT-4{tuple_delimiter}RLHF{tuple_delimiter}training method, alignment{tuple_delimiter}The text states GPT-4 used RLHF extensively.
relation{tuple_delimiter}GPT-3{tuple_delimiter}GPT-4{tuple_delimiter}succession, model lineage{tuple_delimiter}The text describes GPT-4 as the successor to GPT-3.
{completion_delimiter}
""",

    # ── Example 9 · Biomedical academic paper ───────────────────────────────
    """<Entity_types>
["person","organization","location","event","artifact","work","naturalentity","concept","process"]

<Input Text>
```
CRISPR-Cas9 enables precise gene editing by cutting DNA at targeted sequences.
The BRCA1 gene mutation increases breast cancer risk. Doxorubicin, a chemotherapy
drug, intercalates into DNA strands to inhibit replication.
```

<Output>
entity{tuple_delimiter}CRISPR-Cas9{tuple_delimiter}process{tuple_delimiter}Gene editing procedure described in the text as enabling precise cuts in DNA at targeted sequences.
entity{tuple_delimiter}DNA{tuple_delimiter}naturalentity{tuple_delimiter}Biological molecule described in the text as cut by CRISPR-Cas9 and intercalated into by Doxorubicin.
entity{tuple_delimiter}Targeted Sequences{tuple_delimiter}concept{tuple_delimiter}Sequences described in the text as the DNA locations cut by CRISPR-Cas9.
entity{tuple_delimiter}BRCA1{tuple_delimiter}naturalentity{tuple_delimiter}Gene described in the text as having a mutation that increases breast cancer risk.
entity{tuple_delimiter}Breast Cancer{tuple_delimiter}concept{tuple_delimiter}Condition described in the text as having its risk increased by BRCA1 gene mutations.
entity{tuple_delimiter}Doxorubicin{tuple_delimiter}artifact{tuple_delimiter}Drug described in the text as intercalating into DNA strands to inhibit replication.
entity{tuple_delimiter}DNA Replication{tuple_delimiter}process{tuple_delimiter}Replication process described in the text as inhibited by Doxorubicin after intercalating into DNA strands.
relation{tuple_delimiter}CRISPR-Cas9{tuple_delimiter}DNA{tuple_delimiter}gene editing, DNA cutting{tuple_delimiter}The text states CRISPR-Cas9 cuts DNA at targeted sequences.
relation{tuple_delimiter}CRISPR-Cas9{tuple_delimiter}Targeted Sequences{tuple_delimiter}targeting, sequence specificity{tuple_delimiter}The text states CRISPR-Cas9 cuts DNA at targeted sequences.
relation{tuple_delimiter}BRCA1{tuple_delimiter}Breast Cancer{tuple_delimiter}genetic risk, disease association{tuple_delimiter}The text states a BRCA1 gene mutation increases breast cancer risk.
relation{tuple_delimiter}Doxorubicin{tuple_delimiter}DNA{tuple_delimiter}drug mechanism, intercalation{tuple_delimiter}The text states Doxorubicin intercalates into DNA strands.
relation{tuple_delimiter}Doxorubicin{tuple_delimiter}DNA Replication{tuple_delimiter}replication inhibition{tuple_delimiter}The text states Doxorubicin inhibits replication.
{completion_delimiter}
""",

    # ── Example 10 · Enterprise document (product / process / report) ────────
    """<Entity_types>
["person","organization","location","event","artifact","work","naturalentity","concept","process"]

<Input Text>
```
The Q3 2024 Product Review concluded that the Apollo Platform failed to meet
its 99.9% uptime SLA. The root cause analysis, led by Diana Chen, identified
a memory leak in the Data Ingestion Pipeline as the primary fault.
A remediation plan was submitted to the Engineering Steering Committee.
```

<Output>
entity{tuple_delimiter}Q3 2024 Product Review{tuple_delimiter}event{tuple_delimiter}Review described in the text as having concluded that the Apollo Platform failed to meet its uptime SLA.
entity{tuple_delimiter}Apollo Platform{tuple_delimiter}work{tuple_delimiter}Platform described in the text as failing to meet its 99.9% uptime SLA in the Q3 2024 Product Review.
entity{tuple_delimiter}Uptime SLA{tuple_delimiter}work{tuple_delimiter}Service-level agreement described in the text as the 99.9% uptime target the Apollo Platform failed to meet.
entity{tuple_delimiter}Root Cause Analysis{tuple_delimiter}process{tuple_delimiter}Analysis described in the text as led by Diana Chen and as identifying a memory leak in the Data Ingestion Pipeline.
entity{tuple_delimiter}Diana Chen{tuple_delimiter}person{tuple_delimiter}Individual described in the text as having led the root cause analysis.
entity{tuple_delimiter}Memory Leak{tuple_delimiter}concept{tuple_delimiter}Fault described in the text as identified in the Data Ingestion Pipeline and as the primary fault.
entity{tuple_delimiter}Data Ingestion Pipeline{tuple_delimiter}process{tuple_delimiter}Pipeline described in the text as containing the memory leak identified as the primary fault.
entity{tuple_delimiter}Engineering Steering Committee{tuple_delimiter}organization{tuple_delimiter}Committee described in the text as the recipient of the remediation plan.
entity{tuple_delimiter}Remediation Plan{tuple_delimiter}work{tuple_delimiter}Document described in the text as submitted to the Engineering Steering Committee following the root cause analysis.
relation{tuple_delimiter}Q3 2024 Product Review{tuple_delimiter}Apollo Platform{tuple_delimiter}evaluation, SLA failure{tuple_delimiter}The text states the Q3 2024 Product Review concluded the Apollo Platform failed to meet its 99.9% uptime SLA.
relation{tuple_delimiter}Apollo Platform{tuple_delimiter}Uptime SLA{tuple_delimiter}SLA failure{tuple_delimiter}The text states the Apollo Platform failed to meet its 99.9% uptime SLA.
relation{tuple_delimiter}Root Cause Analysis{tuple_delimiter}Diana Chen{tuple_delimiter}analysis leadership{tuple_delimiter}The text states Diana Chen led the root cause analysis.
relation{tuple_delimiter}Root Cause Analysis{tuple_delimiter}Memory Leak{tuple_delimiter}fault identification{tuple_delimiter}The text states the root cause analysis identified a memory leak as the primary fault.
relation{tuple_delimiter}Memory Leak{tuple_delimiter}Data Ingestion Pipeline{tuple_delimiter}fault location{tuple_delimiter}The text states the memory leak was in the Data Ingestion Pipeline.
relation{tuple_delimiter}Memory Leak{tuple_delimiter}Apollo Platform{tuple_delimiter}root cause, SLA failure{tuple_delimiter}The text identifies the memory leak as the primary fault behind the Apollo Platform's SLA failure.
relation{tuple_delimiter}Remediation Plan{tuple_delimiter}Root Cause Analysis{tuple_delimiter}follow-up action{tuple_delimiter}The text states the remediation plan followed the root cause analysis.
relation{tuple_delimiter}Remediation Plan{tuple_delimiter}Engineering Steering Committee{tuple_delimiter}submission, governance{tuple_delimiter}The text states the remediation plan was submitted to the Engineering Steering Committee.
{completion_delimiter}
""",
]

PROMPTS["entity_extraction_normalization_examples"] = [
    # ── Example 11 · Lowercase-only names are normalized ─────────────────────
    """<Entity_types>
["person","organization","location","event","artifact","work","naturalentity","concept","process"]

<Input Text>
```
The llm application integrates a rag pipeline for customer support.
```

<Output>
entity{tuple_delimiter}LLM Application{tuple_delimiter}work{tuple_delimiter}System described in the text as integrating a RAG pipeline for customer support.
entity{tuple_delimiter}RAG Pipeline{tuple_delimiter}process{tuple_delimiter}Pipeline described in the text as integrated into the LLM Application for customer support.
relation{tuple_delimiter}LLM Application{tuple_delimiter}RAG Pipeline{tuple_delimiter}integration, retrieval architecture{tuple_delimiter}The text states the LLM Application integrates a RAG pipeline for customer support.
{completion_delimiter}
""",
    # ── Example 12 · Existing uppercase names are preserved ───────────────────
    """<Entity_types>
["person","organization","location","event","artifact","work","naturalentity","concept","process"]

<Input Text>
```
OpenAI API documentation explains how BERT can be used for semantic search.
```

<Output>
entity{tuple_delimiter}OpenAI API{tuple_delimiter}work{tuple_delimiter}Documentation described in the text as explaining how BERT can be used for semantic search.
entity{tuple_delimiter}BERT{tuple_delimiter}work{tuple_delimiter}Model described in the text as usable for semantic search according to the OpenAI API documentation.
entity{tuple_delimiter}Semantic Search{tuple_delimiter}process{tuple_delimiter}Search procedure described in the text as a use case for BERT.
relation{tuple_delimiter}OpenAI API{tuple_delimiter}BERT{tuple_delimiter}usage guidance, semantic search{tuple_delimiter}The text states the OpenAI API documentation explains how BERT can be used for semantic search.
relation{tuple_delimiter}BERT{tuple_delimiter}Semantic Search{tuple_delimiter}model usage{tuple_delimiter}The text states BERT can be used for semantic search.
{completion_delimiter}
""",
]

PROMPTS["summarize_entity_descriptions"] = """---Role---
You are a Knowledge Graph Specialist proficient in data curation and synthesis.

---Task---
Synthesize a list of descriptions of a given entity or relationship into a single,
comprehensive, and cohesive summary.

---Instructions---
1.  Input format: descriptions are provided in JSON format, one object per line,
    inside the Description List section.

2.  Output format: plain text, multiple paragraphs if needed.
    No additional formatting, no preamble, no concluding remarks.

3.  Comprehensiveness: integrate all key information from every provided description.
    Do not omit any important facts or details.

4.  Perspective: write in objective third person.
    Begin the summary by explicitly naming the entity or relationship.

5.  Conflict handling:
    - First determine whether conflicts arise from multiple distinct entities
      sharing the same name. If so, summarize each one separately.
    - If conflicts are within a single entity (e.g. historical discrepancies),
      attempt to reconcile them or present both viewpoints with noted uncertainty.

6.  Length: the summary must not exceed {summary_length} tokens while maintaining
    depth and completeness.

7.  Language: write the entire output in {language}.
    Retain proper nouns in their original language when no widely accepted
    translation exists.

---Input---
{description_type} Name: {description_name}

Description List:
```
{description_list}
```

---Output---
"""

PROMPTS["fail_response"] = (
    "Sorry, I'm not able to provide an answer to that question.[no-context]"
)

PROMPTS["rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph and Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize both `Knowledge Graph Data` and `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunks which directly support the facts. Correlate reference_id with the entries in the `Reference Document List` to generate the References section.
  - Generate a references section at the end of the response. ONLY include reference_id values that actually appear in the provided `Reference Document List`. Do NOT invent or hallucinate reference_id values that are not in the list.
  - The References section is MANDATORY. Always output it even if only one source is cited.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title must be taken VERBATIM from the `Reference Document List` provided in the Context. Do NOT invent document titles.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{context_data}
"""

PROMPTS["naive_rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a **References** section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{content_data}
"""

PROMPTS["kg_query_context"] = """
Knowledge Graph Data (Entity):

```json
{entities_str}
```

Knowledge Graph Data (Relationship):

```json
{relations_str}
```

Document Chunks (Each entry has an `id` for inline citation and a `reference_id` for the Reference Document List):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS["naive_query_context"] = """
Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS["keywords_extraction"] = """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON. It will be parsed directly by a JSON parser.
2. **Source of Truth**: All keywords must be explicitly derived from the user query, with both high-level and low-level keyword categories are required to contain content.
3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from "latest financial report of Apple Inc.", you should extract "latest financial report" and "Apple Inc." rather than "latest", "financial", "report", and "Apple".
4. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), you must return a JSON object with empty lists for both keyword types.
5. **Language**: All extracted keywords MUST be in {language}. Proper nouns (e.g., personal names, place names, organization names) should be kept in their original language.
6. **Casing (high_level_keywords)**: Use lowercase phrases by default. Preserve meaningful uppercase or mixed-case proper nouns/acronyms (e.g., OpenAI, BERT, API, 6G).
7. **Casing (low_level_keywords)**: Use entity-style casing. Preserve mixed-case proper nouns/acronyms; otherwise normalize case-insensitive phrases to canonical title-style wording.

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"

Output:
{
  "high_level_keywords": ["international trade", "global economic stability", "economic impact"],
  "low_level_keywords": ["trade agreements", "tariffs", "currency exchange", "imports", "exports"]
}

""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"

Output:
{
  "high_level_keywords": ["environmental consequences", "deforestation", "biodiversity loss"],
  "low_level_keywords": ["species extinction", "habitat destruction", "carbon emissions", "rainforest", "ecosystem"]
}

""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"

Output:
{
  "high_level_keywords": ["education", "poverty reduction", "socioeconomic development"],
  "low_level_keywords": ["school access", "literacy rates", "job training", "income inequality"]
}

""",
    """Example 4:

Query: "How can OpenAI API and BERT be used for semantic search in a 6G assistant?"

Output:
{
  "high_level_keywords": ["semantic search", "assistant design", "6G application"],
  "low_level_keywords": ["OpenAI API", "BERT", "6G assistant"]
}

""",
]
