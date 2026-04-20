from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

# All delimiters must be formatted as "<|UPPER_CASE_STRING|>"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["ENTITY_NAME_CASE_RULE_DEFAULT"] = (
    "Preserve source lexical content. If casing is case-insensitive or inconsistent, "
    "title-case each significant word without changing words, aliases, or abbreviations."
)
PROMPTS["ENTITY_NAME_CASE_RULE_NORMALIZED"] = (
    "Preserve source lexical content. Normalize casing and separator artifacts only; do not rewrite aliases, abbreviations, or lexical meaning.\n"
    "                       Normalize each word to canonical casing unless the phrase is already canonical.\n"
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
- concept        : Is it a stable domain concept with a specific source-grounded referent,
                   best answered by "what IS it"?
- process        : Is it a named or stable domain method/procedure with a specific
                   source-grounded referent, best answered by "how IS IT done"?

Stable/source-grounded does not mean globally famous. It means the phrase is
explicitly supported by the source as a concrete named referent, named internal
object, reusable domain concept, or reusable method/procedure, not merely a local
role, filler, or modifier phrase.

Disambiguation rules:
  concept vs process   → "Attention Mechanism" (WHAT) = concept;
                         "Gradient Descent" (HOW)     = process.
                         When both apply, prefer process.
  artifact vs work     → H100 GPU (touchable physical chip) = artifact;
                         GPT-4 (citable/deployable model)   = work.
                         The chip running a model = artifact; the model itself = work.
  event vs process     → "Q3 Business Review 2024" (happened once, anchored) = event;
                         "Quarterly Review Procedure" (repeatable workflow)   = process.
  work vs process      -> If the text refers to a document, report, plan,
                         specification, runbook, postmortem, or written artifact,
                         prefer work. If the text refers to an executed method,
                         analysis activity, operational procedure, or workflow,
                         prefer process.

Use ONLY the types provided in <Entity_types>. No other type values are permitted.

---Ambiguity Protocol---
Use this protocol only when a candidate mention is relevant but its validity or
type is unclear. The Extraction Workflow below remains the main execution order.
When you cannot immediately assign a type, follow these steps in order:

  Step 1  Is it a specific, stable, source-grounded referent, or only a
          descriptor / modifier / placeholder phrase?
          "Hybrid vehicle technology" -> descriptor phrase -> DO NOT EXTRACT.
          "the query", "retrieved documents", "the generator" -> generic/filler -> DO NOT EXTRACT.
          "Prius" -> named product -> artifact.
          "Self-Attention" -> stable domain method -> process.

  Step 2  Apply the WHAT / HOW test.
          "X is ___" completes naturally with a definition?  → concept.
          "X works by ___" completes naturally with steps?   → process.
          Both complete? → process. Neither completes? → DO NOT EXTRACT.

  Step 3  Is this entity central to the passage's argument?
          YES -> concept only if it is an abstract, stable, source-grounded
                 referent. Otherwise -> DO NOT EXTRACT.
          NO  -> DO NOT EXTRACT.

Prioritize entities that can form clear, meaningful relationships with other
extracted entities. Avoid outputting isolated placeholder-like entities that
cannot connect to anything else in the graph.

Generic mechanisms, roles, objects, or unnamed placeholders belong in
relationship_keywords or relationship_description unless they are stable named
concepts, named methods, or concrete named referents.

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

  Unnamed generics  ("a model", "the algorithm", "the team", "the query",
                     "retrieved documents", "the generator", "the layer",
                     "values", "inputs", "outputs", "results") -> skip entirely.
                     Do NOT extract a generic placeholder just to satisfy endpoint closure.

  Pure time labels  ("Q3 2024", "FY2023", "deadline", "launch date")
                    -> skip as standalone entities. Use them as attributes in
                    descriptions unless they are part of a complete named event
                    or work, such as "Q3 2024 Product Review".

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

  entity_name        : Preserve source lexical content from the text.
                       {entity_name_case_rule}
                       Do NOT rewrite aliases or abbreviations:
                       "RAG" must not become "Retrieval-Augmented Generation"
                       unless that full lexical form appears in the source text.
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
                                  Use lowercase by default; preserve meaningful
                                  mixed/uppercase proper nouns and acronyms.
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
    Use the Ambiguity Protocol above only when a candidate's validity or type is
    unclear.

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
    Even for undirected relations, use consistent subject-like to object-like
    ordering when possible, such as system -> component, person -> organization,
    method -> target, cause -> effect, document -> subject, and event -> location.
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

---Actual Input Shape---
The chat history immediately before this message contains:
1. the original extraction user prompt, including the source input text;
2. the previous assistant extraction for that same input text.
Treat the previous assistant extraction as the current extraction state.
Your response must contain only missed or corrected entries, not a full rewrite.

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

---Continuation Example: Actual Chat Shape---
The continuation call is sent as chat messages in this shape:

role=system:
The entity extraction system prompt shown above.

role=user:
---Task---
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
[person,organization,location,event,artifact,work,naturalentity,concept,process]

<Input Text>
```
Corrective Retrieval-Augmented Generation uses a Retrieval Evaluator module to
score retrieval quality. When retrieval quality is poor, Corrective
Retrieval-Augmented Generation uses a Query Rewriter module.
```

<Output>

role=assistant:
entity{tuple_delimiter}Corrective Retrieval-Augmented Generation{tuple_delimiter}process{tuple_delimiter}Framework described in the text as using retrieval evaluation.
entity{tuple_delimiter}Retrieval Evaluator{tuple_delimiter}process{tuple_delimiter}Module described in the text as evaluating retrieval quality.
relation{tuple_delimiter}Corrective Retrieval-Augmented Generation{tuple_delimiter}Retrieval Evaluator{tuple_delimiter}component, retrieval assessment{tuple_delimiter}The text states Corrective Retrieval-Augmented Generation uses a Retrieval Evaluator.
{completion_delimiter}

role=user:
This continuation prompt.

Correct role=assistant output:
entity{tuple_delimiter}Query Rewriter{tuple_delimiter}process{tuple_delimiter}Module described in the text as rewriting queries when retrieval quality is poor.
relation{tuple_delimiter}Corrective Retrieval-Augmented Generation{tuple_delimiter}Query Rewriter{tuple_delimiter}component, query rewriting{tuple_delimiter}The text states Corrective Retrieval-Augmented Generation uses a Query Rewriter.
{completion_delimiter}

For casing or separator corrections, output the corrected normalized form for the same
lexical content. Do not use continuation to claim that a different lexical referent
supersedes an earlier entity; this prompt is additive/corrective, not a delete operation.

<Output>
"""

PROMPTS["entity_extraction_examples"] = [

    # Example 1 - RAG framework extraction without generic endpoint nodes
    """<Entity_types>
["person","organization","location","event","artifact","work","naturalentity","concept","process"]

<Input Text>
```
Corrective Retrieval-Augmented Generation (CRAG) uses a Retrieval Evaluator module
to score retrieval quality. When retrieval quality is poor, CRAG invokes a Query
Rewriter module before passing evidence to a Knowledge Refinement Module. The
Answer Composer module uses refined evidence to produce final answers. The paper
describes the user query and retrieved documents only as ordinary inputs, not as
named components. The authors evaluate CRAG on Natural Questions and HotpotQA and
report lower hallucination than a baseline RAG pipeline. The abstract also notes
that retrieval noise is handled through the evaluator and query rewriter rather
than by adding a separate generator entity.
```

<Output>
entity{tuple_delimiter}Corrective Retrieval-Augmented Generation{tuple_delimiter}process{tuple_delimiter}Framework described in the text as using retrieval evaluation, query rewriting, knowledge refinement, answer composition, and benchmark evaluation.
entity{tuple_delimiter}Retrieval Evaluator{tuple_delimiter}process{tuple_delimiter}Module described in the text as scoring retrieval quality for Corrective Retrieval-Augmented Generation.
entity{tuple_delimiter}Query Rewriter{tuple_delimiter}process{tuple_delimiter}Module described in the text as invoked by Corrective Retrieval-Augmented Generation when retrieval quality is poor.
entity{tuple_delimiter}Knowledge Refinement Module{tuple_delimiter}process{tuple_delimiter}Module described in the text as receiving evidence before answer composition.
entity{tuple_delimiter}Answer Composer{tuple_delimiter}process{tuple_delimiter}Module described in the text as using refined evidence to produce final answers.
entity{tuple_delimiter}Natural Questions{tuple_delimiter}work{tuple_delimiter}Benchmark described in the text as used to evaluate Corrective Retrieval-Augmented Generation.
entity{tuple_delimiter}HotpotQA{tuple_delimiter}work{tuple_delimiter}Benchmark described in the text as used to evaluate Corrective Retrieval-Augmented Generation.
entity{tuple_delimiter}Hallucination{tuple_delimiter}concept{tuple_delimiter}Stable domain concept described in the text as reduced by Corrective Retrieval-Augmented Generation compared with a baseline RAG pipeline.
relation{tuple_delimiter}Corrective Retrieval-Augmented Generation{tuple_delimiter}Retrieval Evaluator{tuple_delimiter}component, retrieval assessment{tuple_delimiter}The text states Corrective Retrieval-Augmented Generation uses a Retrieval Evaluator module to score retrieval quality.
relation{tuple_delimiter}Corrective Retrieval-Augmented Generation{tuple_delimiter}Query Rewriter{tuple_delimiter}component, query rewriting{tuple_delimiter}The text states Corrective Retrieval-Augmented Generation invokes a Query Rewriter when retrieval quality is poor.
relation{tuple_delimiter}Corrective Retrieval-Augmented Generation{tuple_delimiter}Knowledge Refinement Module{tuple_delimiter}component, knowledge refinement{tuple_delimiter}The text states Corrective Retrieval-Augmented Generation passes evidence to a Knowledge Refinement Module.
relation{tuple_delimiter}Corrective Retrieval-Augmented Generation{tuple_delimiter}Answer Composer{tuple_delimiter}component, answer composition{tuple_delimiter}The text states Corrective Retrieval-Augmented Generation uses an Answer Composer module to produce final answers.
relation{tuple_delimiter}Knowledge Refinement Module{tuple_delimiter}Answer Composer{tuple_delimiter}evidence handoff, answer generation{tuple_delimiter}The text states refined evidence is used by the Answer Composer.
relation{tuple_delimiter}Corrective Retrieval-Augmented Generation{tuple_delimiter}Natural Questions{tuple_delimiter}evaluation benchmark{tuple_delimiter}The text states Corrective Retrieval-Augmented Generation is evaluated on Natural Questions.
relation{tuple_delimiter}Corrective Retrieval-Augmented Generation{tuple_delimiter}HotpotQA{tuple_delimiter}evaluation benchmark{tuple_delimiter}The text states Corrective Retrieval-Augmented Generation is evaluated on HotpotQA.
relation{tuple_delimiter}Corrective Retrieval-Augmented Generation{tuple_delimiter}Hallucination{tuple_delimiter}hallucination reduction{tuple_delimiter}The text states Corrective Retrieval-Augmented Generation reports lower hallucination than a baseline RAG pipeline.
{completion_delimiter}
""",

    # Example 2 - Metrics and roles stay in descriptions, no shortcut edge
    """<Entity_types>
["person","organization","location","event","artifact","work","naturalentity","concept","process"]

<Input Text>
```
AtlasLM achieved 92.3% accuracy on GLUE with 14ms latency on an H100 accelerator
at 700W TDP. The release note says the accuracy is a test-set metric and the
latency excludes preprocessing, so neither number is a standalone entity. Sam
Altman, CEO of OpenAI, presented the AtlasLM results during the OpenAI Benchmark
Review. The CEO title is a role attribute of Sam Altman, not a separate node, and
the hardware and benchmark should not be connected directly through a shortcut
edge because the named model is the evaluated subject.
```

<Output>
entity{tuple_delimiter}AtlasLM{tuple_delimiter}work{tuple_delimiter}Model described in the text as achieving 92.3% accuracy on GLUE with 14ms latency on an H100 accelerator.
entity{tuple_delimiter}GLUE{tuple_delimiter}work{tuple_delimiter}Benchmark described in the text as used to evaluate AtlasLM accuracy.
entity{tuple_delimiter}H100{tuple_delimiter}artifact{tuple_delimiter}Hardware accelerator described in the text as used for AtlasLM latency measurement at 700W TDP.
entity{tuple_delimiter}Sam Altman{tuple_delimiter}person{tuple_delimiter}Sam Altman, CEO of OpenAI per the text, presented the AtlasLM results during the OpenAI Benchmark Review.
entity{tuple_delimiter}OpenAI{tuple_delimiter}organization{tuple_delimiter}Organization described in the text as the organization where Sam Altman serves as CEO.
entity{tuple_delimiter}OpenAI Benchmark Review{tuple_delimiter}event{tuple_delimiter}Event described in the text as where Sam Altman presented the AtlasLM results.
relation{tuple_delimiter}AtlasLM{tuple_delimiter}GLUE{tuple_delimiter}benchmark evaluation, accuracy{tuple_delimiter}The text states AtlasLM achieved 92.3% accuracy on GLUE.
relation{tuple_delimiter}AtlasLM{tuple_delimiter}H100{tuple_delimiter}evaluation hardware, latency{tuple_delimiter}The text states AtlasLM achieved 14ms latency on an H100 accelerator at 700W TDP.
relation{tuple_delimiter}Sam Altman{tuple_delimiter}OpenAI{tuple_delimiter}leadership role{tuple_delimiter}The text identifies Sam Altman as CEO of OpenAI.
relation{tuple_delimiter}Sam Altman{tuple_delimiter}OpenAI Benchmark Review{tuple_delimiter}presentation{tuple_delimiter}The text states Sam Altman presented the AtlasLM results during the OpenAI Benchmark Review.
relation{tuple_delimiter}AtlasLM{tuple_delimiter}OpenAI Benchmark Review{tuple_delimiter}result presentation{tuple_delimiter}The text states the AtlasLM results were presented during the OpenAI Benchmark Review.
{completion_delimiter}
""",

    # Example 3 - Stable concepts and methods, no filler nodes
    """<Entity_types>
["person","organization","location","event","artifact","work","naturalentity","concept","process"]

<Input Text>
```
Attention is a core mechanism in the Transformer Architecture. Self-Attention
computes context-aware token representations, and Multi-Head Attention applies
Self-Attention in parallel. Backpropagation updates model parameters using
gradients, and the Chain Rule supplies its mathematical basis. The explanation
mentions values, layers, inputs, outputs, and representation subspaces only as
ordinary descriptive terms inside the mechanism. Those filler terms should remain
in descriptions or relation keywords instead of becoming entity nodes.
```

<Output>
entity{tuple_delimiter}Attention{tuple_delimiter}concept{tuple_delimiter}Stable mechanism described in the text as a core mechanism in the Transformer Architecture.
entity{tuple_delimiter}Transformer Architecture{tuple_delimiter}concept{tuple_delimiter}Architecture described in the text as using Attention as a core mechanism.
entity{tuple_delimiter}Self-Attention{tuple_delimiter}process{tuple_delimiter}Method described in the text as computing context-aware token representations.
entity{tuple_delimiter}Multi-Head Attention{tuple_delimiter}process{tuple_delimiter}Method described in the text as applying Self-Attention in parallel.
entity{tuple_delimiter}Backpropagation{tuple_delimiter}process{tuple_delimiter}Procedure described in the text as updating model parameters using gradients.
entity{tuple_delimiter}Chain Rule{tuple_delimiter}concept{tuple_delimiter}Mathematical principle described in the text as the basis for Backpropagation.
relation{tuple_delimiter}Transformer Architecture{tuple_delimiter}Attention{tuple_delimiter}core mechanism{tuple_delimiter}The text states Attention is a core mechanism in the Transformer Architecture.
relation{tuple_delimiter}Self-Attention{tuple_delimiter}Attention{tuple_delimiter}attention method{tuple_delimiter}The text describes Self-Attention as an attention method computing context-aware token representations.
relation{tuple_delimiter}Multi-Head Attention{tuple_delimiter}Self-Attention{tuple_delimiter}parallel application{tuple_delimiter}The text states Multi-Head Attention applies Self-Attention in parallel.
relation{tuple_delimiter}Backpropagation{tuple_delimiter}Chain Rule{tuple_delimiter}mathematical foundation{tuple_delimiter}The text states the Chain Rule supplies the mathematical basis for Backpropagation.
{completion_delimiter}
""",

    # Example 4 - Enterprise source-grounded process and document mentions
    """<Entity_types>
["person","organization","location","event","artifact","work","naturalentity","concept","process"]

<Input Text>
```
During Q3 2024, the Q3 2024 Product Review concluded that the Apollo Platform
failed the Apollo Uptime SLA. The Q3 2024 Root Cause Analysis activity, led by
Diana Chen, identified a Memory Leak in the Data Ingestion Pipeline as the
primary fault. The Apollo Remediation Plan document was submitted to the
Engineering Steering Committee. The Apollo Postmortem report cites the Apollo
Remediation Plan and instructs teams to update the Canary Release Procedure.
The standalone phrase Q3 2024 is only a time label in this passage.
```

<Output>
entity{tuple_delimiter}Q3 2024 Product Review{tuple_delimiter}event{tuple_delimiter}Review described in the text as concluding that the Apollo Platform failed the Apollo Uptime SLA.
entity{tuple_delimiter}Apollo Platform{tuple_delimiter}work{tuple_delimiter}Platform described in the text as failing the Apollo Uptime SLA.
entity{tuple_delimiter}Apollo Uptime SLA{tuple_delimiter}work{tuple_delimiter}Service-level agreement described in the text as failed by the Apollo Platform.
entity{tuple_delimiter}Q3 2024 Root Cause Analysis{tuple_delimiter}process{tuple_delimiter}Analysis described in the text as led by Diana Chen and as identifying a Memory Leak in the Data Ingestion Pipeline.
entity{tuple_delimiter}Diana Chen{tuple_delimiter}person{tuple_delimiter}Individual described in the text as leading the Q3 2024 Root Cause Analysis.
entity{tuple_delimiter}Memory Leak{tuple_delimiter}concept{tuple_delimiter}Stable fault concept described in the text as the primary fault found in the Data Ingestion Pipeline.
entity{tuple_delimiter}Data Ingestion Pipeline{tuple_delimiter}process{tuple_delimiter}Pipeline described in the text as containing the Memory Leak identified as the primary fault.
entity{tuple_delimiter}Apollo Remediation Plan{tuple_delimiter}work{tuple_delimiter}Plan described in the text as submitted to the Engineering Steering Committee.
entity{tuple_delimiter}Engineering Steering Committee{tuple_delimiter}organization{tuple_delimiter}Committee described in the text as receiving the Apollo Remediation Plan.
entity{tuple_delimiter}Apollo Postmortem{tuple_delimiter}work{tuple_delimiter}Report described in the text as citing the Apollo Remediation Plan and instructing teams to update the Canary Release Procedure.
entity{tuple_delimiter}Canary Release Procedure{tuple_delimiter}process{tuple_delimiter}Operational procedure described in the text as updated according to the Apollo Postmortem.
relation{tuple_delimiter}Q3 2024 Product Review{tuple_delimiter}Apollo Platform{tuple_delimiter}evaluation, SLA failure{tuple_delimiter}The text states the Q3 2024 Product Review concluded that the Apollo Platform failed the Apollo Uptime SLA.
relation{tuple_delimiter}Apollo Platform{tuple_delimiter}Apollo Uptime SLA{tuple_delimiter}SLA failure{tuple_delimiter}The text states the Apollo Platform failed the Apollo Uptime SLA.
relation{tuple_delimiter}Q3 2024 Root Cause Analysis{tuple_delimiter}Diana Chen{tuple_delimiter}analysis leadership{tuple_delimiter}The text states Diana Chen led the Q3 2024 Root Cause Analysis.
relation{tuple_delimiter}Q3 2024 Root Cause Analysis{tuple_delimiter}Memory Leak{tuple_delimiter}fault identification{tuple_delimiter}The text states the Q3 2024 Root Cause Analysis identified a Memory Leak as the primary fault.
relation{tuple_delimiter}Memory Leak{tuple_delimiter}Data Ingestion Pipeline{tuple_delimiter}fault location{tuple_delimiter}The text states the Memory Leak was in the Data Ingestion Pipeline.
relation{tuple_delimiter}Memory Leak{tuple_delimiter}Apollo Platform{tuple_delimiter}root cause, SLA failure{tuple_delimiter}The text identifies the Memory Leak as the primary fault behind the Apollo Platform SLA failure.
relation{tuple_delimiter}Apollo Remediation Plan{tuple_delimiter}Q3 2024 Root Cause Analysis{tuple_delimiter}follow-up action{tuple_delimiter}The text presents the Apollo Remediation Plan after the Q3 2024 Root Cause Analysis.
relation{tuple_delimiter}Apollo Remediation Plan{tuple_delimiter}Engineering Steering Committee{tuple_delimiter}submission, governance{tuple_delimiter}The text states the Apollo Remediation Plan was submitted to the Engineering Steering Committee.
relation{tuple_delimiter}Apollo Postmortem{tuple_delimiter}Apollo Remediation Plan{tuple_delimiter}citation, remediation tracking{tuple_delimiter}The text states the Apollo Postmortem cites the Apollo Remediation Plan.
relation{tuple_delimiter}Apollo Postmortem{tuple_delimiter}Canary Release Procedure{tuple_delimiter}procedure update{tuple_delimiter}The text states the Apollo Postmortem instructs teams to update the Canary Release Procedure.
{completion_delimiter}
""",

    # Example 5 - Negated facts do not create positive edges
    """<Entity_types>
["person","organization","location","event","artifact","work","naturalentity","concept","process"]

<Input Text>
```
GPT-3 was trained without RLHF, unlike its successor GPT-4, which used RLHF
extensively. The study found no significant correlation between model size and
factual accuracy, and it reports parameter counts only as measurement attributes.
The authors mention that factual accuracy was evaluated in a table, but the table
number, model size value, and accuracy percentage are not named entities. The
only positive training-method statement in the passage links GPT-4 to RLHF.
```

<Output>
entity{tuple_delimiter}GPT-3{tuple_delimiter}work{tuple_delimiter}Model described in the text as trained without RLHF. [negated context: the text explicitly states GPT-3 does NOT use RLHF]
entity{tuple_delimiter}GPT-4{tuple_delimiter}work{tuple_delimiter}Model described in the text as the successor to GPT-3 and as having used RLHF extensively.
entity{tuple_delimiter}RLHF{tuple_delimiter}process{tuple_delimiter}Training procedure described in the text as used extensively by GPT-4 but explicitly not used by GPT-3.
relation{tuple_delimiter}GPT-4{tuple_delimiter}RLHF{tuple_delimiter}training method, alignment{tuple_delimiter}The text states GPT-4 used RLHF extensively.
relation{tuple_delimiter}GPT-3{tuple_delimiter}GPT-4{tuple_delimiter}succession, model lineage{tuple_delimiter}The text describes GPT-4 as the successor to GPT-3.
{completion_delimiter}
""",

    # Example 6 - Biomedical academic paper with scientific processes
    """<Entity_types>
["person","organization","location","event","artifact","work","naturalentity","concept","process"]

<Input Text>
```
The CARDIO-RNA Trial studied RNX-41 for Myocardial Fibrosis and compared its
effects with Doxorubicin in a controlled arm. In the mechanistic experiment,
researchers used CRISPR-Cas9 to knock out BRCA1 in the MCF-7 Cell Line and then
measured DNA Repair through Homologous Recombination. A separate assay reported
that Doxorubicin induced Apoptosis in the MCF-7 Cell Line. The paper lists p<0.01
and Week 12 follow-up as measurement details rather than named entities.
```

<Output>
entity{tuple_delimiter}CARDIO-RNA Trial{tuple_delimiter}event{tuple_delimiter}Trial described in the text as studying RNX-41 for Myocardial Fibrosis and comparing effects with Doxorubicin.
entity{tuple_delimiter}RNX-41{tuple_delimiter}artifact{tuple_delimiter}Therapeutic compound described in the text as studied in the CARDIO-RNA Trial for Myocardial Fibrosis.
entity{tuple_delimiter}Myocardial Fibrosis{tuple_delimiter}concept{tuple_delimiter}Disease concept described in the text as the target condition in the CARDIO-RNA Trial.
entity{tuple_delimiter}Doxorubicin{tuple_delimiter}artifact{tuple_delimiter}Drug described in the text as used in a controlled arm and as inducing Apoptosis in the MCF-7 Cell Line.
entity{tuple_delimiter}CRISPR-Cas9{tuple_delimiter}process{tuple_delimiter}Gene editing procedure described in the text as used to knock out BRCA1 in the MCF-7 Cell Line.
entity{tuple_delimiter}BRCA1{tuple_delimiter}naturalentity{tuple_delimiter}Gene described in the text as knocked out by CRISPR-Cas9 in the MCF-7 Cell Line.
entity{tuple_delimiter}MCF-7 Cell Line{tuple_delimiter}naturalentity{tuple_delimiter}Cell line described in the text as used in the CRISPR-Cas9 and Doxorubicin assays.
entity{tuple_delimiter}DNA Repair{tuple_delimiter}process{tuple_delimiter}Biological process described in the text as measured through Homologous Recombination.
entity{tuple_delimiter}Homologous Recombination{tuple_delimiter}process{tuple_delimiter}DNA repair process described in the text as the mechanism through which DNA Repair was measured.
entity{tuple_delimiter}Apoptosis{tuple_delimiter}process{tuple_delimiter}Biological process described in the text as induced by Doxorubicin in the MCF-7 Cell Line.
relation{tuple_delimiter}CARDIO-RNA Trial{tuple_delimiter}RNX-41{tuple_delimiter}trial intervention{tuple_delimiter}The text states the CARDIO-RNA Trial studied RNX-41.
relation{tuple_delimiter}CARDIO-RNA Trial{tuple_delimiter}Myocardial Fibrosis{tuple_delimiter}target condition{tuple_delimiter}The text states the CARDIO-RNA Trial studied RNX-41 for Myocardial Fibrosis.
relation{tuple_delimiter}CARDIO-RNA Trial{tuple_delimiter}Doxorubicin{tuple_delimiter}controlled arm{tuple_delimiter}The text states the trial compared RNX-41 effects with Doxorubicin in a controlled arm.
relation{tuple_delimiter}CRISPR-Cas9{tuple_delimiter}BRCA1{tuple_delimiter}gene knockout{tuple_delimiter}The text states researchers used CRISPR-Cas9 to knock out BRCA1.
relation{tuple_delimiter}CRISPR-Cas9{tuple_delimiter}MCF-7 Cell Line{tuple_delimiter}experimental model{tuple_delimiter}The text states CRISPR-Cas9 was used in the MCF-7 Cell Line.
relation{tuple_delimiter}DNA Repair{tuple_delimiter}Homologous Recombination{tuple_delimiter}measurement mechanism{tuple_delimiter}The text states DNA Repair was measured through Homologous Recombination.
relation{tuple_delimiter}Doxorubicin{tuple_delimiter}Apoptosis{tuple_delimiter}drug effect{tuple_delimiter}The text states Doxorubicin induced Apoptosis.
relation{tuple_delimiter}Doxorubicin{tuple_delimiter}MCF-7 Cell Line{tuple_delimiter}assay model{tuple_delimiter}The text states Doxorubicin induced Apoptosis in the MCF-7 Cell Line.
{completion_delimiter}
""",

    # Example 7 - Enterprise migration plan with work/process boundary
    """<Entity_types>
["person","organization","location","event","artifact","work","naturalentity","concept","process"]

<Input Text>
```
The Project Helios Migration Plan describes moving the Billing API from Legacy
Queue to the Aurora Stream Processor. The Helios Migration Workflow ran during
Change Window CW-2024-09 under Maya Patel. The Operations Runbook H-17 lists the
Rollback Procedure and Cache Warmup Procedure that operators must follow if the
Aurora Stream Processor reports backpressure. The September 2024 date is only a
schedule label, while the runbook, migration plan, workflow, procedures, and
named software systems are the source-grounded referents.
```

<Output>
entity{tuple_delimiter}Project Helios Migration Plan{tuple_delimiter}work{tuple_delimiter}Migration plan described in the text as moving the Billing API from Legacy Queue to the Aurora Stream Processor.
entity{tuple_delimiter}Billing API{tuple_delimiter}work{tuple_delimiter}Software API described in the text as moved from Legacy Queue to the Aurora Stream Processor.
entity{tuple_delimiter}Legacy Queue{tuple_delimiter}work{tuple_delimiter}Software system described in the text as the source system for the Billing API migration.
entity{tuple_delimiter}Aurora Stream Processor{tuple_delimiter}work{tuple_delimiter}Software system described in the text as the target system for the Billing API migration and as reporting backpressure.
entity{tuple_delimiter}Helios Migration Workflow{tuple_delimiter}process{tuple_delimiter}Workflow described in the text as running during Change Window CW-2024-09 under Maya Patel.
entity{tuple_delimiter}Change Window CW-2024-09{tuple_delimiter}event{tuple_delimiter}Named change window described in the text as when the Helios Migration Workflow ran.
entity{tuple_delimiter}Maya Patel{tuple_delimiter}person{tuple_delimiter}Individual described in the text as overseeing the Helios Migration Workflow.
entity{tuple_delimiter}Operations Runbook H-17{tuple_delimiter}work{tuple_delimiter}Runbook described in the text as listing the Rollback Procedure and Cache Warmup Procedure.
entity{tuple_delimiter}Rollback Procedure{tuple_delimiter}process{tuple_delimiter}Procedure described in the text as listed in Operations Runbook H-17 for operators to follow.
entity{tuple_delimiter}Cache Warmup Procedure{tuple_delimiter}process{tuple_delimiter}Procedure described in the text as listed in Operations Runbook H-17 for operators to follow.
relation{tuple_delimiter}Project Helios Migration Plan{tuple_delimiter}Billing API{tuple_delimiter}migration subject{tuple_delimiter}The text states the Project Helios Migration Plan describes moving the Billing API.
relation{tuple_delimiter}Project Helios Migration Plan{tuple_delimiter}Legacy Queue{tuple_delimiter}migration source{tuple_delimiter}The text states the Billing API is moved from Legacy Queue.
relation{tuple_delimiter}Project Helios Migration Plan{tuple_delimiter}Aurora Stream Processor{tuple_delimiter}migration target{tuple_delimiter}The text states the Billing API is moved to the Aurora Stream Processor.
relation{tuple_delimiter}Helios Migration Workflow{tuple_delimiter}Change Window CW-2024-09{tuple_delimiter}execution window{tuple_delimiter}The text states the Helios Migration Workflow ran during Change Window CW-2024-09.
relation{tuple_delimiter}Helios Migration Workflow{tuple_delimiter}Maya Patel{tuple_delimiter}workflow ownership{tuple_delimiter}The text states the Helios Migration Workflow ran under Maya Patel.
relation{tuple_delimiter}Operations Runbook H-17{tuple_delimiter}Rollback Procedure{tuple_delimiter}runbook procedure{tuple_delimiter}The text states Operations Runbook H-17 lists the Rollback Procedure.
relation{tuple_delimiter}Operations Runbook H-17{tuple_delimiter}Cache Warmup Procedure{tuple_delimiter}runbook procedure{tuple_delimiter}The text states Operations Runbook H-17 lists the Cache Warmup Procedure.
relation{tuple_delimiter}Aurora Stream Processor{tuple_delimiter}Rollback Procedure{tuple_delimiter}backpressure response{tuple_delimiter}The text states operators follow the Rollback Procedure if the Aurora Stream Processor reports backpressure.
relation{tuple_delimiter}Aurora Stream Processor{tuple_delimiter}Cache Warmup Procedure{tuple_delimiter}backpressure response{tuple_delimiter}The text states operators follow the Cache Warmup Procedure if the Aurora Stream Processor reports backpressure.
{completion_delimiter}
""",
]

PROMPTS["entity_extraction_normalization_examples"] = [
    # Example 8 - Lowercase source names canonicalized without lexical rewrite
    """<Entity_types>
["person","organization","location","event","artifact","work","naturalentity","concept","process"]

<Input Text>
```
orion llm assistant integrates the zephyr rag pipeline for Acme Support. The
implementation note writes llm and rag in lowercase, but they are abbreviations
inside the named system and pipeline. The note also mentions support tickets and
retrieved passages as ordinary inputs, not as standalone entities.
```

<Output>
entity{tuple_delimiter}Orion LLM Assistant{tuple_delimiter}work{tuple_delimiter}System described in the text as integrating the Zephyr RAG Pipeline for Acme Support.
entity{tuple_delimiter}Zephyr RAG Pipeline{tuple_delimiter}process{tuple_delimiter}Pipeline described in the text as integrated into Orion LLM Assistant.
entity{tuple_delimiter}Acme Support{tuple_delimiter}organization{tuple_delimiter}Named support organization described in the text as the target context for Orion LLM Assistant.
relation{tuple_delimiter}Orion LLM Assistant{tuple_delimiter}Zephyr RAG Pipeline{tuple_delimiter}integration, retrieval architecture{tuple_delimiter}The text states Orion LLM Assistant integrates the Zephyr RAG Pipeline.
relation{tuple_delimiter}Orion LLM Assistant{tuple_delimiter}Acme Support{tuple_delimiter}support deployment{tuple_delimiter}The text states Orion LLM Assistant integrates the Zephyr RAG Pipeline for Acme Support.
{completion_delimiter}
""",
    # Example 9 - Preserve documentation referent and uppercase names
    """<Entity_types>
["person","organization","location","event","artifact","work","naturalentity","concept","process"]

<Input Text>
```
OpenAI API documentation explains how BERT can be used for semantic search in
the Atlas Help Center Guide. The documentation records latency in milliseconds
as a performance attribute and refers to user questions as inputs. The referent
is the documentation itself, not the OpenAI API service alone.
```

<Output>
entity{tuple_delimiter}OpenAI API Documentation{tuple_delimiter}work{tuple_delimiter}Documentation described in the text as explaining how BERT can be used for semantic search.
entity{tuple_delimiter}BERT{tuple_delimiter}work{tuple_delimiter}Model described in the text as usable for semantic search according to the OpenAI API Documentation.
entity{tuple_delimiter}Semantic Search{tuple_delimiter}process{tuple_delimiter}Stable search procedure described in the text as a use case for BERT.
entity{tuple_delimiter}Atlas Help Center Guide{tuple_delimiter}work{tuple_delimiter}Guide described in the text as the context where the OpenAI API Documentation explains BERT usage for semantic search.
relation{tuple_delimiter}OpenAI API Documentation{tuple_delimiter}BERT{tuple_delimiter}usage guidance, semantic search{tuple_delimiter}The text states the OpenAI API Documentation explains how BERT can be used for semantic search.
relation{tuple_delimiter}BERT{tuple_delimiter}Semantic Search{tuple_delimiter}model usage{tuple_delimiter}The text states BERT can be used for semantic search.
relation{tuple_delimiter}OpenAI API Documentation{tuple_delimiter}Atlas Help Center Guide{tuple_delimiter}documentation context{tuple_delimiter}The text states the OpenAI API Documentation explains BERT usage in the Atlas Help Center Guide.
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
    Do not omit any important facts or details. Merge semantically duplicate facts
    rather than repeating them as separate claims.

4.  Perspective: write in objective third person.
    Begin the summary by explicitly naming the entity or relationship.
    For relationships, begin by explicitly naming both endpoints and the
    relationship being summarized.

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
