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
Source-grounded means explicitly named in the source as a concrete referent—not a
local role, filler, or modifier phrase.

- person        : named individual human (real, historical, or fictional).
- organization  : named group of humans acting as a collective unit.
- location      : named geographic or spatial place.
- event         : named occurrence anchored in time, even if temporally fuzzy.
                  One-time anchored occurrence = event; repeatable workflow = process.
- artifact      : human-made physical object you can hold or point at (chip, device, specimen).
                  Touchable physical object = artifact; citable/deployable intellectual output = work.
- work          : named intellectual output: papers, software, datasets, models, standards,
                  regulations, specifications, reports, plans, runbooks, postmortems.
                  Text refers to a document/plan/runbook/postmortem = work; executed method/workflow = process.
- naturalentity : named entity existing in the physical world independently of human production
                  (gene, species, chemical element, cell line).
                  Concrete biological/physical object = naturalentity; measurable property or abstraction = concept.
- concept       : stable named domain concept best answered by "what IS it"
                  (Attention Mechanism, Ionic Conductivity, Myocardial Fibrosis).
                  WHAT = concept; HOW = process; when both apply, prefer process.
- process       : named/reusable method, workflow, procedure, or analysis activity best
                  answered by "how is it done" (Gradient Descent, CRISPR-Cas9, Homologous Recombination).

Use ONLY the types provided in <Entity_types>. No other type values are permitted.

---What Must Never Be Extracted as an Entity---
Never extract the following as entity nodes:
- metric values (92.3%, 14ms, $2.4M): embed in descriptions instead.
- role titles (CEO, Director, Engineer): embed in person description and relation description.
- unnamed generics (a model, the query, retrieved documents, inputs, outputs, results): skip entirely.
- pure time labels (Q3 2024, FY2023, deadline): skip unless part of a full named entity
  (e.g. "Q3 2024 Product Review" = event, "Q3 2024 Root Cause Analysis" = process).
- file/layout noise (paths, filenames, page numbers, bbox, chunk IDs): skip entirely.
- negated entities: extract them normally as entities.
- negated relations: do NOT create a positive relation edge; note in entity description instead.

---Negation Rule---
  "GPT-3 was trained WITHOUT RLHF"
  → Extract both GPT-3 (work) and RLHF (process) as entities.
  → Do NOT create a GPT-3 –[trained_with]→ RLHF relationship edge.
  → In the GPT-3 description, note: "[negated context: does not use RLHF here]"

---Canonical Type Rule---
One surface name → one entity type, stable across the entire extraction run.
Never output the same surface name with two different types.
If an entity has dual identity, assign the type matching its PRIMARY FUNCTION
in the world and hold it globally.

---Ambiguity Protocol---
Use only when a candidate's validity or type is unclear.
  1) If the phrase is only a descriptor, modifier, role, or generic placeholder
     ("the query", "retrieved documents", "the generator"), DO NOT EXTRACT.
  2) For stable named referents, use the WHAT/HOW test:
     "X is ..." -> concept; "X works by ..." -> process; if both apply, prefer
     process; if neither applies, DO NOT EXTRACT.
  3) Keep only candidates that are central to the passage and can participate in
     source-supported relationships.

Generic roles, unnamed placeholders, and modifier phrases belong in
relationship_description, not as entity nodes.
A correctly dropped entity is always preferable to a wrongly typed one.

---Depth-1 Rule---
Extract only the outermost complete named entity.
  "EU AI Act Article 13" → extract "EU AI Act" (work), not "Article 13" alone.
Exception: extract a sub-component only when it is independently and widely
referenced by that name outside this document.

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
    The number of entities and relations depends solely on the input text's
    content richness. Do not stop early to match any expected count.

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


3.  Decompose N-ary relationships into binary pairs.
    "Alice, Bob, and Carol collaborated on Project X" →
    Alice–Project X, Bob–Project X, Carol–Project X.

4.  Write all entity names and descriptions in the third person.
    Do not use pronouns: "this article", "our company", "I", "you", "he/she".

5.  Output language: {language}.
    Proper nouns without a widely accepted translation stay in their original language.

6.  After all entities and relationships are output, write the completion signal:
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
    canonical type rule, depth-1 rule, negation rule, exclusion rules,
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
[entity extraction system prompt — the system message already present in this conversation]

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
---Task---
Review the previous extraction and add any missed or incorrectly formatted
entities and relationships from the same input text.
[... — the full text of this continuation message ...]

<Output>

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

    # Example 1 - ML system components with metric details embedded in relations
    """<Entity_types>
["person","organization","location","event","artifact","work","naturalentity","concept","process"]

<Input Text>
```
Corrective Retrieval-Augmented Generation (CRAG) uses a Retrieval Evaluator module
to score retrieval quality. When retrieval quality is poor, CRAG invokes a Query
Rewriter module before passing evidence to a Knowledge Refinement Module. The
Answer Composer module uses refined evidence to produce final answers. The paper
evaluates CRAG on Natural Questions and HotpotQA. In the comparison section, CRAG
reports lower hallucination than a baseline RAG Pipeline.
```

<Output>
entity{tuple_delimiter}Corrective Retrieval-Augmented Generation{tuple_delimiter}process{tuple_delimiter}Framework described in the text as using retrieval evaluation, query rewriting, knowledge refinement, answer composition, and benchmark evaluation.
entity{tuple_delimiter}Retrieval Evaluator{tuple_delimiter}process{tuple_delimiter}Module described in the text as scoring retrieval quality for Corrective Retrieval-Augmented Generation.
entity{tuple_delimiter}Query Rewriter{tuple_delimiter}process{tuple_delimiter}Module described in the text as invoked by Corrective Retrieval-Augmented Generation when retrieval quality is poor.
entity{tuple_delimiter}Knowledge Refinement Module{tuple_delimiter}process{tuple_delimiter}Module described in the text as receiving evidence before answer composition.
entity{tuple_delimiter}Answer Composer{tuple_delimiter}process{tuple_delimiter}Module described in the text as using refined evidence to produce final answers.
entity{tuple_delimiter}Natural Questions{tuple_delimiter}work{tuple_delimiter}Benchmark described in the text as used to evaluate Corrective Retrieval-Augmented Generation.
entity{tuple_delimiter}HotpotQA{tuple_delimiter}work{tuple_delimiter}Benchmark described in the text as used to evaluate Corrective Retrieval-Augmented Generation.
entity{tuple_delimiter}Baseline RAG Pipeline{tuple_delimiter}process{tuple_delimiter}Pipeline described in the text as the baseline used for comparison against Corrective Retrieval-Augmented Generation.
relation{tuple_delimiter}Corrective Retrieval-Augmented Generation{tuple_delimiter}Retrieval Evaluator{tuple_delimiter}component, retrieval assessment{tuple_delimiter}The text states Corrective Retrieval-Augmented Generation uses a Retrieval Evaluator module to score retrieval quality.
relation{tuple_delimiter}Corrective Retrieval-Augmented Generation{tuple_delimiter}Query Rewriter{tuple_delimiter}component, query rewriting{tuple_delimiter}The text states Corrective Retrieval-Augmented Generation invokes a Query Rewriter when retrieval quality is poor.
relation{tuple_delimiter}Corrective Retrieval-Augmented Generation{tuple_delimiter}Knowledge Refinement Module{tuple_delimiter}component, knowledge refinement{tuple_delimiter}The text states Corrective Retrieval-Augmented Generation passes evidence to a Knowledge Refinement Module.
relation{tuple_delimiter}Corrective Retrieval-Augmented Generation{tuple_delimiter}Answer Composer{tuple_delimiter}component, answer composition{tuple_delimiter}The text states Corrective Retrieval-Augmented Generation uses an Answer Composer module to produce final answers.
relation{tuple_delimiter}Knowledge Refinement Module{tuple_delimiter}Answer Composer{tuple_delimiter}evidence handoff, answer generation{tuple_delimiter}The text states refined evidence is used by the Answer Composer.
relation{tuple_delimiter}Corrective Retrieval-Augmented Generation{tuple_delimiter}Natural Questions{tuple_delimiter}evaluation benchmark{tuple_delimiter}The text states Corrective Retrieval-Augmented Generation is evaluated on Natural Questions.
relation{tuple_delimiter}Corrective Retrieval-Augmented Generation{tuple_delimiter}HotpotQA{tuple_delimiter}evaluation benchmark{tuple_delimiter}The text states Corrective Retrieval-Augmented Generation is evaluated on HotpotQA.
relation{tuple_delimiter}Corrective Retrieval-Augmented Generation{tuple_delimiter}Baseline RAG Pipeline{tuple_delimiter}comparative evaluation, hallucination reduction{tuple_delimiter}The text states Corrective Retrieval-Augmented Generation reports lower hallucination than the baseline RAG Pipeline.
{completion_delimiter}
""",

    # Example 2 - Enterprise KPI: metrics in descriptions, role in person entity, casing normalization
    """<Entity_types>
["person","organization","location","event","artifact","work","naturalentity","concept","process"]

<Input Text>
```
the q2 service performance report states that Orion Support Platform resolved
98.7% of tickets within 24 hours and recorded an average response time of
11 minutes. Elena Ruiz, VP of Customer Operations at northstar systems,
presented the report during the Monthly Service Review. The review compared
Orion Support Platform against Legacy Queue Desk on escalation rate and
first-response speed.
```

<Output>
entity{tuple_delimiter}Q2 Service Performance Report{tuple_delimiter}work{tuple_delimiter}Report described in the text as presenting Orion Support Platform ticket-resolution and response-time metrics.
entity{tuple_delimiter}Orion Support Platform{tuple_delimiter}work{tuple_delimiter}Service platform described in the text as the evaluated subject in the Q2 Service Performance Report and Monthly Service Review.
entity{tuple_delimiter}Elena Ruiz{tuple_delimiter}person{tuple_delimiter}Elena Ruiz, VP of Customer Operations at Northstar Systems per the text, presented the Q2 Service Performance Report during the Monthly Service Review.
entity{tuple_delimiter}Northstar Systems{tuple_delimiter}organization{tuple_delimiter}Organization described in the text as the company where Elena Ruiz serves as VP of Customer Operations.
entity{tuple_delimiter}Monthly Service Review{tuple_delimiter}event{tuple_delimiter}Event described in the text as the meeting where the Q2 Service Performance Report was presented and platform performance was compared.
entity{tuple_delimiter}Legacy Queue Desk{tuple_delimiter}work{tuple_delimiter}System described in the text as the comparator against Orion Support Platform in the Monthly Service Review.
relation{tuple_delimiter}Q2 Service Performance Report{tuple_delimiter}Orion Support Platform{tuple_delimiter}performance reporting, service metrics{tuple_delimiter}The text states the Q2 Service Performance Report presents Orion Support Platform ticket-resolution and response-time metrics.
relation{tuple_delimiter}Elena Ruiz{tuple_delimiter}Northstar Systems{tuple_delimiter}leadership role{tuple_delimiter}The text identifies Elena Ruiz as VP of Customer Operations at Northstar Systems.
relation{tuple_delimiter}Elena Ruiz{tuple_delimiter}Q2 Service Performance Report{tuple_delimiter}report presentation{tuple_delimiter}The text states Elena Ruiz presented the Q2 Service Performance Report.
relation{tuple_delimiter}Elena Ruiz{tuple_delimiter}Monthly Service Review{tuple_delimiter}meeting presentation{tuple_delimiter}The text states Elena Ruiz presented during the Monthly Service Review.
relation{tuple_delimiter}Q2 Service Performance Report{tuple_delimiter}Monthly Service Review{tuple_delimiter}report context{tuple_delimiter}The text states the Q2 Service Performance Report was presented during the Monthly Service Review.
relation{tuple_delimiter}Monthly Service Review{tuple_delimiter}Orion Support Platform{tuple_delimiter}performance comparison{tuple_delimiter}The text states the Monthly Service Review compared Orion Support Platform on escalation rate and first-response speed.
relation{tuple_delimiter}Monthly Service Review{tuple_delimiter}Legacy Queue Desk{tuple_delimiter}performance comparison{tuple_delimiter}The text states the Monthly Service Review compared Legacy Queue Desk on escalation rate and first-response speed.
relation{tuple_delimiter}Orion Support Platform{tuple_delimiter}Legacy Queue Desk{tuple_delimiter}comparative evaluation{tuple_delimiter}The text states Orion Support Platform was compared against Legacy Queue Desk in the Monthly Service Review.
{completion_delimiter}
""",

    # Example 3 - Enterprise event/work/process extraction with time-label handling
    """<Entity_types>
["person","organization","location","event","artifact","work","naturalentity","concept","process"]

<Input Text>
```
During Q3 2024, the Q3 2024 Product Review concluded that the Apollo Platform
failed the Apollo Uptime SLA. The Q3 2024 Root Cause Analysis activity, led by
Diana Chen, identified a Memory Leak in the Data Ingestion Pipeline as the
primary fault. The Apollo Remediation Plan was submitted to the Engineering
Steering Committee and requires teams to patch the Data Ingestion Pipeline and
update the Canary Release Procedure. The Apollo Postmortem report cites both the
Q3 2024 Root Cause Analysis and the Apollo Remediation Plan.
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
entity{tuple_delimiter}Apollo Postmortem{tuple_delimiter}work{tuple_delimiter}Report described in the text as citing both the Q3 2024 Root Cause Analysis and the Apollo Remediation Plan.
entity{tuple_delimiter}Canary Release Procedure{tuple_delimiter}process{tuple_delimiter}Operational procedure described in the text as updated according to the Apollo Remediation Plan.
relation{tuple_delimiter}Q3 2024 Product Review{tuple_delimiter}Apollo Platform{tuple_delimiter}evaluation, SLA failure{tuple_delimiter}The text states the Q3 2024 Product Review concluded that the Apollo Platform failed the Apollo Uptime SLA.
relation{tuple_delimiter}Q3 2024 Product Review{tuple_delimiter}Apollo Uptime SLA{tuple_delimiter}evaluation criterion{tuple_delimiter}The text states the Q3 2024 Product Review concluded failure against the Apollo Uptime SLA.
relation{tuple_delimiter}Apollo Platform{tuple_delimiter}Apollo Uptime SLA{tuple_delimiter}SLA failure{tuple_delimiter}The text states the Apollo Platform failed the Apollo Uptime SLA.
relation{tuple_delimiter}Q3 2024 Root Cause Analysis{tuple_delimiter}Diana Chen{tuple_delimiter}analysis leadership{tuple_delimiter}The text states Diana Chen led the Q3 2024 Root Cause Analysis.
relation{tuple_delimiter}Q3 2024 Root Cause Analysis{tuple_delimiter}Memory Leak{tuple_delimiter}fault identification{tuple_delimiter}The text states the Q3 2024 Root Cause Analysis identified a Memory Leak as the primary fault.
relation{tuple_delimiter}Q3 2024 Root Cause Analysis{tuple_delimiter}Data Ingestion Pipeline{tuple_delimiter}fault localization{tuple_delimiter}The text states the Q3 2024 Root Cause Analysis identified the primary fault in the Data Ingestion Pipeline.
relation{tuple_delimiter}Memory Leak{tuple_delimiter}Data Ingestion Pipeline{tuple_delimiter}fault location{tuple_delimiter}The text states the Memory Leak was in the Data Ingestion Pipeline.
relation{tuple_delimiter}Apollo Remediation Plan{tuple_delimiter}Engineering Steering Committee{tuple_delimiter}submission, governance{tuple_delimiter}The text states the Apollo Remediation Plan was submitted to the Engineering Steering Committee.
relation{tuple_delimiter}Apollo Remediation Plan{tuple_delimiter}Data Ingestion Pipeline{tuple_delimiter}patch requirement{tuple_delimiter}The text states the Apollo Remediation Plan requires teams to patch the Data Ingestion Pipeline.
relation{tuple_delimiter}Apollo Remediation Plan{tuple_delimiter}Canary Release Procedure{tuple_delimiter}procedure update requirement{tuple_delimiter}The text states the Apollo Remediation Plan requires teams to update the Canary Release Procedure.
relation{tuple_delimiter}Apollo Postmortem{tuple_delimiter}Q3 2024 Root Cause Analysis{tuple_delimiter}citation{tuple_delimiter}The text states the Apollo Postmortem cites the Q3 2024 Root Cause Analysis.
relation{tuple_delimiter}Apollo Postmortem{tuple_delimiter}Apollo Remediation Plan{tuple_delimiter}citation{tuple_delimiter}The text states the Apollo Postmortem cites the Apollo Remediation Plan.
{completion_delimiter}
""",

    # Example 4 - Negated decisions do not create positive execution edges
    """<Entity_types>
["person","organization","location","event","artifact","work","naturalentity","concept","process"]

<Input Text>
```
The FY2025 Cloud Budget Proposal examined migrating the Audit Service from
Archive Cluster to Nimbus Object Store. In the April Decision Meeting, the
Finance Committee did not approve the migration request. The committee approved
a one-quarter support extension for Archive Cluster instead. Priya Nair
presented the FY2025 Cloud Budget Proposal at the same meeting.
```

<Output>
entity{tuple_delimiter}FY2025 Cloud Budget Proposal{tuple_delimiter}work{tuple_delimiter}Budget proposal described in the text as examining migration of the Audit Service from Archive Cluster to Nimbus Object Store. [negated context: the migration request was not approved in the April Decision Meeting]
entity{tuple_delimiter}Audit Service{tuple_delimiter}work{tuple_delimiter}Service described in the text as the migration subject in the FY2025 Cloud Budget Proposal.
entity{tuple_delimiter}Archive Cluster{tuple_delimiter}work{tuple_delimiter}Infrastructure system described in the text as the migration source and as receiving an approved one-quarter support extension.
entity{tuple_delimiter}Nimbus Object Store{tuple_delimiter}work{tuple_delimiter}Target storage system described in the text as the proposed migration destination for the Audit Service.
entity{tuple_delimiter}April Decision Meeting{tuple_delimiter}event{tuple_delimiter}Meeting described in the text as the event where the Finance Committee rejected the migration request and approved Archive Cluster support extension.
entity{tuple_delimiter}Finance Committee{tuple_delimiter}organization{tuple_delimiter}Committee described in the text as making approval decisions in the April Decision Meeting.
entity{tuple_delimiter}Priya Nair{tuple_delimiter}person{tuple_delimiter}Priya Nair per the text presented the FY2025 Cloud Budget Proposal at the April Decision Meeting.
relation{tuple_delimiter}FY2025 Cloud Budget Proposal{tuple_delimiter}Audit Service{tuple_delimiter}migration subject{tuple_delimiter}The text states the FY2025 Cloud Budget Proposal examined migrating the Audit Service.
relation{tuple_delimiter}FY2025 Cloud Budget Proposal{tuple_delimiter}Archive Cluster{tuple_delimiter}migration source{tuple_delimiter}The text states the FY2025 Cloud Budget Proposal examined migration from Archive Cluster.
relation{tuple_delimiter}FY2025 Cloud Budget Proposal{tuple_delimiter}Nimbus Object Store{tuple_delimiter}migration target proposal{tuple_delimiter}The text states the FY2025 Cloud Budget Proposal examined migration to Nimbus Object Store.
relation{tuple_delimiter}FY2025 Cloud Budget Proposal{tuple_delimiter}April Decision Meeting{tuple_delimiter}decision context{tuple_delimiter}The text states the FY2025 Cloud Budget Proposal was discussed in the April Decision Meeting.
relation{tuple_delimiter}Finance Committee{tuple_delimiter}April Decision Meeting{tuple_delimiter}decision authority{tuple_delimiter}The text states the Finance Committee made decisions in the April Decision Meeting.
relation{tuple_delimiter}Finance Committee{tuple_delimiter}Archive Cluster{tuple_delimiter}support extension approval{tuple_delimiter}The text states the Finance Committee approved a one-quarter support extension for Archive Cluster.
relation{tuple_delimiter}Priya Nair{tuple_delimiter}FY2025 Cloud Budget Proposal{tuple_delimiter}proposal presentation{tuple_delimiter}The text states Priya Nair presented the FY2025 Cloud Budget Proposal.
relation{tuple_delimiter}Priya Nair{tuple_delimiter}April Decision Meeting{tuple_delimiter}meeting presentation{tuple_delimiter}The text states Priya Nair presented at the April Decision Meeting.
{completion_delimiter}
""",

    # Example 5 - Biomedical trial with location: covers naturalentity, artifact, concept, location; casing normalization shown in Ex2
    """<Entity_types>
["person","organization","location","event","artifact","work","naturalentity","concept","process"]

<Input Text>
```
The CARDIO-RNA Trial, conducted in Berlin and led by Sofia Meier, studied
RNX-41 for Myocardial Fibrosis and compared its effects with Doxorubicin in
a controlled arm. Researchers used CRISPR-Cas9 to knock out BRCA1 in the
MCF-7 Cell Line, then measured Homologous Recombination activity. A separate
assay reported that Doxorubicin induced Apoptosis in the MCF-7 Cell Line at
Week 12 with p<0.01 significance. All procedures followed the ICH E6 GCP Guideline.
```

<Output>
entity{tuple_delimiter}CARDIO-RNA Trial{tuple_delimiter}event{tuple_delimiter}Trial described in the text as studying RNX-41 for Myocardial Fibrosis and comparing effects with Doxorubicin in a controlled arm.
entity{tuple_delimiter}Berlin{tuple_delimiter}location{tuple_delimiter}Location described in the text as where the CARDIO-RNA Trial was conducted.
entity{tuple_delimiter}Sofia Meier{tuple_delimiter}person{tuple_delimiter}Individual described in the text as leading the CARDIO-RNA Trial.
entity{tuple_delimiter}RNX-41{tuple_delimiter}artifact{tuple_delimiter}Therapeutic compound described in the text as studied in the CARDIO-RNA Trial for Myocardial Fibrosis.
entity{tuple_delimiter}Myocardial Fibrosis{tuple_delimiter}concept{tuple_delimiter}Disease concept described in the text as the target condition studied in the CARDIO-RNA Trial.
entity{tuple_delimiter}Doxorubicin{tuple_delimiter}artifact{tuple_delimiter}Drug described in the text as used in a controlled arm and as inducing Apoptosis in the MCF-7 Cell Line.
entity{tuple_delimiter}CRISPR-Cas9{tuple_delimiter}process{tuple_delimiter}Gene editing procedure described in the text as used to knock out BRCA1 in the MCF-7 Cell Line.
entity{tuple_delimiter}BRCA1{tuple_delimiter}naturalentity{tuple_delimiter}Gene described in the text as knocked out by CRISPR-Cas9 in the MCF-7 Cell Line.
entity{tuple_delimiter}MCF-7 Cell Line{tuple_delimiter}naturalentity{tuple_delimiter}Cell line described in the text as used in the CRISPR-Cas9 knockout and Doxorubicin assays.
entity{tuple_delimiter}Homologous Recombination{tuple_delimiter}process{tuple_delimiter}DNA repair process described in the text as measured after BRCA1 knockout in the MCF-7 Cell Line.
entity{tuple_delimiter}Apoptosis{tuple_delimiter}process{tuple_delimiter}Biological process described in the text as induced by Doxorubicin in the MCF-7 Cell Line.
entity{tuple_delimiter}ICH E6 GCP Guideline{tuple_delimiter}work{tuple_delimiter}Regulatory guideline described in the text as followed by all trial procedures.
relation{tuple_delimiter}CARDIO-RNA Trial{tuple_delimiter}RNX-41{tuple_delimiter}trial intervention{tuple_delimiter}The text states the CARDIO-RNA Trial studied RNX-41.
relation{tuple_delimiter}CARDIO-RNA Trial{tuple_delimiter}Myocardial Fibrosis{tuple_delimiter}target condition{tuple_delimiter}The text states the CARDIO-RNA Trial studied RNX-41 for Myocardial Fibrosis.
relation{tuple_delimiter}RNX-41{tuple_delimiter}Myocardial Fibrosis{tuple_delimiter}therapeutic target{tuple_delimiter}The text states RNX-41 was studied for Myocardial Fibrosis.
relation{tuple_delimiter}CARDIO-RNA Trial{tuple_delimiter}Doxorubicin{tuple_delimiter}controlled arm{tuple_delimiter}The text states the trial compared effects with Doxorubicin in a controlled arm.
relation{tuple_delimiter}CARDIO-RNA Trial{tuple_delimiter}Berlin{tuple_delimiter}trial location{tuple_delimiter}The text states the CARDIO-RNA Trial was conducted in Berlin.
relation{tuple_delimiter}CARDIO-RNA Trial{tuple_delimiter}Sofia Meier{tuple_delimiter}trial leadership{tuple_delimiter}The text states the CARDIO-RNA Trial was led by Sofia Meier.
relation{tuple_delimiter}CRISPR-Cas9{tuple_delimiter}BRCA1{tuple_delimiter}gene knockout{tuple_delimiter}The text states researchers used CRISPR-Cas9 to knock out BRCA1.
relation{tuple_delimiter}CRISPR-Cas9{tuple_delimiter}MCF-7 Cell Line{tuple_delimiter}experimental model{tuple_delimiter}The text states CRISPR-Cas9 was used in the MCF-7 Cell Line.
relation{tuple_delimiter}BRCA1{tuple_delimiter}MCF-7 Cell Line{tuple_delimiter}knockout context{tuple_delimiter}The text states BRCA1 was knocked out in the MCF-7 Cell Line.
relation{tuple_delimiter}BRCA1{tuple_delimiter}Homologous Recombination{tuple_delimiter}functional assessment{tuple_delimiter}The text states Homologous Recombination activity was measured after BRCA1 knockout.
relation{tuple_delimiter}Doxorubicin{tuple_delimiter}Apoptosis{tuple_delimiter}drug effect{tuple_delimiter}The text states Doxorubicin induced Apoptosis.
relation{tuple_delimiter}Doxorubicin{tuple_delimiter}MCF-7 Cell Line{tuple_delimiter}assay model{tuple_delimiter}The text states Doxorubicin induced Apoptosis in the MCF-7 Cell Line.
relation{tuple_delimiter}CARDIO-RNA Trial{tuple_delimiter}ICH E6 GCP Guideline{tuple_delimiter}protocol compliance{tuple_delimiter}The text states all procedures followed the ICH E6 GCP Guideline.
{completion_delimiter}
""",

]

PROMPTS["entity_extraction_normalization_examples"] = []

# Fallback copy used when integrations accidentally omit the primary
# normalization-example key during prompt assembly.
PROMPTS["entity_extraction_normalization_examples_fallback"] = list(
    PROMPTS["entity_extraction_normalization_examples"]
)

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
  - Generate a **References** section at the end of the response. ONLY include reference_id values that actually appear in the provided `Reference Document List`. Do NOT invent or hallucinate reference_id values that are not in the list.
  - Each reference document must directly support the facts presented in the response.
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
  - The Document Title must be taken VERBATIM from the `Reference Document List`. Do NOT invent document titles.
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
1. **high_level_keywords**: overarching themes and relational concepts that describe the query's intent. Use lowercase short phrases (e.g., "gene knockout", "performance metrics", "api integration").
2. **low_level_keywords**: specific named entities mentioned or implied by the query. Use title-cased style. Covers any named referent of these types: person, organization, location, event, artifact, work, naturalentity, concept, process.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON. It will be parsed directly by a JSON parser.
2. **Source of Truth**: Keywords must be grounded in the user query. Direct mentions are always included; reasonable inferences are permitted—for low_level_keywords, infer entity-type terms (person, organization, location, event, artifact, work, naturalentity, concept, process); for high_level_keywords, infer related relational themes. Both categories must contain at least one keyword.
3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from "What did Elena Ruiz present about the Orion Support Platform in the Q2 Service Performance Report?", extract "Elena Ruiz", "Orion Support Platform", and "Q2 Service Performance Report" as low_level_keywords and "report presentation" as a high_level_keyword — not individual words like "present", "Q2", or "Platform".
4. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), you must return a JSON object with empty lists for both keyword types.
5. **Language**: All extracted keywords MUST be in {language}. Proper nouns (e.g., personal names, place names, organization names) should be kept in their original language.
6. **Casing (high_level_keywords)**: Use lowercase phrases by default. Preserve meaningful uppercase or mixed-case proper nouns/acronyms (e.g., OpenAI, BERT, API, 6G).
7. **Casing (low_level_keywords)**: Use entity-style casing. Preserve mixed-case proper nouns/acronyms; otherwise normalize case-insensitive phrases to canonical title-style wording.
8. **Exclusions**: Do not include as keywords: raw metric values (90%, $2M, 14ms), standalone role titles without a name (CEO, Director), unnamed generic placeholders (the model, the system, the team), standalone time labels (Q3 2024, FY2023) unless they are part of a full named entity (e.g. "Q3 2024 Root Cause Analysis"). These are not entity-type terms and make poor keywords.
9. **Complete Names**: For low_level_keywords, use the outermost complete named entity. Prefer "EU AI Act" over "Article 13"; prefer "Q3 2024 Root Cause Analysis" over "Q3 2024" alone.

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "What are the environmental consequences of deforestation on biodiversity?"

Output:
{
  "high_level_keywords": ["environmental consequences", "biodiversity impact", "ecological loss"],
  "low_level_keywords": ["Deforestation", "Biodiversity", "Carbon Emissions", "Habitat Loss"]
}

""",
    """Example 2:

Query: "What performance metrics did the Orion Support Platform achieve in the Q2 Service Performance Report?"

Output:
{
  "high_level_keywords": ["performance metrics", "service performance", "operational reporting"],
  "low_level_keywords": ["Orion Support Platform", "Q2 Service Performance Report"]
}

""",
    """Example 3:

Query: "How was CRISPR-Cas9 used to knock out BRCA1 in the CARDIO-RNA Trial?"

Output:
{
  "high_level_keywords": ["gene editing application", "gene knockout", "clinical trial methodology"],
  "low_level_keywords": ["CRISPR-Cas9", "BRCA1", "CARDIO-RNA Trial"]
}

""",
    """Example 4:

Query: "How can OpenAI API and BERT be used for semantic search in a 6G assistant?"

Output:
{
  "high_level_keywords": ["model usage", "api integration", "assistant design"],
  "low_level_keywords": ["OpenAI API", "BERT", "Semantic Search", "6G Assistant"]
}

""",
]
