"""
Prompt templates optimized for Qwen2-VL-7B and other lightweight VLMs
=====================================================================

IMPORTANT RESEARCH NOTE:
------------------------
This file contains prompt engineering optimizations specifically designed for 
lightweight Vision-Language Models (VLMs) like Qwen2-VL-7B.

RESEARCH CONTRIBUTION:
- Demonstrates how proper prompt constraints can improve structured output 
  reliability from weaker models without increasing model size
- Addresses the "output divergence" problem where lightweight VLMs generate 
  extra explanatory text instead of pure JSON
- Can be used as a baseline for comparing prompt engineering techniques

MODIFICATIONS FROM OFFICIAL prompt.py:
--------------------------------------
1. Added explicit JSON output constraints (START/END markers)
2. Added prohibition statements (NO text before/after JSON)
3. Added format warnings (NO markdown blocks)
4. Simplified language for better model comprehension
5. Retained all original content requirements

USAGE:
------
To use these optimized prompts:
1. Import this module instead of the default prompt.py in modalprocessors.py
2. Or selectively replace specific prompts in PROMPTS dictionary

BASELINE FOR COMPARISON:
------------------------
Official prompts: raganything/prompt.py
Optimized prompts: raganything/prompt_qwen2_vl.py

Author: [Your Name]
Date: 2026-02-02
Purpose: DocBench Evaluation with Qwen2-VL-7B
"""

from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

# =============================================================================
# System prompts (unchanged - these work fine)
# =============================================================================

PROMPTS["IMAGE_ANALYSIS_SYSTEM"] = (
    "You are an expert image analyst. Provide detailed, accurate descriptions."
)
PROMPTS["IMAGE_ANALYSIS_FALLBACK_SYSTEM"] = (
    "You are an expert image analyst. Provide detailed analysis based on available information."
)
PROMPTS["TABLE_ANALYSIS_SYSTEM"] = (
    "You are an expert data analyst. Provide detailed table analysis with specific insights."
)
PROMPTS["EQUATION_ANALYSIS_SYSTEM"] = (
    "You are an expert mathematician. Provide detailed mathematical analysis."
)
PROMPTS["GENERIC_ANALYSIS_SYSTEM"] = (
    "You are an expert content analyst specializing in {content_type} content."
)

# =============================================================================
# OPTIMIZED: Image analysis prompt
# =============================================================================
# CHANGES: Added strict JSON output constraints at beginning and end
# REASON: Prevent model from adding explanations before/after JSON

PROMPTS[
    "vision_prompt"
] = """CRITICAL INSTRUCTION: Your response MUST be a valid JSON object only. Start with {{ and end with }}. No text before or after.

Analyze this image and provide a JSON response:

{{
    "detailed_description": "A comprehensive and detailed visual description of the image following these guidelines:
    - Describe the overall composition and layout
    - Identify all objects, people, text, and visual elements
    - Explain relationships between elements
    - Note colors, lighting, and visual style
    - Describe any actions or activities shown
    - Include technical details if relevant (charts, diagrams, etc.)
    - Always use specific names instead of pronouns",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "concise summary of the image content and its significance (max 100 words)"
    }}
}}

Additional context:
- Image Path: {image_path}
- Captions: {captions}
- Footnotes: {footnotes}

REMINDER: Output ONLY the JSON object above. Do not add explanations, commentary, or markdown code blocks."""

# =============================================================================
# OPTIMIZED: Image analysis with context
# =============================================================================

PROMPTS[
    "vision_prompt_with_context"
] = """CRITICAL INSTRUCTION: Your response MUST be a valid JSON object only. Start with {{ and end with }}. No text before or after.

Analyze this image considering the context:

{{
    "detailed_description": "A comprehensive and detailed visual description of the image following these guidelines:
    - Describe the overall composition and layout
    - Identify all objects, people, text, and visual elements
    - Explain relationships between elements and how they relate to the surrounding context
    - Note colors, lighting, and visual style
    - Describe any actions or activities shown
    - Include technical details if relevant (charts, diagrams, etc.)
    - Reference connections to the surrounding content when relevant
    - Always use specific names instead of pronouns",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "concise summary of the image content, its significance, and relationship to surrounding content (max 100 words)"
    }}
}}

Context from surrounding content:
{context}

Image details:
- Image Path: {image_path}
- Captions: {captions}
- Footnotes: {footnotes}

REMINDER: Output ONLY the JSON object. No additional text."""

# =============================================================================
# UNCHANGED: Text fallback
# =============================================================================

PROMPTS["text_prompt"] = """Based on the following image information, provide analysis:

Image Path: {image_path}
Captions: {captions}
Footnotes: {footnotes}

{vision_prompt}"""

# =============================================================================
# OPTIMIZED: Table analysis prompt
# =============================================================================

PROMPTS[
    "table_prompt"
] = """CRITICAL INSTRUCTION: Your response MUST be a valid JSON object only. Start with {{ and end with }}. No text before or after.

Analyze this table and provide a JSON response:

{{
    "detailed_description": "A comprehensive analysis of the table including:
    - Table structure and organization
    - Column headers and their meanings
    - Key data points and patterns
    - Statistical insights and trends
    - Relationships between data elements
    - Significance of the data presented
    Always use specific names and values instead of general references.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "concise summary of the table's purpose and key findings (max 100 words)"
    }}
}}

Table Information:
Image Path: {table_img_path}
Caption: {table_caption}
Body: {table_body}
Footnotes: {table_footnote}

REMINDER: Output ONLY the JSON object. No additional text."""

# =============================================================================
# OPTIMIZED: Table analysis with context
# =============================================================================

PROMPTS[
    "table_prompt_with_context"
] = """CRITICAL INSTRUCTION: Your response MUST be a valid JSON object only. Start with {{ and end with }}. No text before or after.

Analyze this table considering the context:

{{
    "detailed_description": "A comprehensive analysis of the table including:
    - Table structure and organization
    - Column headers and their meanings
    - Key data points and patterns
    - Statistical insights and trends
    - Relationships between data elements
    - Significance of the data presented in relation to surrounding context
    - How the table supports or illustrates concepts from the surrounding content
    Always use specific names and values instead of general references.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "concise summary of the table's purpose, key findings, and relationship to surrounding content (max 100 words)"
    }}
}}

Context from surrounding content:
{context}

Table Information:
Image Path: {table_img_path}
Caption: {table_caption}
Body: {table_body}
Footnotes: {table_footnote}

REMINDER: Output ONLY the JSON object. No additional text."""

# =============================================================================
# OPTIMIZED: Equation analysis prompt
# =============================================================================

PROMPTS[
    "equation_prompt"
] = """CRITICAL INSTRUCTION: Your response MUST be a valid JSON object only. Start with {{ and end with }}. No text before or after.

Analyze this equation and provide a JSON response:

{{
    "detailed_description": "A comprehensive analysis of the equation including:
    - Mathematical meaning and interpretation
    - Variables and their definitions
    - Mathematical operations and functions used
    - Application domain and context
    - Physical or theoretical significance
    - Relationship to other mathematical concepts
    - Practical applications or use cases
    Always use specific mathematical terminology.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "equation",
        "summary": "concise summary of the equation's purpose and significance (max 100 words)"
    }}
}}

Equation Information:
Equation: {equation_text}
Format: {equation_format}

REMINDER: Output ONLY the JSON object. No additional text."""

# =============================================================================
# OPTIMIZED: Equation analysis with context
# =============================================================================

PROMPTS[
    "equation_prompt_with_context"
] = """CRITICAL INSTRUCTION: Your response MUST be a valid JSON object only. Start with {{ and end with }}. No text before or after.

Analyze this equation considering the context:

{{
    "detailed_description": "A comprehensive analysis of the equation including:
    - Mathematical meaning and interpretation
    - Variables and their definitions in the context of surrounding content
    - Mathematical operations and functions used
    - Application domain and context based on surrounding material
    - Physical or theoretical significance
    - Relationship to other mathematical concepts mentioned in the context
    - Practical applications or use cases
    - How the equation relates to the broader discussion or framework
    Always use specific mathematical terminology.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "equation",
        "summary": "concise summary of the equation's purpose, significance, and role in the surrounding context (max 100 words)"
    }}
}}

Context from surrounding content:
{context}

Equation Information:
Equation: {equation_text}
Format: {equation_format}

REMINDER: Output ONLY the JSON object. No additional text."""

# =============================================================================
# OPTIMIZED: Generic content analysis (MOST IMPORTANT)
# =============================================================================
# This is the most commonly used prompt and the most likely to cause issues
# with lightweight VLMs like Qwen2-VL-7B

PROMPTS[
    "generic_prompt"
] = """CRITICAL OUTPUT REQUIREMENTS:
1. Your response MUST be ONLY a valid JSON object
2. Start your response with {{
3. End your response with }}
4. NO text, explanation, or commentary before the JSON
5. NO text, explanation, or commentary after the JSON
6. NO markdown code blocks (do not use ```json)

Analyze this {content_type} content:

{{
    "detailed_description": "A comprehensive analysis of the content including:
    - Content structure and organization
    - Key information and elements
    - Relationships between components
    - Context and significance
    - Relevant details for knowledge retrieval
    Always use specific terminology appropriate for {content_type} content.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "{content_type}",
        "summary": "concise summary of the content's purpose and key points (max 100 words)"
    }}
}}

Content to analyze: {content}

FINAL REMINDER: Output ONLY the JSON object shown above. Start with {{ and end with }}. Nothing else."""

# =============================================================================
# OPTIMIZED: Generic content with context
# =============================================================================

PROMPTS[
    "generic_prompt_with_context"
] = """CRITICAL OUTPUT REQUIREMENTS:
1. Your response MUST be ONLY a valid JSON object
2. Start your response with {{
3. End your response with }}
4. NO text, explanation, or commentary before the JSON
5. NO text, explanation, or commentary after the JSON
6. NO markdown code blocks (do not use ```json)

Analyze this {content_type} content considering the context:

{{
    "detailed_description": "A comprehensive analysis of the content including:
    - Content structure and organization
    - Key information and elements
    - Relationships between components
    - Context and significance in relation to surrounding content
    - How this content connects to or supports the broader discussion
    - Relevant details for knowledge retrieval
    Always use specific terminology appropriate for {content_type} content.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "{content_type}",
        "summary": "concise summary of the content's purpose, key points, and relationship to surrounding context (max 100 words)"
    }}
}}

Context from surrounding content:
{context}

Content to analyze: {content}

FINAL REMINDER: Output ONLY the JSON object shown above. Start with {{ and end with }}. Nothing else."""

# =============================================================================
# Modal chunk templates (unchanged)
# =============================================================================

PROMPTS["image_chunk"] = """
Image Content Analysis:
Image Path: {image_path}
Captions: {captions}
Footnotes: {footnotes}

Visual Analysis: {enhanced_caption}"""

PROMPTS["table_chunk"] = """Table Analysis:
Image Path: {table_img_path}
Caption: {table_caption}
Structure: {table_body}
Footnotes: {table_footnote}

Analysis: {enhanced_caption}"""

PROMPTS["equation_chunk"] = """Mathematical Equation Analysis:
Equation: {equation_text}
Format: {equation_format}

Mathematical Analysis: {enhanced_caption}"""

PROMPTS["generic_chunk"] = """{content_type} Content Analysis:
Content: {content}

Analysis: {enhanced_caption}"""

# =============================================================================
# Query-related prompts (unchanged - these are fine for queries)
# =============================================================================

PROMPTS["QUERY_IMAGE_DESCRIPTION"] = (
    "Please briefly describe the main content, key elements, and important information in this image."
)

PROMPTS["QUERY_IMAGE_ANALYST_SYSTEM"] = (
    "You are a professional image analyst who can accurately describe image content."
)

PROMPTS[
    "QUERY_TABLE_ANALYSIS"
] = """Please analyze the main content, structure, and key information of the following table data:

Table data:
{table_data}

Table caption: {table_caption}

Please briefly summarize the main content, data characteristics, and important findings of the table."""

PROMPTS["QUERY_TABLE_ANALYST_SYSTEM"] = (
    "You are a professional data analyst who can accurately analyze table data."
)

PROMPTS[
    "QUERY_EQUATION_ANALYSIS"
] = """Please explain the meaning and purpose of the following mathematical formula:

LaTeX formula: {latex}
Formula caption: {equation_caption}

Please briefly explain the mathematical meaning, application scenarios, and importance of this formula."""

PROMPTS["QUERY_EQUATION_ANALYST_SYSTEM"] = (
    "You are a mathematics expert who can clearly explain mathematical formulas."
)

PROMPTS[
    "QUERY_GENERIC_ANALYSIS"
] = """Please analyze the following {content_type} type content and extract its main information and key features:

Content: {content_str}

Please briefly summarize the main characteristics and important information of this content."""

PROMPTS["QUERY_GENERIC_ANALYST_SYSTEM"] = (
    "You are a professional content analyst who can accurately analyze {content_type} type content."
)

PROMPTS["QUERY_ENHANCEMENT_SUFFIX"] = (
    "\n\nPlease provide a comprehensive answer based on the user query and the provided multimodal content information."
)
