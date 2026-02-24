"""
Prompt templates for InternVL2-26B-AWQ.

Policy:
1) Start from official prompts in raganything.prompt.
2) Keep official content guidance (including "Focus on ...").
3) Add strict JSON output constraints for keys that are parsed as JSON.
"""

from __future__ import annotations

from typing import Any

from raganything.prompt import PROMPTS as _OFFICIAL_PROMPTS


# Copy all official prompts first.
PROMPTS: dict[str, Any] = dict(_OFFICIAL_PROMPTS)


# Strict JSON wrappers for modal analysis prompts.
PROMPTS[
    "vision_prompt"
] = """CRITICAL INSTRUCTION: Your response MUST be a valid JSON object only. Start with {{ and end with }}. No text before or after.

Please analyze this image in detail and provide a JSON response with the following structure:

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

Focus on providing accurate, detailed visual analysis that would be useful for knowledge retrieval.

REMINDER: Output ONLY the JSON object above. Do not add explanations, commentary, or markdown code blocks."""


PROMPTS[
    "vision_prompt_with_context"
] = """CRITICAL INSTRUCTION: Your response MUST be a valid JSON object only. Start with {{ and end with }}. No text before or after.

Please analyze this image in detail, considering the surrounding context. Provide a JSON response with the following structure:

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

Focus on providing accurate, detailed visual analysis that incorporates the context and would be useful for knowledge retrieval.

REMINDER: Output ONLY the JSON object. No additional text."""


PROMPTS[
    "table_prompt"
] = """CRITICAL INSTRUCTION: Your response MUST be a valid JSON object only. Start with {{ and end with }}. No text before or after.

Please analyze this table content and provide a JSON response with the following structure:

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

Focus on extracting meaningful insights and relationships from the tabular data.

REMINDER: Output ONLY the JSON object. No additional text."""


PROMPTS[
    "table_prompt_with_context"
] = """CRITICAL INSTRUCTION: Your response MUST be a valid JSON object only. Start with {{ and end with }}. No text before or after.

Please analyze this table content considering the surrounding context, and provide a JSON response with the following structure:

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

Focus on extracting meaningful insights and relationships from the tabular data in the context of the surrounding content.

REMINDER: Output ONLY the JSON object. No additional text."""


PROMPTS[
    "equation_prompt"
] = """CRITICAL INSTRUCTION: Your response MUST be a valid JSON object only. Start with {{ and end with }}. No text before or after.

Please analyze this mathematical equation and provide a JSON response with the following structure:

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

Focus on providing mathematical insights and explaining the equation's significance.

REMINDER: Output ONLY the JSON object. No additional text."""


PROMPTS[
    "equation_prompt_with_context"
] = """CRITICAL INSTRUCTION: Your response MUST be a valid JSON object only. Start with {{ and end with }}. No text before or after.

Please analyze this mathematical equation considering the surrounding context, and provide a JSON response with the following structure:

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

Focus on providing mathematical insights and explaining the equation's significance within the broader context.

REMINDER: Output ONLY the JSON object. No additional text."""


PROMPTS[
    "generic_prompt"
] = """CRITICAL OUTPUT REQUIREMENTS:
1. Your response MUST be ONLY a valid JSON object
2. Start your response with {{
3. End your response with }}
4. NO text, explanation, or commentary before the JSON
5. NO text, explanation, or commentary after the JSON
6. NO markdown code blocks (do not use ```json)

Please analyze this {content_type} content and provide a JSON response with the following structure:

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

Content: {content}

Focus on extracting meaningful information that would be useful for knowledge retrieval.

FINAL REMINDER: Output ONLY the JSON object shown above. Start with {{ and end with }}. Nothing else."""


PROMPTS[
    "generic_prompt_with_context"
] = """CRITICAL OUTPUT REQUIREMENTS:
1. Your response MUST be ONLY a valid JSON object
2. Start your response with {{
3. End your response with }}
4. NO text, explanation, or commentary before the JSON
5. NO text, explanation, or commentary after the JSON
6. NO markdown code blocks (do not use ```json)

Please analyze this {content_type} content considering the surrounding context, and provide a JSON response with the following structure:

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

Content: {content}

Focus on extracting meaningful information that would be useful for knowledge retrieval and understanding the content's role in the broader context.

FINAL REMINDER: Output ONLY the JSON object shown above. Start with {{ and end with }}. Nothing else."""
