#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAG-Anything Local Multimodal Pipeline (Final Optimized Version)
----------------------------------------------------------------
Key Features:
1. Native Rerank Integration via lightrag_kwargs (No monkey patching).
2. Prompt Reordering (Images -> Text -> Citation Reminder) for Qwen2-VL.
3. optimized Retrieval Parameters (chunk_top_k=30, top_k=15).
"""

import os
import sys
import logging
import logging.config
import asyncio
import argparse
import numpy as np
from datetime import datetime
from functools import partial
import base64
import io
from PIL import Image

# Core Libraries
from raganything import RAGAnything, RAGAnythingConfig
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import AsyncOpenAI
from lightrag.utils import EmbeddingFunc

# ==========================================
# 1. Configuration & Global Init
# ==========================================

# Paths
os.environ["TIKTOKEN_CACHE_DIR"] = "/data/y50056788/Yaliang/projects/lightrag/tiktoken_cache"
EMBEDDING_MODEL_PATH = "/data/h50056787/models/bge-m3"
RERANK_MODEL_PATH = "/data/h50056787/models/bge-reranker-v2-m3"
LOG_DIR = "/data/y50056788/Yaliang/projects/lightrag/output_local/logs"

# API Settings
VLLM_API_BASE = "http://localhost:8001/v1"
VLLM_API_KEY = "EMPTY"
LLM_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

# Global Logger Placeholder
logger = None

# Initialize Models Globally
print(f"Loading Embedding Model: {EMBEDDING_MODEL_PATH} ...")
print(f"Loading Rerank Model: {RERANK_MODEL_PATH} ...")

try:
    st_model = SentenceTransformer(EMBEDDING_MODEL_PATH, trust_remote_code=True, device="cuda:0")
    reranker_model = CrossEncoder(RERANK_MODEL_PATH, device="cuda:0", trust_remote_code=True)
except Exception as e:
    print(f"❌ Critical Error loading models: {e}")
    sys.exit(1)

# Initialize OpenAI Client
client = AsyncOpenAI(api_key=VLLM_API_KEY, base_url=VLLM_API_BASE)


# ==========================================
# 2. Global Helper Functions
# ==========================================

# ==========================================
# 图片压缩配置（根据需要调整）
# ==========================================
IMAGE_MAX_SIZE = (1024, 1024)  # 推荐：1024×1024（Qwen2-VL-7B 平衡点）
IMAGE_QUALITY = 85             # 推荐：85（质量和大小平衡）

# 其他选项：
# - DocBench 评测（激进压缩）: IMAGE_MAX_SIZE = (768, 768), IMAGE_QUALITY = 75
# - 高质量需求（更大图）: IMAGE_MAX_SIZE = (1280, 1280), IMAGE_QUALITY = 90
# - 极限压缩（省 token）: IMAGE_MAX_SIZE = (512, 512), IMAGE_QUALITY = 70

def compress_and_encode_image(image_path: str, max_size: tuple = IMAGE_MAX_SIZE, quality: int = IMAGE_QUALITY) -> str:
    """
    Compress image and encode to base64 (Local override for RAG-Anything)
    
    Args:
        image_path: Path to the image file
        max_size: Maximum size (width, height). Default (1024, 1024) for Qwen2-VL-7B
        quality: JPEG quality (1-100). Default 85 for balance
    
    Returns:
        str: Base64 encoded string, empty string if encoding fails
    """
    try:
        # Open image
        with Image.open(image_path) as img:
            original_size = img.size
            
            # Resize if larger than max_size (keep aspect ratio)
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                print(f"🖼️  Resized image: {original_size} → {img.size}")
            
            # Convert to RGB if needed (for JPEG compatibility)
            if img.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "RGBA":
                    background.paste(img, mask=img.split()[-1])
                else:
                    background.paste(img)
                img = background
            elif img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            
            # Save to bytes with compression
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=quality, optimize=True)
            img_byte_arr.seek(0)
            
            # Encode to base64
            encoded_string = base64.b64encode(img_byte_arr.read()).decode("utf-8")
            
            # Calculate size reduction
            original_size_kb = os.path.getsize(image_path) / 1024
            compressed_size_kb = len(encoded_string) * 3 / 4 / 1024  # Base64 is ~4/3 of binary
            print(f"📦 Compressed: {original_size_kb:.1f}KB → {compressed_size_kb:.1f}KB ({compressed_size_kb/original_size_kb*100:.1f}%)")
            
            return encoded_string
            
    except Exception as e:
        print(f"❌ Image compression error: {e}")
        # Fallback to original encoding
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except:
            return ""


async def global_rerank_func(query: str, documents: list[str], top_n: int) -> list[dict]:
    """
    Global Rerank function using local BGE-Reranker.
    Must be defined globally to be pickle-safe and accessible by LightRAG.
    """
    if not documents:
        return []
    
    # Debug print to verify execution
    print(f"⚡ Reranking {len(documents)} docs for query: '{query[:20]}...'")
    
    try:
        # Construct [Query, Doc] pairs
        pairs = [[query, doc] for doc in documents]
        
        # Inference
        scores = reranker_model.predict(pairs)
        
        # Format results
        results = []
        for i, score in enumerate(scores):
            results.append({
                "index": i,
                "relevance_score": float(score),
            })
        
        # Sort by score descending
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        HARD_LIMIT = 15
        print(f"✂️  Hijack: Requested {top_n}, but forcing cut-off at {HARD_LIMIT} best chunks.")
        return results[:HARD_LIMIT]
        
    except Exception as e:
        print(f"❌ Rerank Error: {e}")
        return []
        
    #     return results[:top_n]
    # except Exception as e:
    #     print(f"❌ Rerank Error: {e}")
    #     return []


def configure_logging():
    """Setup logging configuration."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(LOG_DIR, f"run_{timestamp}.log")
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"\nRAGAnything log file: {log_file_path}\n")

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": "%(levelname)s: %(message)s"},
            "detailed": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "detailed",
                "filename": log_file_path,
                "maxBytes": 10 * 1024 * 1024, # 10MB
                "backupCount": 5,
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "": {"handlers": ["console", "file"], "level": "INFO"},
        },
    })
    
    global logger
    logger = logging.getLogger(__name__)


# ==========================================
# 3. RAG Logic
# ==========================================

async def process_with_rag(file_path: str, output_dir: str, working_dir: str = None):
    """Main pipeline execution."""
    try:
        # ---------------------------------------------------------
        # A. Define Functions (Embedding, LLM, VLM)
        # ---------------------------------------------------------
        
        # 1. Embedding Function
        async def _compute_embedding(texts: list[str]) -> np.ndarray:
            return st_model.encode(texts, normalize_embeddings=True)

        embedding_func = EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=_compute_embedding
        )

        # 2. Text LLM Function
        async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            cleaned_kwargs = {k: v for k, v in kwargs.items() 
                            if k not in ['hashing_kv', 'keyword_extraction', 'enable_cot']}
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history_messages:
                messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})

            try:
                response = await client.chat.completions.create(
                    model=LLM_MODEL_NAME,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4096,
                    **cleaned_kwargs
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"LLM Error: {e}")
                return ""

        # 3. Vision LLM Function (Prompt Reordering Strategy)
        async def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs):
            cleaned_kwargs = {k: v for k, v in kwargs.items() 
                            if k not in ['hashing_kv', 'keyword_extraction', 'enable_cot']}
            
            # Handling RAG-built messages
            if messages:
                logger.debug("Processing RAG-built multimodal messages (Reordering Strategy)")
                
                # Enhanced System Prompt
                # base_system = (
                #     "You are Qwen2-VL, an expert multimodal AI assistant. "
                #     "1. First, analyze ALL images provided at the start. "
                #     "2. Then, read the retrieved text context carefully. "
                #     "3. Answer the user's question based ONLY on the provided images and text. "
                #     "4. CRITICAL: Cite your sources using the [Source ID] format found in the text. "
                #     "5. If the answer is not in the context/images, say 'I don't know'."
                #     "6. SPECIFICALLY FOR TABLES: If the table is not provided, DO NOT guess numbers."
                # )
                # base_system = (

                #     "You are Qwen2-VL, an expert multimodal AI assistant. "

                #     "1. First, analyze ALL images provided at the start. "

                #     "2. Then, read the retrieved text context carefully. "

                #     "3. Answer the user's question based ONLY on the provided images and text. "

                #     "4. CRITICAL: Cite your sources using the [Source ID] format found in the text. "

                #     "5. If the requested item (e.g., Equation N, Table N, Figure N) is not present in the provided context/images, say 'I don't know' and do NOT substitute a different item. "

                #     "6. If the answer is not in the context/images, say 'I don't know'. "

                #     "7. SPECIFICALLY FOR TABLES OR NUMBERS: If the table/value is not explicitly provided, DO NOT guess any numbers. "

                #     "8. Never invent identifiers, equation numbers, or table numbers. "

                #     "9. If you answer with a number, it must appear verbatim in the provided context/images and be cited."

                # )

#                 # Enhanced System Prompt (Based on LightRAG rag_response structure)
#                 base_system = """---Role---

# You are Qwen2-VL, an expert multimodal AI assistant specializing in synthesizing information from both visual and textual content. Your primary function is to answer user queries accurately by ONLY using the information within the provided images and text context.

# ---Goal---

# Generate a comprehensive, well-structured answer to the user query by integrating relevant information from:
# 1. **Images** (provided at the beginning for priority analysis)
# 2. **Knowledge Graph Data** and **Document Chunks** (in the text context)
# 3. **Reference Document List** (for citation purposes)

# ---Critical Instructions for Qwen2-VL---

# 1. **Multimodal Analysis Order (CRITICAL for Qwen2-VL)**:
#    - **FIRST**: Analyze ALL images thoroughly. Images are placed at the beginning specifically for Qwen2-VL's attention mechanism.
#    - **SECOND**: Read the retrieved text context (Knowledge Graph + Document Chunks) carefully.
#    - **THIRD**: Synthesize information from both visual and textual sources to answer the query.

# 2. **Content Grounding & Source Verification**:
#    - Answer ONLY based on the provided images and text context.
#    - DO NOT invent, assume, infer, or use any external knowledge not explicitly present in the context.
#    - If the answer cannot be found in the provided context/images, explicitly state: "I don't know" or "The information is not available in the provided context."
   
# 3. **Strict Entity & Identifier Constraints**:
#    - If a requested specific item (e.g., "Equation 3", "Table 5", "Figure 2") is NOT present in the context/images, say "I don't know".
#    - DO NOT substitute with a different item (e.g., do not answer about "Table 4" when asked about "Table 5").
#    - Never invent equation numbers, table numbers, figure numbers, or any identifiers.

# 4. **Numerical Data Accuracy**:
#    - DO NOT guess, estimate, or approximate any numbers or numerical values.
#    - If a table or specific numerical value is not explicitly provided, state that it is not available.
#    - Any number in your answer MUST appear verbatim in the provided context/images.
#    - All numerical claims MUST be cited with their source.

# 5. **Citation & Reference Requirements**:
#    - Track the reference_id (or [Source ID]) of document chunks that support your answer.
#    - Cite sources using the format **[doc_id]** or **[Source ID]** when using information from the context.
#    - Correlate reference_id with entries in the `Reference Document List` provided in the context.
#    - Generate a **### References** section at the end of your response listing the sources used (see format below).

# 6. **Response Formatting**:
#    - Answer concisely and directly.
#    - Use Markdown formatting when it enhances clarity (headings, bold text, bullet points).
#    - The response should be in the same language as the user query.

# 7. **References Section Format**:
#    ```
#    ### References
   
#    - [1] Document Title One
#    - [2] Document Title Two
#    - [3] Document Title Three
#    ```
#    - Each reference must directly support facts in your answer.
#    - Do NOT include footnotes or additional comments after the References section.

# ---Context Structure---

# The context you receive contains:
# - **Images**: Provided at the start of the message (base64 encoded, already processed by Qwen2-VL)
# - **Text Context**: Following the images, structured as:
#   - Knowledge Graph Data (Entities & Relations)
#   - Document Chunks (with reference_id)
#   - Reference Document List (maps reference_id to document titles)
  
# ---Important Reminders---

# - Prioritize visual information when answering questions about images, charts, diagrams, or figures.
# - Cross-reference visual and textual information for comprehensive answers.
# - When uncertain, err on the side of caution and state uncertainty rather than guessing.
# - Maintain objectivity and avoid introducing personal interpretations beyond what is explicitly shown/stated.
# """

                
                
                # Enhanced System Prompt (Based on LightRAG rag_response structure) For DocBench
                base_system = """---Role---

You are Qwen2-VL, an expert multimodal AI assistant specializing in synthesizing information from both visual and textual content. Your primary function is to answer user queries accurately by ONLY using the information within the provided images and text context.

---Goal---

Generate a comprehensive, well-structured answer to the user query by integrating relevant information from:
1. **Images** (provided at the beginning for priority analysis)
2. **Knowledge Graph Data** and **Document Chunks** (in the text context)
3. **Reference Document List** (for citation purposes)

---Critical Instructions for Qwen2-VL---

1. **Multimodal Analysis Order (CRITICAL for Qwen2-VL)**:
   - **FIRST**: Analyze ALL images thoroughly. Images are placed at the beginning specifically for Qwen2-VL's attention mechanism.
   - **SECOND**: Read the retrieved text context (Knowledge Graph + Document Chunks) carefully.
   - **THIRD**: Synthesize information from both visual and textual sources to answer the query.

2. **Content Grounding & Source Verification**:
   - Answer ONLY based on the provided images and text context.
   - DO NOT invent, assume, infer, or use any external knowledge not explicitly present in the context.
   - If the answer cannot be found in the provided context/images, explicitly state: "I don't know" or "The information is not available in the provided context."
   
3. **Strict Entity & Identifier Constraints**:
   - If a requested specific item (e.g., "Equation 3", "Table 5", "Figure 2") is NOT present in the context/images, say "I don't know".
   - DO NOT substitute with a different item (e.g., do not answer about "Table 4" when asked about "Table 5").
   - Never invent equation numbers, table numbers, figure numbers, or any identifiers.

4. **Numerical Data Accuracy**:
   - DO NOT guess, estimate, or approximate any numbers or numerical values.
   - If a table or specific numerical value is not explicitly provided, state that it is not available.
   - Any number in your answer MUST appear verbatim in the provided context/images.

5. **Response Formatting**:
   - Answer concisely and directly to the question.
   - Provide ONLY the answer without additional explanations, citations, or references unless specifically requested.
   - Use simple, clear language.
   - The response should be in the same language as the user query.
   - Do NOT include a References section or citations at the end.

---Context Structure---

The context you receive contains:
- **Images**: Provided at the start of the message (base64 encoded, already processed by Qwen2-VL)
- **Text Context**: Following the images, structured as:
  - Knowledge Graph Data (Entities & Relations)
  - Document Chunks (with reference_id)
  - Reference Document List (maps reference_id to document titles)
  
---Important Reminders---

- Prioritize visual information when answering questions about images, charts, diagrams, or figures.
- Cross-reference visual and textual information for comprehensive answers.
- When uncertain, err on the side of caution and state uncertainty rather than guessing.
- Maintain objectivity and avoid introducing personal interpretations beyond what is explicitly shown/stated.
"""
                original_sys = next((m['content'] for m in messages if m['role'] == 'system'), "")
                full_system = f"{base_system}\n\n---Additional Instructions---\n\n{original_sys}"

                # Extract content
                user_content_list = []
                for msg in messages:
                    if msg['role'] == 'user':
                        if isinstance(msg['content'], list):
                            user_content_list.extend(msg['content'])
                        else:
                            user_content_list.append({"type": "text", "text": str(msg['content'])})

                # Separate Images and Text
                images_part = [item for item in user_content_list if item.get('type') == 'image_url']
                texts_part = [item.get('text', '').strip() for item in user_content_list if item.get('type') == 'text']
                full_text_context = "\n\n".join([t for t in texts_part if t])

                # Append Citation Reminder
                citation_reminder = (
                    "\n\n----------------\n"
                    "FINAL INSTRUCTION:\n"
                    "You MUST cite your sources using the format [doc_id] or [Source ID] "
                    "at the end of every sentence where you use information from the context. "
                )
                final_text_payload = f"--- RETRIEVED CONTEXT & QUESTION ---\n{full_text_context}{citation_reminder}"

                # Reconstruct: Images FIRST -> Text LAST
                final_user_content = []
                final_user_content.extend(images_part)
                final_user_content.append({"type": "text", "text": final_text_payload})

                final_messages = [
                    {"role": "system", "content": full_system},
                    {"role": "user", "content": final_user_content}
                ]

                try:
                    response = await client.chat.completions.create(
                        model=LLM_MODEL_NAME, messages=final_messages,
                        temperature=0.1, max_tokens=2048, **cleaned_kwargs
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    logger.error(f"Vision LLM Error: {e}")
                    # Retry logic for token overflow could go here
                    raise
            
            # Fallback for text-only or raw image_data
            elif image_data:
                 # Simplified fallback (omitted for brevity, same as original)
                 pass 
            else:
                return await llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        # ---------------------------------------------------------
        # B. Initialize RAG (The Correct Way)
        # ---------------------------------------------------------
        
        # Initialize Config with lightrag_kwargs
        config = RAGAnythingConfig(
            working_dir=working_dir or "./rag_workspace",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
            
            
        )

        logger.info("Initializing RAG-Anything...")
        
        # Monkey patch: Replace RAG-Anything's image encoding with our compressed version
        import raganything.utils
        import raganything.modalprocessors
        raganything.utils.encode_image_to_base64 = compress_and_encode_image
        # Also patch the modalprocessors module's encode if it uses a different import
        logger.info("✅ Patched image encoding with compression (1024×1024, quality=85)")
        
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
            lightrag_kwargs={
                "rerank_model_func": global_rerank_func,
            }
        )

        # ---------------------------------------------------------
        # C. Process Documents
        # ---------------------------------------------------------
        if os.path.isfile(file_path):
            logger.info(f"Processing single file: {file_path}")
            await rag.process_document_complete(
                file_path=file_path, 
                output_dir=output_dir, 
                parse_method="auto"
            )
        elif os.path.isdir(file_path):
            logger.info(f"Processing folder: {file_path}")
            await rag.process_folder_complete(file_path, recursive=False)
        else:
            logger.error(f"Invalid path: {file_path}")
            return

        logger.info("Index built successfully.")

        # ---------------------------------------------------------
        # D. Execute Queries
        # ---------------------------------------------------------
        queries = [
            "Explain the architecture shown in Figure 1.",
            "In table 3, what is the number of the params of the base and big model?",
            "What is the name of the chapter that mentions the table 1 in the rag-anything paper?",
            "Give me the latex format expression of the equation (1) in the rag-anything paper.",
            "Explain the equation 1",
        ]

        for i, query in enumerate(queries, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Query {i}/{len(queries)}: {query}")
            logger.info(f"{'='*80}")

            # 优化的查询参数
            query_param = {
                "mode": "hybrid", 
                "top_k": 60,           # 给 VLM 15 个精华片段
                # "chunk_top_k": 60,     # 给 Rerank 30 个初筛片段 (扩大漏斗口)
                "enable_rerank": True, # 开启 Rerank (现在一定生效)
                "vlm_enhanced": True,
            }
            
            try:
                result = await rag.aquery(query, **query_param)
                logger.info(f"\n✅ Answer:\n{result}\n")
                
                if '[' in result and ']' in result:
                    logger.info("✓ Reference detected")
                else:
                    logger.warning("⚠ No reference found")
            
            except Exception as e:
                logger.error(f"❌ Query failed: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())

    except Exception as e:
        logger.error(f"Error processing with RAG: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


# ==========================================
# 4. Entry Point
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="RAGAnything Local Pipeline")
    parser.add_argument("--input", "-i", default="/data/y50056788/Yaliang/datasets_raw/test.pdf", help="Input file/dir")
    parser.add_argument("--working_dir", "-w", default="/data/y50056788/Yaliang/projects/lightrag/output_local/rag_storage", help="Storage dir")
    parser.add_argument("--output", "-o", default="/data/y50056788/Yaliang/projects/lightrag/output_local/output", help="Output dir")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Input not found: {args.input}")
        return

    asyncio.run(process_with_rag(args.input, args.output, args.working_dir))

if __name__ == "__main__":
    configure_logging()
    main()