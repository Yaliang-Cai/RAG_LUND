#!/usr/bin/env python
"""
RAG-Anything Local Multimodal Pipeline
--------------------------------------
This script demonstrates a local deployment of a Multimodal RAG system using:
1. RAG-Anything & LightRAG for document processing and retrieval.
2. vLLM (hosting Qwen2-VL-7B) for high-performance inference.
3. Local BGE-M3 for embedding generation.

Key Features:
- Handles mixed inputs (PDFs with tables, figures, equations).
- Custom "Universal Receiver" logic to handle both text-only and multimodal queries.
- Robust context injection to prevent instruction loss.
- Intelligent image quota management to respect vLLM limits.

Usage:
    python raganything_local.py --input ./data/Attention.pdf
"""

import os
import sys
import ssl
import logging
import logging.config
import asyncio
import argparse
import numpy as np
from datetime import datetime

# ==========================================
# 1. Environment & Path Configuration
# ==========================================

# Path to Tiktoken cache to prevent re-downloading
os.environ["TIKTOKEN_CACHE_DIR"] = "/data/y50056788/Yaliang/projects/lightrag/tiktoken_cache"

# Model paths and API configuration
# Ensure these paths match your local file system
EMBEDDING_MODEL_PATH = "/data/h50056787/models/bge-m3"
VLLM_API_BASE = "http://localhost:8001/v1"  # vLLM API endpoint
VLLM_API_KEY = "EMPTY"                      # vLLM typically requires no API key locally
LLM_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

# ==========================================
# 2. Core Library Imports
# ==========================================
from raganything import RAGAnything, RAGAnythingConfig
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
from lightrag.utils import EmbeddingFunc, set_verbose_debug

# Global initialization of the Embedding Model
# Loaded once to avoid overhead during function calls
print(f"Loading Embedding Model: {EMBEDDING_MODEL_PATH} ...")
try:
    # device='cuda:0' ensures embeddings don't conflict with vLLM if on multi-GPU
    st_model = SentenceTransformer(EMBEDDING_MODEL_PATH, trust_remote_code=True, device="cuda:0")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    sys.exit(1)

# Initialize Async OpenAI Client for vLLM
client = AsyncOpenAI(api_key=VLLM_API_KEY, base_url=VLLM_API_BASE)

# ==========================================
# 3. Logging Configuration
# ==========================================
def configure_logging():
    """
    Configure logging to both console (stdout) and a rotating file.
    """
    # Define log directory
    log_dir = "/data/y50056788/Yaliang/projects/lightrag/output_local/logs"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(log_dir, f"run_{timestamp}.log")

    print(f"\nRAGAnything log file: {log_file_path}\n")
    os.makedirs(log_dir, exist_ok=True)

    log_max_bytes = 10485760  # 10MB
    log_backup_count = 5

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(levelname)s: %(message)s",
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "console": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout", 
            },
            "file": {
                "formatter": "detailed",
                "class": "logging.handlers.RotatingFileHandler",
                "filename": log_file_path,
                "maxBytes": log_max_bytes,
                "backupCount": log_backup_count,
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "": { # Root logger
                "handlers": ["console", "file"],
                "level": "INFO",
            },
        },
    })
    
    # Set global logger
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

# ==========================================
# 4. RAG Processing Logic (Core Implementation)
# ==========================================
async def process_with_rag(
    file_path: str,
    output_dir: str,
    working_dir: str = None,
):
    """
    Main asynchronous function to process documents and execute queries using RAG.
    """
    try:
        # Closure variable to track the current question across callbacks
        # This is essential for injecting the user query into the VLM context
        query_tracker = {"current_question": ""} 

        # Initialize RAG configuration
        config = RAGAnythingConfig(
            working_dir=working_dir or "./rag_workspace",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        # ---------------------------------------------------------
        # A. Define LLM Function (Text-only fallback / Extraction)
        # ---------------------------------------------------------
        async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            # Clean arguments to prevent conflicts with OpenAI API
            cleaned_kwargs = {k: v for k, v in kwargs.items() if k not in ['hashing_kv', 'keyword_extraction', 'enable_cot']}
            
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
                    max_tokens=4096,       # Ample space for entity extraction
                    frequency_penalty=1.0, # Prevent repetition loops
                    **cleaned_kwargs
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"LLM Error: {e}")
                return ""

        # ---------------------------------------------------------
        # B. Define Vision Function (The "Universal Receiver")
        # ---------------------------------------------------------
        async def vision_model_func(
            prompt, 
            system_prompt=None, 
            history_messages=[], 
            image_data=None, 
            messages=None,  # Explicitly receive 'messages' from RAG-Anything
            **kwargs
        ):
            """
            Handles multimodal inputs. It intercepts the pre-built 'messages' from RAG,
            injects the system override, manages image quotas, and ensures the user
            query is present.
            """
            # Clean kwargs: CRITICAL to remove 'messages' to avoid "multiple values" error
            exclude_keys = ['hashing_kv', 'keyword_extraction', 'messages', 'enable_cot']
            cleaned_kwargs = {k: v for k, v in kwargs.items() if k not in exclude_keys}
            
            real_question = query_tracker["current_question"]
            
            # System Prompt Override: Forces the model to acknowledge visual capabilities
            SYSTEM_OVERRIDE = {
                "role": "system", 
                "content": "You are Qwen2-VL. Look at the images provided. Extract precise values from tables. Do NOT ignore images."
            }
            
            final_messages = []
            
            # --- Strategy A: RAG provided pre-built messages (Standard Multimodal Flow) ---
            if messages:
                print(f"\n[DEBUG] Received RAG-built messages (Multimodal)...")
                final_messages.append(SYSTEM_OVERRIDE)
                
                found_user_msg = False
                for msg in messages:
                    if msg['role'] == 'user':
                        found_user_msg = True
                        content = msg['content']
                        
                        # Handle Mixed Content (Images + Text)
                        if isinstance(content, list):
                            new_content = []
                            has_image = False
                            for item in content:
                                new_content.append(item) # Copy existing items
                                if item.get('type') == 'image_url':
                                    has_image = True
                            
                            if has_image:
                                print("[DEBUG] Images detected. Appending user instruction...")
                            
                            # Append the actual user query at the end
                            new_content.append({
                                "type": "text", 
                                "text": f"\n\n--- USER INSTRUCTION ---\n{real_question}"
                            })
                            final_messages.append({"role": "user", "content": new_content})
                        
                        # Handle Pure Text within User Message
                        else:
                            new_text = f"{content}\n\n--- USER INSTRUCTION ---\n{real_question}"
                            final_messages.append({"role": "user", "content": new_text})
                    
                    elif msg['role'] == 'system':
                        # Downgrade RAG's internal system prompt to User Context
                        # This prevents it from overriding our visual capabilities
                        final_messages.append({"role": "user", "content": f"Context Info: {msg['content']}"})
                    else:
                        final_messages.append(msg)
                
                # Fallback: If no user message found, append the question
                if not found_user_msg:
                    final_messages.append({"role": "user", "content": real_question})

            # --- Strategy B: Only image_data provided (Manual Construction) ---
            elif image_data:
                print(f"\n[DEBUG] Received raw image_data. Constructing message manually...")
                final_messages.append(SYSTEM_OVERRIDE)
                
                user_content = []
                # Limit images to prevent 400 Errors (ensure this matches vLLM startup args)
                MAX_IMAGES = 10
                imgs = image_data if isinstance(image_data, list) else [image_data]
                
                for img in imgs[:MAX_IMAGES]:
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
                
                # Combine context and instruction
                text_payload = f"Context:\n{system_prompt}\n\nRetrieved:\n{prompt}\n\nInstruction:\n{real_question}"
                user_content.append({"type": "text", "text": text_payload})
                
                final_messages.append({"role": "user", "content": user_content})
            
            # --- Strategy C: Pure Text Fallback ---
            else:
                print(f"\n[DEBUG] No image data. Falling back to text-only mode...")
                # Delegate to the text-only LLM function
                return await llm_model_func(prompt + f"\n{real_question}", system_prompt, history_messages, **kwargs)

            # Send request to vLLM
            try:
                response = await client.chat.completions.create(
                    model=LLM_MODEL_NAME,
                    messages=final_messages, # Use our constructed payload
                    temperature=0.1,
                    max_tokens=2048,
                    **cleaned_kwargs         # Use cleaned kwargs
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Vision LLM Error: {e}")
                return ""

        # ---------------------------------------------------------
        # C. Define Embedding Function
        # ---------------------------------------------------------
        async def _compute_embedding(texts: list[str]) -> np.ndarray:
            # Use local SentenceTransformer
            return st_model.encode(texts, normalize_embeddings=True)

        embedding_func = EmbeddingFunc(
            embedding_dim=1024, # BGE-M3 dimension
            max_token_size=8192,
            func=_compute_embedding
        )

        # ---------------------------------------------------------
        # D. Initialize and Run RAG
        # ---------------------------------------------------------
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        # Process the input (File or Folder)
        if os.path.isfile(file_path):
            logger.info(f"Processing single file: {file_path}")
            await rag.process_document_complete(file_path=file_path, output_dir=output_dir, parse_method="auto")
        elif os.path.isdir(file_path):
            logger.info(f"Processing folder: {file_path}")
            await rag.process_folder_complete(file_path, recursive=False)
        else:
            logger.error(f"Invalid path: {file_path}")
            return

        logger.info("Index built successfully.")

        # ---------------------------------------------------------
        # E. Execute Queries
        # ---------------------------------------------------------
        queries = [
             # "Explain the architecture shown in Figure 1.",
             # "In table 3, what is the number of the params of the base and big model?",
             "In the paper, which models correspond to the parameter counts 65 and 213 (in millions)?",
             # RIGHT

             # "Where do the values 65 x 10^6 and 213 x 10^6 appear in the document, and what do they represent?"  
             # "What do the values 65 x 10^6 and 213 x 10^6 represent in the document?" 
             # "What do the values 65 and 213 represent in the document?"
             # WRONG

        ]

        for query in queries:
            logger.info(f"\n❓ Question: {query}")
            
            # Update the tracker so the vision function knows the current question
            query_tracker["current_question"] = query 
            
            # Increase top_k to ensure relevant tables/images are retrieved
            query_param = {
                "mode": "hybrid",
                "top_k": 15, 
            }
            
            result = await rag.aquery(query, **query_param)
            logger.info(f" Answer:\n{result}")

    except Exception as e:
        logger.error(f"Error processing with RAG: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

# ==========================================
# 5. Main Entry Point
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="RAGAnything Local Multimodal Example")
    parser.add_argument("--input", "-i", default="/data/y50056788/Yaliang/datasets_raw/test.pdf", help="Path to input file or directory")
    parser.add_argument("--working_dir", "-w", default="/data/y50056788/Yaliang/projects/lightrag/output_local/rag_storage", help="Path to RAG storage directory")
    parser.add_argument("--output", "-o", default="/data/y50056788/Yaliang/projects/lightrag/output_local/output", help="Path to intermediate output directory")
    
    args = parser.parse_args()
    
    # Input validation
    if not os.path.exists(args.input):
        # Fallback check for default directory
        if os.path.exists("./data"): 
             pass 
        else:
             print(f"Input not found: {args.input}")
             return

    # Run the async pipeline
    asyncio.run(
        process_with_rag(
            args.input, args.output, args.working_dir
        )
    )

if __name__ == "__main__":
    configure_logging()
    main()