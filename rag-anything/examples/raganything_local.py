#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAG-Anything Local Multimodal Pipeline (Final Optimized Version)
----------------------------------------------------------------
Key Features:
1. Native Rerank Integration via lightrag_kwargs (No monkey patching).
2. Prompt Reordering (System instructions + context + aligned images).
3. optimized Retrieval Parameters (chunk_top_k=30, top_k=15).
"""

import asyncio
import argparse
import os

from raganything.services.local_rag import LocalRagService, LocalRagSettings

# ==========================================
# 1. 配置与初始化
# ==========================================

async def process_with_rag(service: LocalRagService, file_path: str, doc_id: str):
    # 先入库，再逐题查询。
    final_doc_id = await service.ingest(file_path, doc_id=doc_id)

    queries = [
        "According to the paper, what are the specific visual encoder and language model (LLM) backbones used in the PaddleOCR-VL-1.5 architecture, and why was the 0.9B parameter size chosen?",
        "Based on the evaluation tables, how does PaddleOCR-VL-1.5 compare with GPT-4o and Qwen2-VL-7B on the OmniDocBench benchmark? Please specify the scores for text recognition and layout analysis.",
        "Does PaddleOCR-VL-1.5 use a dynamic resolution strategy (like 'AnyRes') or a fixed-grid approach for high-resolution document images? Describe how it handles images with extreme aspect ratios.",
        "The paper mentions a multi-stage training process. Can you list and describe the three specific training phases (e.g., Pre-training, SFT) and the primary objective of each phase?",
        "Look at the model's performance on the Wild Table Warehouse (WTW) dataset. What specific metrics are used to measure the cell-level structure recognition, and what score did PaddleOCR-VL-1.5 achieve?",
        "Based on the visual samples in the figures, what specific types of 'in-the-wild' distortions (e.g., curved text, reflection, rotation) is the model trained to handle, and what techniques are used to improve its robustness?",
        "The paper introduces a specialized capability for 'Seal Recognition'. What are the unique challenges of parsing seals in documents, and how does the model represent the output of a detected seal (e.g., text content or coordinates)?",
        "What is the total number of document images used in the SFT stage, and what percentage of this data is synthetic versus real-world collected data?",
        "In the context of robust document parsing, how does the paper differentiate the performance and efficiency of PaddleOCR-VL-1.5 compared to the MinerU system?",
        "Does the model support LaTeX output for complex mathematical formulas? If so, what dataset was utilized to fine-tune its performance on formula recognition?",
    ]

    for i, query in enumerate(queries, 1):
        service.logger.info(f"\n{'='*80}")
        service.logger.info(f"Query {i}/{len(queries)}: {query}")
        service.logger.info(f"{'='*80}")

        query_param = {
            "mode": "hybrid",
            "enable_rerank": True,
            "vlm_enhanced": True,
        }

        result = await service.query(final_doc_id, query, **query_param)
        service.logger.info(f"\n✅ Answer:\n{result}\n")

        # if "[" in result and "]" in result:
        #      service.logger.info("✓ Reference detected")
        # else:
        #      service.logger.warning("⚠ No reference found")


# ==========================================
# 4. 入口
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="RAGAnything Local Pipeline")

    parser.add_argument("--path", "-p", required=True, help="要入库的文件或文件夹路径")
    parser.add_argument("--id", "-i", required=True, help="工作空间名称 (doc_id)")

    args = parser.parse_args()
    
    if not os.path.exists(args.path):
        print(f"❌ Input not found: {args.path}")
        return

    settings = LocalRagSettings.from_env()
    service = LocalRagService(settings)

    asyncio.run(process_with_rag(service, args.path, args.id))

if __name__ == "__main__":
    main()