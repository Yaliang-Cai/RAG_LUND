#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DocBench Evaluation Script for RAG-Anything Local (Manual Server Mode)
======================================================================

使用流程：
---------
1. 手动启动 InternVL2-26B-AWQ 服务（端口 8001）
   cd /data/y50056788/Yaliang/projects/lightrag
   bash start_server_internvl2.sh

2. 生成系统答案（可后台运行）
   python evaluate.py --mode generate
   或：nohup python evaluate.py --mode generate > run_generate.log 2>&1 &

3. 停止 InternVL2-26B-AWQ，启动 Qwen2.5-32B（端口 8002）
   # 在server终端按 Ctrl+C
   bash start_server_qwen2.5_32b_awq.sh

4. 评估答案
   python evaluate.py --mode evaluate

5. 查看统计
   python evaluate.py --mode stats
"""

import os

# ===== 限制 MinerU 内部 vLLM 的 GPU 内存占用 =====
# 0.10 = 4.74GB, 0.15 = 7.1GB, 0.20 = 9.47GB
# 推荐 0.15：给 MinerU 7.1GB，GPU 0 总占用 ~12GB
os.environ['MINERU_VLLM_GPU_MEMORY_UTILIZATION'] = '0.15'
# =================================================

import json
import asyncio
import logging
import re
import sys
from pathlib import Path
from typing import Any
from openai import AsyncOpenAI

# Add parent directory to path to import local_rag
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from raganything.services.local_rag import LocalRagService, LocalRagSettings

# ==========================================
# 配置区（绝对路径，方便在任意目录运行）
# ==========================================

# 脚本所在目录
SCRIPT_DIR = Path("/data/y50056788/Yaliang/projects/rag-anything/evaluate_local/DocBench")

# DocBench 数据集目录
DATA_ROOT = Path("/data/y50056788/Yaliang/datasets_for_eval/data_for_DocBench")

# 输出根目录（所有结果保存在这里）
OUTPUT_DIR = SCRIPT_DIR / "docbench_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPT_DUMP_DIR = OUTPUT_DIR / "prompt_dumps"
PROMPT_DUMP_DIR.mkdir(parents=True, exist_ok=True)
FINAL_MESSAGES_DUMP_DIR = OUTPUT_DIR / "final_vlm_messages"
FINAL_MESSAGES_DUMP_DIR.mkdir(parents=True, exist_ok=True)

# RAG 工作目录（每个文档一个独立图谱）
# 输出：docbench_results/rag_workspaces/docbench_0/, docbench_1/, ...
WORKING_DIR_ROOT = OUTPUT_DIR / "rag_workspaces"
WORKING_DIR_ROOT.mkdir(parents=True, exist_ok=True)

# MinerU 输出目录（每个文档独立）
# 输出：docbench_results/mineru_outputs/docbench_0/{pdf_name}/hybrid_auto/, ...
OUTPUT_MD_DIR = OUTPUT_DIR / "mineru_outputs"
OUTPUT_MD_DIR.mkdir(parents=True, exist_ok=True)

# API 配置
RAG_API_BASE = "http://localhost:8001/v1"      # InternVL2-26B-AWQ（生成答案）
JUDGE_API_BASE = "http://localhost:8002/v1"    # Qwen2.5-32B（评估）
RAG_API_KEY = "EMPTY"

RAG_MODEL_NAME = "OpenGVLab/InternVL2-26B-AWQ"
JUDGE_MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"

# Query 参数（DocBench 优化）
DOCBENCH_QUERY_PARAMS = {
    "mode": "hybrid",
    "top_k": 30,
    "chunk_top_k": 50,
}

# ==========================================
# 日志配置
# ==========================================

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),  # 确保输出到终端
    ]
)
logger = logging.getLogger(__name__)

# 同时配置其他模块的 logger，确保能看到所有日志
logging.getLogger("raganything").setLevel(logging.INFO)
logging.getLogger("raganything.processor").setLevel(logging.INFO)
logging.getLogger("raganything.parser").setLevel(logging.INFO)


def _extract_reference_lines(raw_prompt: str) -> list[str]:
    """
    Extract lines under "Reference Document List" from raw prompt.
    """
    marker = (
        "Reference Document List (Each entry starts with a [reference_id] "
        "that corresponds to entries in the Document Chunks):"
    )
    marker_pos = raw_prompt.rfind(marker)
    if marker_pos < 0:
        return []

    # Parse only the fenced block after the final Reference Document List marker.
    tail = raw_prompt[marker_pos + len(marker) :]
    block_match = re.search(r"```(?:[a-zA-Z0-9_+-]+)?\s*\n(.*?)\n```", tail, re.DOTALL)
    if not block_match:
        return []

    return [line.strip() for line in block_match.group(1).splitlines() if line.strip()]


def _dump_raw_prompt(
    doc_name: str,
    qa_idx: int,
    question: str,
    raw_prompt: str,
) -> Path:
    """
    Dump raw prompt for debugging reference construction.
    """
    doc_dump_dir = PROMPT_DUMP_DIR / f"docbench_{doc_name}"
    doc_dump_dir.mkdir(parents=True, exist_ok=True)
    out_file = doc_dump_dir / f"q{qa_idx + 1:03d}_raw_prompt.txt"

    ref_lines = _extract_reference_lines(raw_prompt)
    header_lines = [
        f"# doc_id: {doc_name}",
        f"# question_index: {qa_idx + 1}",
        f"# question: {question}",
        f"# reference_line_count: {len(ref_lines)}",
        "# reference_lines:",
    ]
    if ref_lines:
        header_lines.extend([f"#   {line}" for line in ref_lines])
    else:
        header_lines.append("#   <not found>")
    header_lines.extend(["", "===== RAW PROMPT START =====", ""])

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines))
        f.write(raw_prompt)

    return out_file


def _extract_reference_lines_from_messages(messages_payload: list[dict[str, Any]]) -> list[str]:
    """
    Extract reference lines from the final chat.completions messages payload.
    """
    user_text_parts: list[str] = []
    for msg in messages_payload:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if text:
                        user_text_parts.append(str(text))
        elif isinstance(content, str):
            user_text_parts.append(content)

    if not user_text_parts:
        return []

    return _extract_reference_lines("\n".join(user_text_parts))


def _dump_final_messages(
    doc_name: str,
    qa_idx: int,
    question: str,
    captured_calls: list[dict[str, Any]],
) -> Path:
    """
    Dump captured chat.completions messages for this query.
    """
    doc_dump_dir = FINAL_MESSAGES_DUMP_DIR / f"docbench_{doc_name}"
    doc_dump_dir.mkdir(parents=True, exist_ok=True)
    out_file = doc_dump_dir / f"q{qa_idx + 1:03d}_final_messages.json"

    selected_messages = (
        captured_calls[-1].get("messages", [])
        if captured_calls
        else []
    )
    ref_lines = _extract_reference_lines_from_messages(selected_messages)
    payload = {
        "doc_id": doc_name,
        "question_index": qa_idx + 1,
        "question": question,
        "capture_count": len(captured_calls),
        "selected_client": captured_calls[-1].get("client") if captured_calls else None,
        "reference_line_count": len(ref_lines),
        "reference_lines": ref_lines,
        "messages": selected_messages,
        "captured_calls": captured_calls,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_file


def _set_doc_workspace(workspace: str) -> None:
    """
    Force per-document LightRAG namespace isolation.
    """
    os.environ["WORKSPACE"] = workspace
    try:
        from lightrag.kg.shared_storage import set_default_workspace

        set_default_workspace(workspace)
    except Exception as exc:
        logger.warning(f"Failed to set LightRAG workspace '{workspace}': {exc}")


def _bridge_lightrag_logs_to_run_file() -> None:
    """
    LightRAG uses its own logger with propagate=False and a console-only handler.
    Attach root file handlers so LightRAG warnings/info are persisted in run_*.log.
    """
    try:
        from lightrag.utils import logger as lightrag_logger
    except Exception as exc:
        logger.warning(f"Failed to import lightrag logger: {exc}")
        return

    root_logger = logging.getLogger()
    file_handlers = [
        h for h in root_logger.handlers if isinstance(h, logging.FileHandler)
    ]
    if not file_handlers:
        logger.warning("No root file handler found. LightRAG logs may be console-only.")
        return

    existing_files = {getattr(h, "baseFilename", None) for h in lightrag_logger.handlers}
    attached = 0
    for handler in file_handlers:
        file_path = getattr(handler, "baseFilename", None)
        if file_path and file_path not in existing_files:
            lightrag_logger.addHandler(handler)
            attached += 1

    lightrag_logger.setLevel(logging.INFO)
    logger.info(f"LightRAG log bridge ready (attached file handlers: {attached}).")


# ==========================================
# Step 1: Generate Answers
# ==========================================

async def generate_answers(
    start_id: int = 0,
    end_id: int = 229,
    resume: bool = True,
    dump_raw_prompt: bool = False,
    dump_final_messages: bool = False,
):
    """
    为 DocBench 生成系统答案
    
    前提：InternVL2-26B-AWQ 服务已在 8001 端口运行
    
    Args:
        start_id: 起始文档 ID
        end_id: 结束文档 ID（不包含）
        resume: 是否跳过已处理的文档
        dump_raw_prompt: 是否落盘每题 raw_prompt（用于调试引用列表）
        dump_final_messages: 是否落盘最终发送给 VLM 的 messages（用于检查引用列表）
    """
    output_file = OUTPUT_DIR / "system_answers.jsonl"
    
    # 显式配置路径，保证评测时每个文档目录和日志目录可控
    settings = LocalRagSettings.from_env()
    settings.working_dir_root = str(WORKING_DIR_ROOT)
    settings.output_dir = str(OUTPUT_MD_DIR)
    settings.log_dir = str(SCRIPT_DIR / "logs")
    settings.vllm_api_base = settings.vision_vllm_api_base = RAG_API_BASE
    settings.vllm_api_key = settings.vision_vllm_api_key = RAG_API_KEY
    
    # 由于 GPU 1 不可访问，所有模型都在 GPU 0
    # GPU 0 分配：
    # - 其他进程: ~13 GiB
    # - MinerU: ~2.4 GiB (0.05 * 47.35)
    # - Embedding/Rerank: ~5 GiB
    # - 总计: ~20 GiB (应该够用，虽然紧张)
    settings.device = "cuda:0"
    settings.llm_model_name = settings.vision_model_name = RAG_MODEL_NAME
    settings.temperature = 0.0
    settings.query_max_tokens = 2048
    settings.ingest_max_tokens = 8192
    settings.vlm_max_images = 10
    settings.vlm_enable_json_schema = True
    
    # 只创建一次 service
    service = LocalRagService(settings)
    _bridge_lightrag_logs_to_run_file()
    logger.info(f"✅ RAG Service initialized")
    
    # 加载已处理的文档（用于断点续传）
    processed_docs = set()
    if resume and output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                processed_docs.add(data['doc_id'])
        logger.info(f"📁 Resume: found {len(processed_docs)} processed documents")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"📝 Generating answers for documents {start_id} to {end_id-1}")
    logger.info(f"📂 Output: {output_file}")
    logger.info(f"{'='*80}\n")
    
    # 遍历每个文档
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for doc_id in range(start_id, end_id):
            doc_name = str(doc_id)
            
            # 跳过已处理
            if resume and doc_name in processed_docs:
                logger.info(f"⏭️  [{doc_id}] Already processed, skipping")
                continue
            
            # 检查文件夹是否存在
            folder_path = DATA_ROOT / doc_name
            if not folder_path.exists():
                logger.warning(f"⚠️  [{doc_id}] Folder not found: {folder_path}")
                continue
            
            # 查找 PDF 和 QA 文件
            pdf_file = None
            qa_file = None
            for file in folder_path.iterdir():
                if file.suffix == '.pdf':
                    pdf_file = file
                elif file.name.endswith('_qa.jsonl'):
                    qa_file = file
            
            if not pdf_file or not qa_file:
                logger.warning(f"⚠️  [{doc_id}] Missing PDF or QA file")
                continue
            
            logger.info(f"\n{'='*80}")
            logger.info(f"📄 [{doc_id}/{end_id-1}] Processing: {pdf_file.name}")
            logger.info(f"{'='*80}")
            
            try:
                # 每个文档独立的 ID
                rag_doc_id = f"docbench_{doc_name}"
                _set_doc_workspace(rag_doc_id)
                logger.info(f"🔧 Processing: {pdf_file.name} → doc_id: {rag_doc_id}")
                
                # MinerU 独立输出目录（避免 PDF 重名冲突）
                doc_output_dir = str(OUTPUT_MD_DIR / f"docbench_{doc_name}")
                
                # Ingest 文档
                logger.info(f"📖 Ingesting document...")
                returned_doc_id = await service.ingest(
                    file_path=str(pdf_file),
                    output_dir=doc_output_dir,  # 每个文档独立 MinerU 输出
                    doc_id=rag_doc_id
                )
                logger.info(f"✅ Ingestion complete, doc_id: {returned_doc_id}")
                
                # 加载问题
                with open(qa_file, 'r', encoding='utf-8') as f_qa:
                    qa_list = [json.loads(line) for line in f_qa]
                
                logger.info(f"\n❓ Answering {len(qa_list)} questions...")
                
                # 回答每个问题
                for qa_idx, qa_item in enumerate(qa_list):
                    question = qa_item['question']
                    logger.info(f"  [{qa_idx+1}/{len(qa_list)}] {question[:60]}...")
                    
                    try:
                        if dump_raw_prompt:
                            try:
                                from lightrag.base import QueryParam

                                rag = await service.get_rag(rag_doc_id)
                                raw_prompt_param = QueryParam(
                                    mode=DOCBENCH_QUERY_PARAMS["mode"],
                                    top_k=DOCBENCH_QUERY_PARAMS["top_k"],
                                    chunk_top_k=DOCBENCH_QUERY_PARAMS["chunk_top_k"],
                                    only_need_prompt=True,
                                )
                                raw_prompt_result = await rag.lightrag.aquery(
                                    question, param=raw_prompt_param
                                )
                                raw_prompt_text = (
                                    raw_prompt_result.content
                                    if hasattr(raw_prompt_result, "content")
                                    else str(raw_prompt_result)
                                )
                                dump_path = _dump_raw_prompt(
                                    doc_name, qa_idx, question, raw_prompt_text
                                )
                                ref_count = len(_extract_reference_lines(raw_prompt_text))
                                logger.info(
                                    f"      🔎 Prompt dumped: {dump_path} (reference lines: {ref_count})"
                                )
                            except Exception as dump_exc:
                                logger.warning(f"      ⚠️ Raw prompt dump failed: {dump_exc}")

                        captured_calls: list[dict[str, Any]] = []
                        patched_completions: list[tuple[Any, Any]] = []
                        if dump_final_messages:
                            def _patch_client(client_name: str, client_obj: Any) -> None:
                                if client_obj is None:
                                    return
                                chat_obj = getattr(client_obj, "chat", None)
                                completions_api = getattr(chat_obj, "completions", None)
                                if completions_api is None:
                                    return
                                for api_obj, _ in patched_completions:
                                    if api_obj is completions_api:
                                        return

                                original_create = completions_api.create

                                async def _capture_create(
                                    *args,
                                    _orig=original_create,
                                    _client=client_name,
                                    **kwargs,
                                ):
                                    messages_payload = kwargs.get("messages")
                                    if isinstance(messages_payload, list):
                                        captured_calls.append(
                                            {
                                                "client": _client,
                                                "messages": messages_payload,
                                            }
                                        )
                                    return await _orig(*args, **kwargs)

                                completions_api.create = _capture_create
                                patched_completions.append((completions_api, original_create))

                            _patch_client("vision", getattr(service, "vision_client", None))
                            _patch_client("text", getattr(service, "text_client", None))

                        # 使用 RAG 查询
                        try:
                            answer = await service.query(
                                doc_id=rag_doc_id,
                                query=question,
                                **DOCBENCH_QUERY_PARAMS
                            )
                        finally:
                            if dump_final_messages:
                                for completions_api, original_create in patched_completions:
                                    completions_api.create = original_create

                        if dump_final_messages:
                            if captured_calls:
                                dump_path = _dump_final_messages(
                                    doc_name, qa_idx, question, captured_calls
                                )
                                final_messages = captured_calls[-1].get("messages", [])
                                ref_count = len(
                                    _extract_reference_lines_from_messages(final_messages)
                                )
                                logger.info(
                                    f"      🧾 Final messages dumped: {dump_path} (calls: {len(captured_calls)}, reference lines: {ref_count})"
                                )
                            else:
                                logger.warning(
                                    "      ⚠️ Final messages not captured for this query."
                                )
                        
                        logger.info(f"      ✓ Answer: {answer[:80]}...")
                        
                        # 保存结果（符合 DocBench 格式）
                        result = {
                            'doc_id': doc_name,                    # 文档ID（用于匹配）
                            'question': question,                   # 问题
                            'sys_ans': answer,                      # 系统答案
                            'ref_ans': qa_item['answer'],          # 参考答案
                            'type': qa_item['type'],                # 问题类型
                            'evidence': qa_item['evidence'],        # 证据文本
                        }
                        f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f_out.flush()
                        
                    except Exception as e:
                        logger.error(f"      ✗ Error: {e}")
                        # 保存空答案
                        result = {
                            'doc_id': doc_name,
                            'question': question,
                            'sys_ans': "",
                            'ref_ans': qa_item['answer'],
                            'type': qa_item['type'],
                            'evidence': qa_item['evidence'],
                        }
                        f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f_out.flush()
                
                logger.info(f"✅ [{doc_id}] Completed: {len(qa_list)} questions answered\n")
                
                # 清理当前文档的 RAG 实例，释放内存
                # 注意：必须 await finalize_storages() 而不是调用 close()
                # 因为 close() 在 async 上下文中会异步调度清理，不会等待完成
                if rag_doc_id in service._rag_instances:
                    try:
                        # 直接调用 finalize_storages() 并等待完成，确保资源真正释放
                        await service._rag_instances[rag_doc_id].finalize_storages()
                        del service._rag_instances[rag_doc_id]
                        logger.info(f"🧹 Cleaned up RAG instance for {rag_doc_id}")
                    except Exception as cleanup_error:
                        logger.warning(f"⚠️  Failed to cleanup RAG instance: {cleanup_error}")
                else:
                    logger.info(f"ℹ️  No RAG instance to clean for {rag_doc_id}")
                
                # 每个文档处理后清理 GPU 缓存，防止内存泄漏累积
                # 记录清理前后的 GPU 内存，验证是否真的释放了
                try:
                    import torch
                    if torch.cuda.is_available():
                        # 记录清理前的内存
                        mem_before = torch.cuda.memory_allocated(0) / 1024**3  # GB
                        reserved_before = torch.cuda.memory_reserved(0) / 1024**3  # GB
                        
                        torch.cuda.empty_cache()
                        
                        # 记录清理后的内存
                        mem_after = torch.cuda.memory_allocated(0) / 1024**3  # GB
                        reserved_after = torch.cuda.memory_reserved(0) / 1024**3  # GB
                        
                        freed = reserved_before - reserved_after
                        logger.info(
                            f"🧹 GPU cache cleared after doc {doc_id}: "
                            f"Allocated {mem_before:.2f}→{mem_after:.2f} GB, "
                            f"Reserved {reserved_before:.2f}→{reserved_after:.2f} GB "
                            f"(freed {freed:.2f} GB)"
                        )
                except Exception as cache_error:
                    logger.warning(f"⚠️  Failed to clear GPU cache: {cache_error}")
                
            except Exception as e:
                import traceback
                logger.error(f"❌ [{doc_id}] Error: {e}")
                logger.error(f"❌ [{doc_id}] Traceback:\n{traceback.format_exc()}")
                # 即使出错也尝试清理
                if 'rag_doc_id' in locals() and rag_doc_id in service._rag_instances:
                    try:
                        await service._rag_instances[rag_doc_id].finalize_storages()
                        del service._rag_instances[rag_doc_id]
                    except:
                        pass
                continue
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ Answer generation complete!")
    logger.info(f"📁 Saved to: {output_file}")
    logger.info(f"{'='*80}")


# ==========================================
# Step 2: Evaluate Answers
# ==========================================

async def evaluate_answers(resume: bool = True):
    """
    使用 Qwen2.5-32B 评估系统答案
    
    前提：Qwen2.5-32B 服务已在 8002 端口运行
    
    Args:
        resume: 是否跳过已评估的答案
    """
    input_file = OUTPUT_DIR / "system_answers.jsonl"
    output_file = OUTPUT_DIR / "eval_results.jsonl"
    
    if not input_file.exists():
        logger.error(f"❌ Input file not found: {input_file}")
        logger.info("💡 Please run: python evaluate.py --mode generate first")
        return
    
    # 加载评估 prompt
    eval_prompt_file = SCRIPT_DIR / "evaluation_prompt.txt"
    with open(eval_prompt_file, 'r', encoding='utf-8') as f:
        eval_prompt = f.read()
    
    # 加载系统答案
    with open(input_file, 'r', encoding='utf-8') as f:
        answers = [json.loads(line) for line in f]
    
    # 加载已评估的答案（用于断点续传）
    evaluated_keys = set()
    if resume and output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                key = f"{data['doc_id']}_{data['question']}"
                evaluated_keys.add(key)
        logger.info(f"📁 Resume: found {len(evaluated_keys)} evaluated answers")
    
    # 初始化 Judge 客户端
    judge_client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=JUDGE_API_BASE
    )
    
    logger.info(f"\n{'='*80}")
    logger.info(f"⚖️  Evaluating {len(answers)} answers using {JUDGE_MODEL_NAME}")
    logger.info(f"{'='*80}\n")
    
    # 评估每个答案
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for i, item in enumerate(answers, 1):
            key = f"{item['doc_id']}_{item['question']}"
            
            # 跳过已评估
            if resume and key in evaluated_keys:
                logger.info(f"⏭️  [{i}/{len(answers)}] Already evaluated, skipping")
                continue
            
            logger.info(f"\n[{i}/{len(answers)}] Doc {item['doc_id']}")
            logger.info(f"  Q: {item['question'][:60]}...")
            logger.info(f"  A: {item['sys_ans'][:60]}...")
            
            # 构建评估 prompt
            cur_prompt = eval_prompt.replace('{{question}}', item['question']) \
                                     .replace('{{sys_ans}}', item['sys_ans']) \
                                     .replace('{{ref_ans}}', item['ref_ans']) \
                                     .replace('{{ref_text}}', item['evidence'])
            
            try:
                # 调用 Judge 模型
                response = await judge_client.chat.completions.create(
                    model=JUDGE_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a helpful evaluator."},
                        {"role": "user", "content": cur_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=50
                )
                
                eval_result = response.choices[0].message.content.strip()
                
                # 解析分数（在前20个字符中查找 0 或 1）
                score = 1 if '1' in eval_result[:20] else 0
                logger.info(f"  ⚖️  Score: {score} | {eval_result[:40]}...")
                
                # 保存评估结果
                result = {
                    **item,
                    'eval': eval_result,
                    'score': score
                }
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_out.flush()
                
            except Exception as e:
                logger.error(f"  ❌ Error: {e}")
                # 保存错误结果（score=0）
                result = {
                    **item,
                    'eval': f"[ERROR: {str(e)}]",
                    'score': 0
                }
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_out.flush()
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ Evaluation complete!")
    logger.info(f"📁 Saved to: {output_file}")
    logger.info(f"{'='*80}")


# ==========================================
# Step 3: Calculate Statistics
# ==========================================

def calculate_statistics():
    """计算评估统计数据"""
    result_file = OUTPUT_DIR / "eval_results.jsonl"
    
    if not result_file.exists():
        logger.error(f"❌ Result file not found: {result_file}")
        logger.info("💡 Please run: python evaluate.py --mode evaluate first")
        return
    
    # 加载评估结果
    with open(result_file, 'r', encoding='utf-8') as f:
        results = [json.loads(line) for line in f]
    
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 DocBench Evaluation Statistics")
    logger.info(f"{'='*80}\n")
    
    # 总体准确率
    total = len(results)
    correct = sum(1 for r in results if r.get('score', 0) == 1)
    overall_acc = correct / total * 100 if total > 0 else 0
    
    logger.info(f"Overall Accuracy: {overall_acc:.2f}% ({correct}/{total})")
    
    # 按问题类型统计
    type_stats = {}
    for r in results:
        qtype = r['type']
        if qtype not in type_stats:
            type_stats[qtype] = {'correct': 0, 'total': 0}
        type_stats[qtype]['total'] += 1
        if r.get('score', 0) == 1:
            type_stats[qtype]['correct'] += 1
    
    logger.info(f"\n📋 Accuracy by Question Type:")
    for qtype in sorted(type_stats.keys()):
        stats = type_stats[qtype]
        acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        logger.info(f"  {qtype:20s}: {acc:5.2f}% ({stats['correct']:3d}/{stats['total']:3d})")
    
    # 按领域统计（参考 DocBench 官方分布）
    domain_ranges = {
        'Academic': range(0, 49),
        'Finance': range(49, 89),
        'Government': range(89, 133),
        'Law': range(133, 179),
        'News': range(179, 229)
    }
    
    domain_stats = {domain: {'correct': 0, 'total': 0} for domain in domain_ranges.keys()}
    for r in results:
        try:
            doc_num = int(r['doc_id'])
            for domain, id_range in domain_ranges.items():
                if doc_num in id_range:
                    domain_stats[domain]['total'] += 1
                    if r.get('score', 0) == 1:
                        domain_stats[domain]['correct'] += 1
                    break
        except:
            continue
    
    logger.info(f"\n🌐 Accuracy by Domain:")
    for domain in ['Academic', 'Finance', 'Government', 'Law', 'News']:
        stats = domain_stats[domain]
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total'] * 100
            logger.info(f"  {domain:15s}: {acc:5.2f}% ({stats['correct']:3d}/{stats['total']:3d})")
    
    # 保存统计到 JSON
    stats_output = {
        'overall': {'accuracy': overall_acc, 'correct': correct, 'total': total},
        'by_type': {
            qtype: {
                'accuracy': stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0,
                'correct': stats['correct'],
                'total': stats['total']
            }
            for qtype, stats in type_stats.items()
        },
        'by_domain': {
            domain: {
                'accuracy': stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0,
                'correct': stats['correct'],
                'total': stats['total']
            }
            for domain, stats in domain_stats.items() if stats['total'] > 0
        }
    }
    
    stats_file = OUTPUT_DIR / "statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n📁 Statistics saved to: {stats_file}")
    logger.info(f"{'='*80}\n")


# ==========================================
# Main Entry Point
# ==========================================

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="DocBench Evaluation Script (Manual Server Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 生成所有文档的答案
  python evaluate.py --mode generate
  
  # 生成部分文档的答案
  python evaluate.py --mode generate --start_id 0 --end_id 10
  
  # 评估所有答案
  python evaluate.py --mode evaluate
  
  # 查看统计
  python evaluate.py --mode stats
  
  # 后台运行（推荐）
  nohup python evaluate.py --mode generate > run.log 2>&1 &
  tail -f run.log
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['generate', 'evaluate', 'stats'],
        help='运行模式'
    )
    parser.add_argument(
        '--start_id',
        type=int,
        default=0,
        help='起始文档 ID'
    )
    parser.add_argument(
        '--end_id',
        type=int,
        default=229,
        help='结束文档 ID（不包含）'
    )
    parser.add_argument(
        '--no_resume',
        action='store_true',
        help='不恢复之前的进度，从头开始'
    )
    parser.add_argument(
        '--dump_raw_prompt',
        action='store_true',
        help='生成模式下落盘每题 raw_prompt（用于检查 Reference Document List）'
    )
    parser.add_argument(
        '--dump_final_messages',
        action='store_true',
        help='生成模式下落盘最终发给 VLM 的 messages（用于检查真实输入）'
    )
    
    args = parser.parse_args()
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🎯 DocBench Evaluation - Manual Server Mode")
    logger.info(f"{'='*80}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Range: {args.start_id} to {args.end_id-1}")
    logger.info(f"Resume: {not args.no_resume}")
    logger.info(f"DumpRawPrompt: {args.dump_raw_prompt}")
    logger.info(f"DumpFinalMessages: {args.dump_final_messages}")
    logger.info(f"{'='*80}\n")
    
    # 执行对应的步骤
    if args.mode == 'generate':
        logger.info("⚠️  Please ensure InternVL2-26B-AWQ is running on port 8001")
        logger.info(f"   Check: curl http://localhost:8001/v1/models\n")
        await generate_answers(
            start_id=args.start_id,
            end_id=args.end_id,
            resume=not args.no_resume,
            dump_raw_prompt=args.dump_raw_prompt,
            dump_final_messages=args.dump_final_messages,
        )
    
    elif args.mode == 'evaluate':
        logger.info("⚠️  Please ensure Qwen2.5-32B is running on port 8002")
        logger.info(f"   Check: curl http://localhost:8002/v1/models\n")
        await evaluate_answers(resume=not args.no_resume)
    
    elif args.mode == 'stats':
        calculate_statistics()
    
    logger.info(f"\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
