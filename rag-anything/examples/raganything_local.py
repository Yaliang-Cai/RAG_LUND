#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Local RAGAnything example with optional resilience and callbacks.
"""

import argparse
import asyncio
import os
from typing import Any

from raganything.callbacks import ProcessingCallback
from raganything.services.local_rag import LocalRagService, LocalRagSettings


class DemoLoggingCallback(ProcessingCallback):
    def on_parse_start(self, file_path: str, parser: str = "", **kwargs: Any) -> None:
        print(f"[callback] parse_start: {file_path} parser={parser}")

    def on_document_complete(
        self,
        file_path: str,
        doc_id: str = "",
        duration_seconds: float = 0.0,
        **kwargs: Any,
    ) -> None:
        print(
            f"[callback] document_complete: file={file_path} doc_id={doc_id} "
            f"duration={duration_seconds:.2f}s"
        )

    def on_query_complete(
        self,
        query: str,
        mode: str = "",
        duration_seconds: float = 0.0,
        result_length: int = 0,
        **kwargs: Any,
    ) -> None:
        print(
            f"[callback] query_complete: mode={mode} duration={duration_seconds:.2f}s "
            f"result_length={result_length}"
        )


async def process_with_rag(service: LocalRagService, file_path: str, doc_id: str) -> None:
    final_doc_id = await service.ingest(file_path, doc_id=doc_id)

    queries = [
        "What is the paper's main contribution?",
        "Which baseline methods are compared and what are the core results?",
        "What are the limitations discussed by the authors?",
    ]

    for i, query in enumerate(queries, 1):
        service.logger.info("Query %d/%d: %s", i, len(queries), query)
        result = await service.query(
            final_doc_id,
            query,
            mode="hybrid",
            enable_rerank=True,
            vlm_enhanced=True,
        )
        service.logger.info("Answer:\n%s\n", result)

    metrics_summary = service.get_metrics_summary()
    if metrics_summary:
        service.logger.info("\n%s", metrics_summary)

    events = service.get_callback_events(final_doc_id)
    if events:
        service.logger.info("Callback event count for %s: %d", final_doc_id, len(events))


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAnything local example")
    parser.add_argument("--path", "-p", required=True, help="Input file or folder path")
    parser.add_argument("--id", "-i", required=True, help="Workspace name (doc_id)")
    parser.add_argument(
        "--enable_resilience",
        action="store_true",
        help="Enable service-level retry + circuit breaker for ingest/query",
    )
    parser.add_argument(
        "--enable_metrics_callback",
        action="store_true",
        help="Enable built-in metrics callback and print summary",
    )
    parser.add_argument(
        "--enable_callback_event_log",
        action="store_true",
        help="Enable callback event log collection in callback manager",
    )
    parser.add_argument(
        "--register_demo_callback",
        action="store_true",
        help="Register a demo callback that prints parse/document/query events",
    )
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Input not found: {args.path}")
        return

    settings = LocalRagSettings.from_env()
    if args.enable_resilience:
        settings.enable_resilience = True
    if args.enable_metrics_callback:
        settings.enable_metrics_callback = True
    if args.enable_callback_event_log:
        settings.enable_callback_event_log = True

    service = LocalRagService(settings)
    if args.register_demo_callback:
        service.register_callback(DemoLoggingCallback())

    asyncio.run(process_with_rag(service, args.path, args.id))


if __name__ == "__main__":
    main()
