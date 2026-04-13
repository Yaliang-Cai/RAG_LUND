#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProfileSpec:
    key: str
    description: str
    enable_entity_disambiguation: bool
    enable_synonym_linking: bool
    enable_multi_hop: bool
    reuse_index_from: str | None = None


PROFILE_SPECS: dict[str, ProfileSpec] = {
    "v0": ProfileSpec(
        key="v0",
        description="V0 (V1=off,V2=off,V3=off)",
        enable_entity_disambiguation=False,
        enable_synonym_linking=False,
        enable_multi_hop=False,
    ),
    "v0_v1": ProfileSpec(
        key="v0_v1",
        description="V0+V1 (V1=on,V2=off,V3=off)",
        enable_entity_disambiguation=True,
        enable_synonym_linking=False,
        enable_multi_hop=False,
    ),
    "v0_v1_v2": ProfileSpec(
        key="v0_v1_v2",
        description="V0+V1+V2 (V1=on,V2=on,V3=off)",
        enable_entity_disambiguation=True,
        enable_synonym_linking=True,
        enable_multi_hop=False,
    ),
    "v0_v1_v2_v3": ProfileSpec(
        key="v0_v1_v2_v3",
        description="V0+V1+V2+V3 query-only on V0+V1+V2 index",
        enable_entity_disambiguation=True,
        enable_synonym_linking=True,
        enable_multi_hop=True,
        reuse_index_from="v0_v1_v2",
    ),
}


PROFILE_ALIASES: dict[str, str] = {
    "db_only": "v0",
    "db_v1": "v0_v1",
    "db_v1_v2": "v0_v1_v2",
    "db_v1_v2_v3": "v0_v1_v2_v3",
    "db+v1": "v0_v1",
    "db+v1+v2": "v0_v1_v2",
    "db+v1+v2+v3": "v0_v1_v2_v3",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bool_arg(v: bool) -> str:
    return "true" if bool(v) else "false"


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _sanitize_workspace_fragment(raw: str) -> str:
    token = re.sub(r"[^0-9A-Za-z_-]+", "_", str(raw or "").strip())
    token = token.strip("_")
    return token or "run"


def _resolve_workspace_prefixes(
    *,
    run_id: str,
    shared_workspace_prefix: str,
    surge_workspace_prefix: str,
) -> tuple[str, str]:
    run_token = _sanitize_workspace_fragment(run_id)
    shared_prefix = str(shared_workspace_prefix or "").strip()
    surge_prefix = str(surge_workspace_prefix or "").strip()
    if not shared_prefix:
        shared_prefix = f"docbench_shared_{run_token}"
    if not surge_prefix:
        surge_prefix = f"surge_fast_{run_token}"
    return shared_prefix, surge_prefix


def _resolve_profile_key(raw: str) -> str:
    token = str(raw or "").strip().lower().replace("-", "_")
    if token in PROFILE_SPECS:
        return token
    alias = PROFILE_ALIASES.get(token)
    if alias:
        return alias
    raise ValueError(
        f"unknown profile: {raw!r}. valid: {sorted(PROFILE_SPECS.keys()) + sorted(PROFILE_ALIASES.keys())}"
    )


def _resolve_profiles(raw_profiles: list[str] | None, include_v3: bool) -> list[ProfileSpec]:
    if raw_profiles:
        keys: list[str] = []
        for raw in raw_profiles:
            resolved = _resolve_profile_key(raw)
            if resolved not in keys:
                keys.append(resolved)
    else:
        keys = ["v0", "v0_v1", "v0_v1_v2"]
        if include_v3:
            keys.append("v0_v1_v2_v3")
    return [PROFILE_SPECS[key] for key in keys]


def _make_base_env(project_root: Path, lightrag_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    pythonpath_entries: list[str] = []
    existing = str(env.get("PYTHONPATH", "")).strip()
    if existing:
        pythonpath_entries.append(existing)
    pythonpath_entries.append(str(project_root))
    pythonpath_entries.append(str(lightrag_root))
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_progress(
    *,
    progress_file: Path,
    latest_file: Path,
    run_id: str,
    profile_key: str | None,
    stage: str,
    status: str,
    message: str,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "timestamp": _now_iso(),
        "run_id": run_id,
        "profile": profile_key,
        "stage": stage,
        "status": status,
        "message": message,
    }
    if extra:
        payload["extra"] = extra
    _append_jsonl(progress_file, payload)
    _write_json(latest_file, payload)


def _run_command(
    *,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_file: Path,
) -> tuple[int, float]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("# command\n")
        f.write(" ".join(command) + "\n\n")
        f.flush()
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(proc.returncode), time.time() - start


def _build_shared_base_command(
    *,
    python_exe: str,
    profile: ProfileSpec,
    shared_workspace_id: str,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        python_exe,
        "-m",
        "evaluate_local.DocBench.evaluate_shared",
        "--start_id",
        str(args.start_id),
        "--end_id",
        str(args.end_id),
        "--shared_workspace_id",
        shared_workspace_id,
        "--max_async_ingest",
        str(args.max_async_ingest),
        "--max_async_generate",
        str(args.max_async_generate),
        "--max_async_judge",
        str(args.max_async_judge),
        "--raganything_eval_setup",
        "--enable-entity-disambiguation",
        _bool_arg(profile.enable_entity_disambiguation),
        "--enable-synonym-linking",
        _bool_arg(profile.enable_synonym_linking),
        "--enable-multi-hop",
        _bool_arg(profile.enable_multi_hop),
        "--multi-hop-depth",
        str(args.multi_hop_depth),
        "--ppr-damping",
        str(args.ppr_damping),
        "--ppr-top-k",
        str(args.ppr_top_k),
        "--passage-node-weight",
        str(args.passage_node_weight),
    ]
    if not args.resume:
        cmd.append("--no_resume")
    return cmd


def _build_surge_base_command(
    *,
    python_exe: str,
    profile: ProfileSpec,
    surge_workspace_id: str,
    args: argparse.Namespace,
) -> list[str]:
    return [
        python_exe,
        "-m",
        "evaluate_local.SurGE.evaluate_surge_fast",
        "--data-root",
        str(args.surge_data_root),
        "--subset-dir",
        str(args.surge_subset_dir),
        "--queries-file",
        str(args.surge_queries_file),
        "--surveys-file",
        str(args.surge_surveys_file),
        "--chunks-file",
        str(args.surge_chunks_file),
        "--corpus-file",
        str(args.surge_corpus_file),
        "--workspace-id",
        surge_workspace_id,
        "--query-mode",
        str(args.query_mode),
        "--top-k",
        str(args.top_k),
        "--chunk-top-k",
        str(args.chunk_top_k),
        "--k-list",
        str(args.k_list),
        "--survey-k-list",
        str(args.survey_k_list),
        "--enable-rerank",
        "true",
        "--batch-doc-concurrency",
        str(args.batch_doc_concurrency),
        "--ingest-batch-size",
        str(args.ingest_batch_size),
        "--llm-model-max-async",
        str(args.llm_model_max_async),
        "--max-concurrency",
        str(args.max_concurrency),
        "--max-retries",
        str(args.max_retries),
        "--limit",
        str(args.limit),
        "--enable-entity-disambiguation",
        _bool_arg(profile.enable_entity_disambiguation),
        "--enable-synonym-linking",
        _bool_arg(profile.enable_synonym_linking),
        "--enable-multi-hop",
        _bool_arg(profile.enable_multi_hop),
        "--multi-hop-depth",
        str(args.multi_hop_depth),
        "--ppr-damping",
        str(args.ppr_damping),
        "--ppr-top-k",
        str(args.ppr_top_k),
        "--passage-node-weight",
        str(args.passage_node_weight),
    ]


def _run_one_stage(
    *,
    run_id: str,
    progress_file: Path,
    latest_file: Path,
    profile_key: str,
    stage_name: str,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_file: Path,
    dry_run: bool,
) -> tuple[bool, float]:
    _write_progress(
        progress_file=progress_file,
        latest_file=latest_file,
        run_id=run_id,
        profile_key=profile_key,
        stage=stage_name,
        status="running",
        message="stage started",
        extra={"command": command, "log_file": str(log_file)},
    )
    if dry_run:
        _write_progress(
            progress_file=progress_file,
            latest_file=latest_file,
            run_id=run_id,
            profile_key=profile_key,
            stage=stage_name,
            status="dry_run",
            message="stage skipped in dry-run mode",
            extra={"command": command, "log_file": str(log_file)},
        )
        return True, 0.0

    code, elapsed = _run_command(command=command, cwd=cwd, env=env, log_file=log_file)
    ok = code == 0
    _write_progress(
        progress_file=progress_file,
        latest_file=latest_file,
        run_id=run_id,
        profile_key=profile_key,
        stage=stage_name,
        status="ok" if ok else "failed",
        message="stage completed" if ok else f"stage failed with exit code {code}",
        extra={"returncode": code, "elapsed_sec": round(elapsed, 3), "log_file": str(log_file)},
    )
    return ok, elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run evaluate_shared + evaluate_surge_fast ablation groups with strict output "
            "isolation and progress tracking."
        )
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=None,
        help=(
            "Profiles to run. Supports keys/aliases: "
            "v0 v0_v1 v0_v1_v2 v0_v1_v2_v3 db_only db_v1 db_v1_v2 db_v1_v2_v3"
        ),
    )
    parser.add_argument(
        "--include-v3",
        action="store_true",
        help="Also run DB+V1+V2+V3 query-only profile on DB+V1+V2 index workspace.",
    )
    parser.add_argument(
        "--tasks",
        choices=["both", "shared", "surge"],
        default="both",
        help="Which evaluator pipelines to run.",
    )
    parser.add_argument("--run-id", type=str, default="", help="Run id folder name.")
    parser.add_argument(
        "--allow-reuse-run-id",
        action="store_true",
        help="Allow writing into an existing run-id directory. Default: fail-fast to avoid overwrite/mixing.",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="evaluate_local/ablation_runs",
        help="Root folder for ablation run outputs.",
    )
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Pass resume behavior to evaluate_shared (default is no_resume).",
    )
    parser.add_argument(
        "--run-shared-evaluate",
        action="store_true",
        help="Also run evaluate_shared --mode evaluate after shared_generate.",
    )
    parser.add_argument(
        "--run-shared-stats",
        action="store_true",
        help="Also run evaluate_shared --mode stats (typically together with --run-shared-evaluate).",
    )
    parser.add_argument(
        "--skip-shared-generate",
        action="store_true",
        help=(
            "Skip shared_generate stage. Useful when answers are already generated "
            "and you only want evaluate/stats."
        ),
    )

    parser.add_argument(
        "--shared-workspace-prefix",
        type=str,
        default="",
        help=(
            "Workspace prefix for evaluate_shared. Default: auto -> "
            "docbench_shared_<run-id>."
        ),
    )
    parser.add_argument(
        "--surge-workspace-prefix",
        type=str,
        default="",
        help=(
            "Workspace prefix for evaluate_surge_fast. Default: auto -> "
            "surge_fast_<run-id>."
        ),
    )
    parser.add_argument(
        "--docbench-data-root",
        type=str,
        default="",
        help="Optional override for DOCBENCH_SHARED_DATA_ROOT.",
    )
    parser.add_argument(
        "--shared-mineru-output-dir",
        type=str,
        default="evaluate_local/DocBench/docbench_shared_results/mineru_outputs",
        help=(
            "Directory for shared MinerU artifacts. "
            "Defaults to evaluate_local/DocBench/docbench_shared_results/mineru_outputs."
        ),
    )

    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=49)
    parser.add_argument("--max-async-ingest", type=int, default=4)
    parser.add_argument("--max-async-generate", type=int, default=6)
    parser.add_argument("--max-async-judge", type=int, default=4)

    parser.add_argument(
        "--surge-data-root",
        type=str,
        default="/data/y50056788/Yaliang/datasets_for_eval/data_for_SurGE",
    )
    parser.add_argument("--surge-subset-dir", type=str, default="subset_output")
    parser.add_argument("--surge-queries-file", type=str, default="subset_queries.json")
    parser.add_argument("--surge-surveys-file", type=str, default="subset_surveys.json")
    parser.add_argument("--surge-chunks-file", type=str, default="subset_chunks.jsonl")
    parser.add_argument("--surge-corpus-file", type=str, default="subset_corpus.json")
    parser.add_argument(
        "--query-mode",
        choices=["local", "global", "hybrid", "naive", "mix", "bypass"],
        default="hybrid",
    )
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--chunk-top-k", type=int, default=0)
    parser.add_argument("--k-list", type=str, default="5,10,20,30,50")
    parser.add_argument("--survey-k-list", type=str, default="50,100,200,500")
    parser.add_argument("--batch-doc-concurrency", type=int, default=2)
    parser.add_argument("--ingest-batch-size", type=int, default=384)
    parser.add_argument("--llm-model-max-async", type=int, default=48)
    parser.add_argument("--max-concurrency", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--multi-hop-depth", type=int, default=2)
    parser.add_argument("--ppr-damping", type=float, default=0.5)
    parser.add_argument("--ppr-top-k", type=int, default=50)
    parser.add_argument("--passage-node-weight", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.run_shared_stats and not args.run_shared_evaluate:
        raise ValueError("--run-shared-stats requires --run-shared-evaluate.")
    if (
        args.tasks in ("both", "shared")
        and bool(args.skip_shared_generate)
        and not (bool(args.run_shared_evaluate) or bool(args.run_shared_stats))
    ):
        raise ValueError(
            "--skip-shared-generate requires at least one of "
            "--run-shared-evaluate/--run-shared-stats for shared tasks."
        )
    profiles = _resolve_profiles(args.profiles, include_v3=bool(args.include_v3))

    eval_local_dir = Path(__file__).resolve().parent
    project_root = eval_local_dir.parent
    projects_root = project_root.parent
    lightrag_root = projects_root / "lightrag"

    run_id = str(args.run_id or "").strip() or _default_run_id()
    shared_workspace_prefix, surge_workspace_prefix = _resolve_workspace_prefixes(
        run_id=run_id,
        shared_workspace_prefix=str(args.shared_workspace_prefix),
        surge_workspace_prefix=str(args.surge_workspace_prefix),
    )
    runs_root = (project_root / args.runs_root).resolve()
    run_root = (runs_root / run_id).resolve()
    if run_root.exists() and any(run_root.iterdir()) and not bool(args.allow_reuse_run_id):
        raise RuntimeError(
            f"run-id already exists and is not empty: {run_root}. "
            "Use a new --run-id (recommended) or pass --allow-reuse-run-id explicitly."
        )
    run_root.mkdir(parents=True, exist_ok=True)
    progress_file = run_root / "progress.jsonl"
    latest_file = run_root / "progress_latest.json"

    summary: dict[str, Any] = {
        "run_id": run_id,
        "run_root": str(run_root),
        "started_at": _now_iso(),
        "profiles": [],
        "failed_profiles": [],
    }

    python_exe = str(args.python_exe)
    base_env = _make_base_env(project_root=project_root, lightrag_root=lightrag_root)
    if str(args.docbench_data_root or "").strip():
        base_env["DOCBENCH_SHARED_DATA_ROOT"] = str(args.docbench_data_root).strip()
    shared_mineru_output_dir: Path | None = None
    if str(args.shared_mineru_output_dir or "").strip():
        shared_mineru_output_dir = Path(str(args.shared_mineru_output_dir).strip())
        if not shared_mineru_output_dir.is_absolute():
            shared_mineru_output_dir = (project_root / shared_mineru_output_dir).resolve()

    _write_progress(
        progress_file=progress_file,
        latest_file=latest_file,
        run_id=run_id,
        profile_key=None,
        stage="bootstrap",
        status="running",
        message="ablation runner started",
        extra={
            "run_root": str(run_root),
            "profiles": [p.key for p in profiles],
            "tasks": args.tasks,
            "dry_run": bool(args.dry_run),
            "run_shared_evaluate": bool(args.run_shared_evaluate),
            "run_shared_stats": bool(args.run_shared_stats),
            "skip_shared_generate": bool(args.skip_shared_generate),
            "shared_mineru_output_dir": (
                str(shared_mineru_output_dir) if shared_mineru_output_dir else ""
            ),
            "shared_workspace_prefix": shared_workspace_prefix,
            "surge_workspace_prefix": surge_workspace_prefix,
        },
    )

    abort_run = False
    for profile in profiles:
        workspace_key = profile.reuse_index_from or profile.key
        shared_workspace_id = f"{shared_workspace_prefix}_{workspace_key}"
        surge_workspace_id = f"{surge_workspace_prefix}_{workspace_key}"

        profile_dir = run_root / profile.key
        shared_output_dir = profile_dir / "evaluate_shared"
        surge_output_dir = profile_dir / "evaluate_surge_fast"
        profile_log_dir = profile_dir / "logs"
        profile_log_dir.mkdir(parents=True, exist_ok=True)

        profile_meta = {
            "profile": profile.key,
            "description": profile.description,
            "workspace_key": workspace_key,
            "shared_workspace_id": shared_workspace_id,
            "surge_workspace_id": surge_workspace_id,
            "enable_entity_disambiguation": profile.enable_entity_disambiguation,
            "enable_synonym_linking": profile.enable_synonym_linking,
            "enable_multi_hop": profile.enable_multi_hop,
            "shared_output_dir": str(shared_output_dir),
            "surge_output_dir": str(surge_output_dir),
            "shared_mineru_output_dir": (
                str(shared_mineru_output_dir) if shared_mineru_output_dir else ""
            ),
        }
        _write_json(profile_dir / "profile_config.json", profile_meta)
        summary["profiles"].append(profile_meta)

        _write_progress(
            progress_file=progress_file,
            latest_file=latest_file,
            run_id=run_id,
            profile_key=profile.key,
            stage="profile_start",
            status="running",
            message="profile started",
            extra=profile_meta,
        )

        profile_ok = True

        if args.tasks in ("both", "shared"):
            env_shared = dict(base_env)
            env_shared["DOCBENCH_SHARED_OUTPUT_DIR"] = str(shared_output_dir)
            if shared_mineru_output_dir is not None:
                env_shared["DOCBENCH_SHARED_MINERU_OUTPUT_DIR"] = str(shared_mineru_output_dir)
            env_shared["NEO4J_WORKSPACE"] = shared_workspace_id
            env_shared["QDRANT_WORKSPACE"] = shared_workspace_id

            shared_base = _build_shared_base_command(
                python_exe=python_exe,
                profile=profile,
                shared_workspace_id=shared_workspace_id,
                args=args,
            )

            run_shared_generate = not bool(args.skip_shared_generate)
            if run_shared_generate:
                shared_generate = list(shared_base) + ["--mode", "generate"]
                ok, _ = _run_one_stage(
                    run_id=run_id,
                    progress_file=progress_file,
                    latest_file=latest_file,
                    profile_key=profile.key,
                    stage_name="shared_generate",
                    command=shared_generate,
                    cwd=project_root,
                    env=env_shared,
                    log_file=profile_log_dir / "shared_generate.log",
                    dry_run=bool(args.dry_run),
                )
                profile_ok = profile_ok and ok
                if not ok and not args.continue_on_error:
                    abort_run = True
            else:
                _write_progress(
                    progress_file=progress_file,
                    latest_file=latest_file,
                    run_id=run_id,
                    profile_key=profile.key,
                    stage="shared_generate",
                    status="skipped",
                    message="shared_generate skipped by --skip-shared-generate",
                    extra={"shared_workspace_id": shared_workspace_id},
                )

            if profile_ok and bool(args.run_shared_evaluate):
                shared_evaluate = list(shared_base) + ["--mode", "evaluate"]
                ok, _ = _run_one_stage(
                    run_id=run_id,
                    progress_file=progress_file,
                    latest_file=latest_file,
                    profile_key=profile.key,
                    stage_name="shared_evaluate",
                    command=shared_evaluate,
                    cwd=project_root,
                    env=env_shared,
                    log_file=profile_log_dir / "shared_evaluate.log",
                    dry_run=bool(args.dry_run),
                )
                profile_ok = profile_ok and ok
                if not ok and not args.continue_on_error:
                    abort_run = True

            if profile_ok and bool(args.run_shared_stats):
                shared_stats = list(shared_base) + ["--mode", "stats"]
                ok, _ = _run_one_stage(
                    run_id=run_id,
                    progress_file=progress_file,
                    latest_file=latest_file,
                    profile_key=profile.key,
                    stage_name="shared_stats",
                    command=shared_stats,
                    cwd=project_root,
                    env=env_shared,
                    log_file=profile_log_dir / "shared_stats.log",
                    dry_run=bool(args.dry_run),
                )
                profile_ok = profile_ok and ok
                if not ok and not args.continue_on_error:
                    abort_run = True

        if profile_ok and args.tasks in ("both", "surge"):
            env_surge = dict(base_env)
            env_surge["SURGE_FAST_OUTPUT_DIR"] = str(surge_output_dir)
            env_surge["NEO4J_WORKSPACE"] = surge_workspace_id
            env_surge["QDRANT_WORKSPACE"] = surge_workspace_id

            surge_base = _build_surge_base_command(
                python_exe=python_exe,
                profile=profile,
                surge_workspace_id=surge_workspace_id,
                args=args,
            )

            surge_retrieval = list(surge_base) + ["--mode", "retrieval"]
            ok, _ = _run_one_stage(
                run_id=run_id,
                progress_file=progress_file,
                latest_file=latest_file,
                profile_key=profile.key,
                stage_name="surge_retrieval",
                command=surge_retrieval,
                cwd=project_root,
                env=env_surge,
                log_file=profile_log_dir / "surge_retrieval.log",
                dry_run=bool(args.dry_run),
            )
            profile_ok = profile_ok and ok
            if not ok and not args.continue_on_error:
                abort_run = True

            if profile_ok:
                surge_survey = list(surge_base) + ["--mode", "survey", "--survey-stage", "retrieval"]
                ok, _ = _run_one_stage(
                    run_id=run_id,
                    progress_file=progress_file,
                    latest_file=latest_file,
                    profile_key=profile.key,
                    stage_name="surge_survey_retrieval",
                    command=surge_survey,
                    cwd=project_root,
                    env=env_surge,
                    log_file=profile_log_dir / "surge_survey_retrieval.log",
                    dry_run=bool(args.dry_run),
                )
                profile_ok = profile_ok and ok
                if not ok and not args.continue_on_error:
                    abort_run = True

        _write_progress(
            progress_file=progress_file,
            latest_file=latest_file,
            run_id=run_id,
            profile_key=profile.key,
            stage="profile_end",
            status="ok" if profile_ok else "failed",
            message="profile completed" if profile_ok else "profile failed",
            extra={"profile_dir": str(profile_dir)},
        )

        if not profile_ok:
            summary["failed_profiles"].append(profile.key)
        if abort_run:
            break

    summary["finished_at"] = _now_iso()
    summary["status"] = "ok" if not summary["failed_profiles"] else "failed"
    summary_path = run_root / "summary.json"
    _write_json(summary_path, summary)

    _write_progress(
        progress_file=progress_file,
        latest_file=latest_file,
        run_id=run_id,
        profile_key=None,
        stage="finish",
        status=summary["status"],
        message="ablation runner finished",
        extra={"summary_file": str(summary_path), "failed_profiles": summary["failed_profiles"]},
    )

    return 0 if not summary["failed_profiles"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
