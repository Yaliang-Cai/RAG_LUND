#!/usr/bin/env python
"""Optuna runner for MultiHopQA PPR + hybrid retrieval + synonym edges HPO."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


DATASETS = ("hotpotqa", "musique", "2wiki")
SEARCH_SPACE = {
    "top_k": [5, 10, 20, 40],
    "ppr_qa_top_k": [3, 5, 8, 10],
    "ppr_top_k": [25, 50, 100],
    "passage_node_weight": [0, 0.02, 0.05, 0.1, 0.2],
    "ppr_damping": [0.35, 0.5, 0.65, 0.8],
    "hub_penalty_threshold": [0, 1, 10, 25, 50, 100],
}


@dataclass(frozen=True)
class HPOConfig:
    top_k: int
    ppr_qa_top_k: int
    ppr_top_k: int
    passage_node_weight: float
    ppr_damping: float
    hub_penalty_threshold: int

    def to_params(self) -> dict[str, int | float]:
        return {
            "top_k": self.top_k,
            "ppr_qa_top_k": self.ppr_qa_top_k,
            "ppr_top_k": self.ppr_top_k,
            "passage_node_weight": self.passage_node_weight,
            "ppr_damping": self.ppr_damping,
            "hub_penalty_threshold": self.hub_penalty_threshold,
        }

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "HPOConfig":
        return cls(
            top_k=int(params["top_k"]),
            ppr_qa_top_k=int(params["ppr_qa_top_k"]),
            ppr_top_k=int(params["ppr_top_k"]),
            passage_node_weight=float(params["passage_node_weight"]),
            ppr_damping=float(params["ppr_damping"]),
            hub_penalty_threshold=int(params["hub_penalty_threshold"]),
        )


ANCHOR_CONFIG = HPOConfig(
    top_k=10,
    ppr_qa_top_k=5,
    ppr_top_k=50,
    passage_node_weight=0.05,
    ppr_damping=0.5,
    hub_penalty_threshold=50,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _dataset_file_prefix(dataset: str) -> str:
    return "2wikimultihopqa" if dataset == "2wiki" else dataset


def _threshold_label(threshold: float) -> str:
    label = f"{float(threshold):.12g}"
    return label.replace("-", "m").replace(".", "p")


def _macro_metrics(dataset_metrics: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    return {
        "macro_f1": round(mean(float(m["f1"]) for m in dataset_metrics.values()), 10),
        "macro_em": round(mean(float(m["em"]) for m in dataset_metrics.values()), 10),
        "macro_recall@2": round(mean(float(m["recall@2"]) for m in dataset_metrics.values()), 10),
        "macro_recall@5": round(mean(float(m["recall@5"]) for m in dataset_metrics.values()), 10),
    }


def _resolve_working_dir(workspace_root: Path, dataset: str, workspace_id: str) -> Path:
    nested = workspace_root / dataset / workspace_id
    flat = workspace_root / dataset
    if (nested / "multihopqa_index_profile.json").exists():
        return nested
    if (flat / "multihopqa_index_profile.json").exists():
        return flat
    raise FileNotFoundError(f"Missing workspace artifacts. Checked {nested} and {flat}.")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {path}") from exc


def _check_data_ready(data_dir: Path, datasets: Sequence[str]) -> None:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing HippoRAG2 data dir: {data_dir}")
    for dataset in datasets:
        prefix = _dataset_file_prefix(dataset)
        for suffix in (".json", "_corpus.json"):
            path = data_dir / f"{prefix}{suffix}"
            if not path.exists():
                raise FileNotFoundError(f"Missing HippoRAG2 data file: {path}")


def _require_int(payload: Mapping[str, Any], key: str, label: str) -> int:
    if key not in payload or payload[key] is None:
        raise ValueError(f"{label}.{key}=missing")
    try:
        return int(payload[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}.{key}={payload[key]!r}") from exc


def _check_workspace_ready(working_dir: Path, dataset: str, workspace_id: str, chunk_size: int) -> None:
    profile_path = working_dir / "multihopqa_index_profile.json"
    manifest_path = working_dir / "multihopqa_ingest_manifest.json"
    source_map_path = working_dir / "multihopqa_chunk_source_map.json"
    for path in (profile_path, manifest_path, source_map_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing workspace file: {path}")

    profile = _load_json(profile_path)
    manifest = _load_json(manifest_path)
    source_map = _load_json(source_map_path)
    index_profile = profile.get("index_profile") or {}
    ingest_stats = manifest.get("ingest_stats") or {}
    source_payload = source_map.get("map") or {}

    errors: list[str] = []
    if profile.get("workspace_id") != workspace_id:
        errors.append(f"profile.workspace_id={profile.get('workspace_id')!r}")
    if manifest.get("workspace_id") != workspace_id:
        errors.append(f"manifest.workspace_id={manifest.get('workspace_id')!r}")
    if source_map.get("workspace_id") != workspace_id:
        errors.append(f"source_map.workspace_id={source_map.get('workspace_id')!r}")
    if profile.get("dataset") != dataset:
        errors.append(f"profile.dataset={profile.get('dataset')!r}")
    if manifest.get("dataset") != dataset:
        errors.append(f"manifest.dataset={manifest.get('dataset')!r}")
    if source_map.get("dataset") != dataset:
        errors.append(f"source_map.dataset={source_map.get('dataset')!r}")
    if manifest.get("corpus_source") != "hipporag2":
        errors.append(f"manifest.corpus_source={manifest.get('corpus_source')!r}")
    if int(index_profile.get("chunk_token_size") or 0) != chunk_size:
        errors.append(f"index_profile.chunk_token_size={index_profile.get('chunk_token_size')!r}")
    if index_profile.get("enable_synonym_linking") is not False:
        errors.append(f"index_profile.enable_synonym_linking={index_profile.get('enable_synonym_linking')!r}")
    try:
        if _require_int(profile, "n_samples", "profile") != 0:
            errors.append(f"profile.n_samples={profile.get('n_samples')!r}")
        if _require_int(profile, "seed", "profile") != 0:
            errors.append(f"profile.seed={profile.get('seed')!r}")
        if _require_int(manifest, "n_samples", "manifest") != 0:
            errors.append(f"manifest.n_samples={manifest.get('n_samples')!r}")
        if _require_int(manifest, "seed", "manifest") != 0:
            errors.append(f"manifest.seed={manifest.get('seed')!r}")
        if _require_int(source_map, "n_samples", "source_map") != 0:
            errors.append(f"source_map.n_samples={source_map.get('n_samples')!r}")
        if _require_int(source_map, "seed", "source_map") != 0:
            errors.append(f"source_map.seed={source_map.get('seed')!r}")
    except ValueError as exc:
        errors.append(str(exc))
    if int(ingest_stats.get("failed_now_batch_count") or 0) != 0:
        errors.append(f"failed_now_batch_count={ingest_stats.get('failed_now_batch_count')!r}")
    if int(source_map.get("map_size") or 0) <= 0:
        errors.append(f"source_map.map_size={source_map.get('map_size')!r}")
    if int(source_map.get("map_size") or 0) != len(source_payload):
        errors.append(f"source_map.map_size={source_map.get('map_size')!r}, actual={len(source_payload)}")
    if errors:
        raise ValueError("Workspace check failed: " + "; ".join(errors))


def _find_synonym_manifest(working_dir: Path, workspace_id: str) -> Path:
    candidates = [
        working_dir / workspace_id / "synonym_linking_manifest.json",
        working_dir / "synonym_linking_manifest.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Missing synonym_linking_manifest.json. Checked "
        + " and ".join(str(path) for path in candidates)
    )


def _check_synonym_manifest(working_dir: Path, workspace_id: str, synonym_threshold: float) -> None:
    manifest_path = _find_synonym_manifest(working_dir, workspace_id)
    payload = _load_json(manifest_path)
    errors: list[str] = []
    if payload.get("workspace_id") != workspace_id:
        errors.append(f"workspace_id={payload.get('workspace_id')!r}")
    if payload.get("status") != "completed":
        errors.append(f"status={payload.get('status')!r}")
    try:
        actual_threshold = float(payload.get("synonymy_threshold"))
    except (TypeError, ValueError):
        actual_threshold = math.nan
    if not math.isclose(actual_threshold, synonym_threshold, rel_tol=0.0, abs_tol=1e-9):
        errors.append(f"synonymy_threshold={payload.get('synonymy_threshold')!r}")
    if int(payload.get("created_edges") or 0) <= 0:
        errors.append(f"created_edges={payload.get('created_edges')!r}")
    if errors:
        raise ValueError("Synonym manifest check failed: " + "; ".join(errors))


def _build_eval_command(
    *,
    python_executable: str,
    evaluate_script: Path,
    dataset: str,
    workspace_id: str,
    working_dir: Path,
    data_dir: Path,
    output_dir: Path,
    config: HPOConfig,
    n_samples: int,
    seed: int,
    concurrency: int,
    recall_k: Sequence[int],
) -> list[str]:
    return [
        python_executable,
        str(evaluate_script),
        "--dataset",
        dataset,
        "--workspace",
        workspace_id,
        "--working-dir",
        str(working_dir),
        "--hipporag2-data-dir",
        str(data_dir),
        "--n-samples",
        str(n_samples),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--modes",
        "ppr",
        "--recall-k",
        *[str(k) for k in recall_k],
        "--concurrency",
        str(concurrency),
        "--top-k",
        str(config.top_k),
        "--chunk-top-k",
        str(config.ppr_qa_top_k),
        "--naive-top-k",
        "10",
        "--max-total-tokens",
        "45000",
        "--ppr-damping",
        str(config.ppr_damping),
        "--ppr-top-k",
        str(config.ppr_top_k),
        "--ppr-qa-top-k",
        str(config.ppr_qa_top_k),
        "--hub-penalty-threshold",
        str(config.hub_penalty_threshold),
        "--passage-node-weight",
        str(config.passage_node_weight),
        "--recognition-top-k",
        "20",
        "--linking-top-k",
        "5",
        "--ppr-post-rerank-fusion",
        "none",
        "--ppr-post-rerank-rrf-k",
        "60",
        "--ppr-synonym-weight-mode",
        "raw",
        "--no-enable-kg-rerank",
        "--no-ppr-enable-rerank",
        "--no-exclude-synonym-edges",
        "--keyword-fanout-mode",
        "joined",
        "--qdrant-retrieval-mode",
        "hybrid",
        "--answer-context-mode",
        "chunk_only_prompt",
        "--qa-prompt-style",
        "semantic_cot",
        "--answer-parse-mode",
        "answer_marker",
        "--bypass-query-cache",
        "--no-bypass-keywords-cache",
    ]


def _load_ppr_metrics(summary_path: Path) -> dict[str, float]:
    payload = _load_json(summary_path)
    metrics = (payload.get("results") or {}).get("ppr")
    if not metrics:
        raise ValueError(f"Missing ppr metrics in {summary_path}")
    return {
        "em": float(metrics["em"]),
        "f1": float(metrics["f1"]),
        "recall@2": float(metrics["recall@2"]),
        "recall@5": float(metrics["recall@5"]),
    }


def _run_dataset_eval(
    *,
    args: argparse.Namespace,
    dataset: str,
    workspace_id: str,
    working_dir: Path,
    output_dir: Path,
    config: HPOConfig,
    n_samples: int,
    seed: int,
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_eval_command(
        python_executable=args.python_executable,
        evaluate_script=args.evaluate_script,
        dataset=dataset,
        workspace_id=workspace_id,
        working_dir=working_dir,
        data_dir=args.data_dir,
        output_dir=output_dir,
        config=config,
        n_samples=n_samples,
        seed=seed,
        concurrency=args.concurrency,
        recall_k=args.recall_k,
    )
    env = os.environ.copy()
    env["RAGANYTHING_MIN_RERANK_SCORE"] = "0.3"
    env["MIN_RERANK_SCORE"] = "0.3"
    subprocess.run(cmd, cwd=args.repo_root, env=env, check=True)
    return _load_ppr_metrics(output_dir / f"{dataset}_summary.json")


def _prepare_workspace_map(args: argparse.Namespace) -> dict[str, tuple[str, Path]]:
    _check_data_ready(args.data_dir, args.datasets)
    workspaces: dict[str, tuple[str, Path]] = {}
    for dataset in args.datasets:
        workspace_id = f"{dataset}_hr2_v0"
        working_dir = _resolve_working_dir(args.workspace_root, dataset, workspace_id)
        _check_workspace_ready(working_dir, dataset, workspace_id, args.chunk_size)
        _check_synonym_manifest(working_dir, workspace_id, args.synonym_threshold)
        workspaces[dataset] = (workspace_id, working_dir)
    return workspaces


def _suggest_config(trial: Any) -> HPOConfig:
    return HPOConfig(
        top_k=trial.suggest_categorical("top_k", SEARCH_SPACE["top_k"]),
        ppr_qa_top_k=trial.suggest_categorical("ppr_qa_top_k", SEARCH_SPACE["ppr_qa_top_k"]),
        ppr_top_k=trial.suggest_categorical("ppr_top_k", SEARCH_SPACE["ppr_top_k"]),
        passage_node_weight=trial.suggest_categorical(
            "passage_node_weight", SEARCH_SPACE["passage_node_weight"]
        ),
        ppr_damping=trial.suggest_categorical("ppr_damping", SEARCH_SPACE["ppr_damping"]),
        hub_penalty_threshold=trial.suggest_categorical(
            "hub_penalty_threshold", SEARCH_SPACE["hub_penalty_threshold"]
        ),
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_config_across_datasets(
    *,
    args: argparse.Namespace,
    workspaces: Mapping[str, tuple[str, Path]],
    config_name: str,
    config: HPOConfig,
    n_samples: int,
    seed: int,
    output_root: Path,
    trial: Any | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    dataset_metrics: dict[str, dict[str, float]] = {}
    for step, dataset in enumerate(args.datasets):
        workspace_id, working_dir = workspaces[dataset]
        dataset_metrics[dataset] = _run_dataset_eval(
            args=args,
            dataset=dataset,
            workspace_id=workspace_id,
            working_dir=working_dir,
            output_dir=output_root / config_name / dataset,
            config=config,
            n_samples=n_samples,
            seed=seed,
        )
        if trial is not None:
            trial.report(_macro_metrics(dataset_metrics)["macro_f1"], step)
            if step < len(args.datasets) - 1 and trial.should_prune():
                raise _optuna().TrialPruned()
    macro = _macro_metrics(dataset_metrics)
    _write_json(
        output_root / config_name / "hpo_summary.json",
        {"config": config.to_params(), "dataset_metrics": dataset_metrics, **macro},
    )
    return dataset_metrics, macro


def _optuna() -> Any:
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "Optuna is required for this runner. Install it in the Linux eval environment with `pip install optuna`."
        ) from exc
    return optuna


def _create_study(args: argparse.Namespace) -> Any:
    optuna = _optuna()
    optuna_jobs = args.optuna_jobs
    args.study_db.parent.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(
        seed=args.optuna_seed,
        n_startup_trials=12,
        multivariate=True,
        group=True,
        constant_liar=optuna_jobs > 1,
    )
    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=2,
            n_min_trials=4,
        )
    else:
        pruner = optuna.pruners.NopPruner()
    return optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{args.study_db}",
        load_if_exists=True,
    )


def _ensure_anchor_trial(study: Any) -> None:
    anchor = ANCHOR_CONFIG.to_params()
    for trial in study.get_trials(deepcopy=False):
        if {k: trial.params.get(k) for k in anchor} == anchor:
            return
    study.enqueue_trial(ANCHOR_CONFIG.to_params())


def _run_dev(args: argparse.Namespace) -> None:
    workspaces = _prepare_workspace_map(args)
    args.results_root.mkdir(parents=True, exist_ok=True)
    study = _create_study(args)
    _ensure_anchor_trial(study)

    def objective(trial: Any) -> float:
        config = _suggest_config(trial)
        config_name = f"trial_{trial.number:04d}"
        dataset_metrics, macro = _run_config_across_datasets(
            args=args,
            workspaces=workspaces,
            config_name=config_name,
            config=config,
            n_samples=args.n_samples,
            seed=args.seed,
            output_root=args.results_root,
            trial=trial,
        )
        trial.set_user_attr("config", config.to_params())
        trial.set_user_attr("dataset_metrics", dataset_metrics)
        for key, value in macro.items():
            trial.set_user_attr(key, value)
        return macro["macro_f1"]

    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.optuna_jobs)
    _write_trials_table(study, args.results_root / "trials.tsv")
    _write_json(args.results_root / "best_trial.json", _trial_payload(study.best_trial))


def _trial_payload(trial: Any) -> dict[str, Any]:
    return {
        "number": trial.number,
        "value": trial.value,
        "params": dict(trial.params),
        "user_attrs": dict(trial.user_attrs),
    }


def _completed_trials_sorted(study: Any) -> list[Any]:
    optuna = _optuna()
    trials = [t for t in study.get_trials(deepcopy=False) if t.state == optuna.trial.TrialState.COMPLETE]
    return sorted(trials, key=lambda t: float(t.value), reverse=True)


def _write_trials_table(study: Any, path: Path) -> None:
    rows = [
        "rank\ttrial\tmacro_f1\tmacro_em\tmacro_recall@2\tmacro_recall@5\t"
        "top_k\tppr_qa_top_k\tppr_top_k\tpassage_node_weight\tppr_damping\thub_penalty_threshold"
    ]
    for rank, trial in enumerate(_completed_trials_sorted(study), start=1):
        attrs = trial.user_attrs
        params = trial.params
        rows.append(
            "\t".join(
                [
                    str(rank),
                    str(trial.number),
                    str(trial.value),
                    str(attrs.get("macro_em", "")),
                    str(attrs.get("macro_recall@2", "")),
                    str(attrs.get("macro_recall@5", "")),
                    str(params.get("top_k", "")),
                    str(params.get("ppr_qa_top_k", "")),
                    str(params.get("ppr_top_k", "")),
                    str(params.get("passage_node_weight", "")),
                    str(params.get("ppr_damping", "")),
                    str(params.get("hub_penalty_threshold", "")),
                ]
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _configs_from_study(args: argparse.Namespace, top_n: int) -> list[tuple[str, HPOConfig]]:
    study = _create_study(args)
    configs: list[tuple[str, HPOConfig]] = [("ppr_hybrid_syn_anchor", ANCHOR_CONFIG)]
    completed_trials = _completed_trials_sorted(study)
    if not completed_trials:
        raise RuntimeError("No completed Optuna trials found; run HPO_STAGE=dev first or pass --configs-file.")
    for trial in completed_trials:
        config = HPOConfig.from_params(trial.params)
        if config == ANCHOR_CONFIG:
            continue
        configs.append((f"trial_{trial.number:04d}", config))
        if len(configs) >= top_n + 1:
            break
    return configs


def _read_config_file(path: Path) -> list[tuple[str, HPOConfig]]:
    configs: list[tuple[str, HPOConfig]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) != 7:
            raise ValueError(f"{path}:{lineno}: expected 7 columns")
        name, top_k, ppr_qa_top_k, ppr_top_k, passage_node_weight, ppr_damping, hub = parts
        configs.append(
            (
                name,
                HPOConfig(
                    top_k=int(top_k),
                    ppr_qa_top_k=int(ppr_qa_top_k),
                    ppr_top_k=int(ppr_top_k),
                    passage_node_weight=float(passage_node_weight),
                    ppr_damping=float(ppr_damping),
                    hub_penalty_threshold=int(hub),
                ),
            )
        )
    return configs


def _write_config_file(path: Path, configs: Sequence[tuple[str, HPOConfig]]) -> None:
    lines = [
        "# name top_k ppr_qa_top_k ppr_top_k passage_node_weight ppr_damping hub_penalty_threshold"
    ]
    for name, config in configs:
        lines.append(
            f"{name} {config.top_k} {config.ppr_qa_top_k} {config.ppr_top_k} "
            f"{config.passage_node_weight} {config.ppr_damping} {config.hub_penalty_threshold}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_static_stage(args: argparse.Namespace) -> None:
    workspaces = _prepare_workspace_map(args)
    args.results_root.mkdir(parents=True, exist_ok=True)
    if args.configs_file:
        configs = _read_config_file(args.configs_file)
    else:
        configs = _configs_from_study(args, args.top_n)

    summaries: dict[str, dict[str, Any]] = {}
    for name, config in configs:
        dataset_metrics, macro = _run_config_across_datasets(
            args=args,
            workspaces=workspaces,
            config_name=name,
            config=config,
            n_samples=args.n_samples,
            seed=args.seed,
            output_root=args.results_root,
        )
        summaries[name] = {"config": config.to_params(), "dataset_metrics": dataset_metrics, **macro}

    ranked = sorted(summaries.items(), key=lambda item: item[1]["macro_f1"], reverse=True)
    _write_json(args.results_root / f"{args.stage}_summary.json", {"ranked": ranked})
    if args.stage == "verify":
        top_configs = [(name, HPOConfig.from_params(payload["config"])) for name, payload in ranked[:3]]
        _write_config_file(args.results_root / "top_configs.tsv", top_configs)


def _parse_args() -> argparse.Namespace:
    repo_root = _repo_root()
    p = argparse.ArgumentParser(description="Optuna HPO for MultiHopQA PPR + hybrid + synonym edges")
    p.add_argument("--stage", choices=["dev", "verify", "full"], default="dev")
    p.add_argument("--repo-root", type=Path, default=repo_root, dest="repo_root")
    p.add_argument("--data-dir", type=Path, default=repo_root / "evaluate_local/MultiHopQA/hipporag2_data")
    p.add_argument("--workspace-root", type=Path, default=repo_root / "evaluate_local/MultiHopQA/workspaces/multihopqa_hr2_v0")
    p.add_argument("--results-root", type=Path, default=None)
    p.add_argument("--study-db", type=Path, default=None)
    p.add_argument("--study-name", default="multihopqa_ppr_hybrid_synonym_hpo")
    p.add_argument("--synonym-threshold", type=float, default=0.8)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--configs-file", type=Path, default=None)
    p.add_argument("--concurrency", type=int, default=100)
    p.add_argument("--recall-k", type=int, nargs="+", default=[2, 5])
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--python-executable", default=sys.executable)
    p.add_argument("--evaluate-script", type=Path, default=repo_root / "evaluate_local/MultiHopQA/evaluate_multihop.py")
    p.add_argument("--optuna-seed", type=int, default=42)
    p.add_argument("--optuna-jobs", type=int, default=1)
    p.add_argument("--pruner", choices=["none", "median"], default="none")
    args = p.parse_args()

    if args.n_samples is None:
        args.n_samples = {"dev": 200, "verify": 300, "full": 1000}[args.stage]
    if args.seed is None:
        args.seed = {"dev": 42, "verify": 43, "full": 42}[args.stage]
    if args.results_root is None:
        suffix = f"syn_t{_threshold_label(args.synonym_threshold)}_{args.stage}"
        args.results_root = (
            args.repo_root
            / "evaluate_local"
            / "MultiHopQA"
            / "results"
            / f"multihopqa_hr2_v0_ppr_hpo_semantic_prompt_{suffix}"
        )
    if args.study_db is None:
        dev_suffix = f"syn_t{_threshold_label(args.synonym_threshold)}_dev"
        dev_root = (
            args.repo_root
            / "evaluate_local"
            / "MultiHopQA"
            / "results"
            / f"multihopqa_hr2_v0_ppr_hpo_semantic_prompt_{dev_suffix}"
        )
        args.study_db = (args.results_root if args.stage == "dev" else dev_root) / "study.db"
    if args.n_samples < 0:
        raise SystemExit("--n-samples must be >= 0")
    if args.n_trials <= 0:
        raise SystemExit("--n-trials must be > 0")
    if args.top_n <= 0:
        raise SystemExit("--top-n must be > 0")
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be > 0")
    if args.optuna_jobs <= 0:
        raise SystemExit("--optuna-jobs must be > 0")
    args.repo_root = args.repo_root.resolve()
    args.data_dir = args.data_dir.resolve()
    args.workspace_root = args.workspace_root.resolve()
    args.results_root = args.results_root.resolve()
    args.study_db = args.study_db.resolve()
    args.evaluate_script = args.evaluate_script.resolve()
    return args


def main() -> None:
    args = _parse_args()
    print("[hpo] PPR + hybrid retrieval + synonym edges")
    print(f"[hpo] stage={args.stage} n_samples={args.n_samples} seed={args.seed}")
    print(f"[hpo] results_root={args.results_root}")
    print(f"[hpo] study_db={args.study_db}")
    print(f"[hpo] synonym_threshold={args.synonym_threshold}")
    if args.stage == "dev":
        _run_dev(args)
    else:
        _run_static_stage(args)


if __name__ == "__main__":
    main()
