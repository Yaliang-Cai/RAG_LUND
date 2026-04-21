#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parent.parent
if _project_root.exists():
    sys.path.insert(0, str(_project_root))

from raganything.constants import (
    DEFAULT_ENABLE_ENTITY_DISAMBIGUATION,
    DEFAULT_ENABLE_MULTI_HOP,
    DEFAULT_ENABLE_SYNONYM_LINKING,
    DEFAULT_MULTI_HOP_DEPTH,
    DEFAULT_PASSAGE_NODE_WEIGHT,
    DEFAULT_PPR_DAMPING,
    DEFAULT_PPR_TOP_K,
)

INDEX_PROFILE_FILE = ".ablation_index_profile.json"

_TRUE_SET = {"1", "true", "yes", "y", "on"}
_FALSE_SET = {"0", "false", "no", "n", "off"}


def as_bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    token = str(v).strip().lower()
    if token in _TRUE_SET:
        return True
    if token in _FALSE_SET:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {v}")


def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return as_bool(v)
    if isinstance(v, (int, float)):
        numeric = float(v)
        if numeric in (0.0, 1.0):
            return bool(int(numeric))
        raise ValueError(f"invalid bool numeric value: {v!r}")
    raise ValueError(f"invalid bool value: {v!r}")


@dataclass(frozen=True)
class AblationFlags:
    enable_entity_disambiguation: bool = DEFAULT_ENABLE_ENTITY_DISAMBIGUATION
    enable_synonym_linking: bool = DEFAULT_ENABLE_SYNONYM_LINKING
    enable_multi_hop: bool = DEFAULT_ENABLE_MULTI_HOP
    multi_hop_depth: int = DEFAULT_MULTI_HOP_DEPTH
    ppr_damping: float = DEFAULT_PPR_DAMPING
    ppr_top_k: int = DEFAULT_PPR_TOP_K
    passage_node_weight: float = DEFAULT_PASSAGE_NODE_WEIGHT

    @classmethod
    def from_mapping(cls, payload: Any) -> "AblationFlags | None":
        if not isinstance(payload, dict):
            return None
        required = {
            "enable_entity_disambiguation",
            "enable_synonym_linking",
            "enable_multi_hop",
            "multi_hop_depth",
            "ppr_damping",
            "ppr_top_k",
            "passage_node_weight",
        }
        if not required.issubset(payload.keys()):
            return None
        try:
            return cls(
                enable_entity_disambiguation=_coerce_bool(
                    payload["enable_entity_disambiguation"]
                ),
                enable_synonym_linking=_coerce_bool(payload["enable_synonym_linking"]),
                enable_multi_hop=_coerce_bool(payload["enable_multi_hop"]),
                multi_hop_depth=int(payload["multi_hop_depth"]),
                ppr_damping=float(payload["ppr_damping"]),
                ppr_top_k=int(payload["ppr_top_k"]),
                passage_node_weight=float(payload["passage_node_weight"]),
            )
        except Exception:
            return None

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "AblationFlags":
        return cls(
            enable_entity_disambiguation=_coerce_bool(args.enable_entity_disambiguation),
            enable_synonym_linking=_coerce_bool(args.enable_synonym_linking),
            enable_multi_hop=_coerce_bool(args.enable_multi_hop),
            multi_hop_depth=int(args.multi_hop_depth),
            ppr_damping=float(args.ppr_damping),
            ppr_top_k=int(args.ppr_top_k),
            passage_node_weight=float(args.passage_node_weight),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enable_entity_disambiguation": self.enable_entity_disambiguation,
            "enable_synonym_linking": self.enable_synonym_linking,
            "enable_multi_hop": self.enable_multi_hop,
            "multi_hop_depth": self.multi_hop_depth,
            "ppr_damping": self.ppr_damping,
            "ppr_top_k": self.ppr_top_k,
            "passage_node_weight": self.passage_node_weight,
        }

    def to_index_profile(self) -> dict[str, Any]:
        # V1/V2 affect index materialization. V3 is query-time only.
        return {
            "enable_entity_disambiguation": self.enable_entity_disambiguation,
            "enable_synonym_linking": self.enable_synonym_linking,
        }

    def to_query_kwargs(self) -> dict[str, Any]:
        return {
            "enable_multi_hop": self.enable_multi_hop,
            "multi_hop_depth": self.multi_hop_depth,
            "ppr_damping": self.ppr_damping,
            "ppr_top_k": self.ppr_top_k,
            "passage_node_weight": self.passage_node_weight,
        }

    def ablation_group(self) -> str:
        if not self.enable_entity_disambiguation and not self.enable_synonym_linking and not self.enable_multi_hop:
            return "DB-only"
        if self.enable_entity_disambiguation and not self.enable_synonym_linking and not self.enable_multi_hop:
            return "DB+V1"
        if self.enable_entity_disambiguation and self.enable_synonym_linking and not self.enable_multi_hop:
            return "DB+V1+V2"
        if self.enable_entity_disambiguation and self.enable_synonym_linking and self.enable_multi_hop:
            return "DB+V1+V2+V3"
        return "custom"

    def is_index_compatible_with(self, other: "AblationFlags") -> bool:
        return self.to_index_profile() == other.to_index_profile()


def add_ablation_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_legacy_underscore_alias: bool = True,
) -> None:
    parser.add_argument(
        "--enable-entity-disambiguation",
        *(
            ["--enable_entity_disambiguation"]
            if include_legacy_underscore_alias
            else []
        ),
        dest="enable_entity_disambiguation",
        type=as_bool,
        default=DEFAULT_ENABLE_ENTITY_DISAMBIGUATION,
    )
    parser.add_argument(
        "--enable-synonym-linking",
        *(["--enable_synonym_linking"] if include_legacy_underscore_alias else []),
        dest="enable_synonym_linking",
        type=as_bool,
        default=DEFAULT_ENABLE_SYNONYM_LINKING,
    )
    parser.add_argument(
        "--enable-multi-hop",
        *(["--enable_multi_hop"] if include_legacy_underscore_alias else []),
        dest="enable_multi_hop",
        type=as_bool,
        default=DEFAULT_ENABLE_MULTI_HOP,
    )
    parser.add_argument(
        "--multi-hop-depth",
        *(["--multi_hop_depth"] if include_legacy_underscore_alias else []),
        dest="multi_hop_depth",
        type=int,
        default=DEFAULT_MULTI_HOP_DEPTH,
    )
    parser.add_argument(
        "--ppr-damping",
        *(["--ppr_damping"] if include_legacy_underscore_alias else []),
        dest="ppr_damping",
        type=float,
        default=DEFAULT_PPR_DAMPING,
    )
    parser.add_argument(
        "--ppr-top-k",
        *(["--ppr_top_k"] if include_legacy_underscore_alias else []),
        dest="ppr_top_k",
        type=int,
        default=DEFAULT_PPR_TOP_K,
    )
    parser.add_argument(
        "--passage-node-weight",
        *(["--passage_node_weight"] if include_legacy_underscore_alias else []),
        dest="passage_node_weight",
        type=float,
        default=DEFAULT_PASSAGE_NODE_WEIGHT,
    )


def validate_ablation_flags(
    args: argparse.Namespace,
    *,
    naming_style: str = "hyphen",
) -> AblationFlags:
    flags = AblationFlags.from_namespace(args)
    if flags.multi_hop_depth <= 0:
        raise ValueError(
            f"{_flag_name('multi-hop-depth', naming_style)} must be > 0, got {flags.multi_hop_depth}"
        )
    if flags.ppr_top_k <= 0:
        raise ValueError(
            f"{_flag_name('ppr-top-k', naming_style)} must be > 0, got {flags.ppr_top_k}"
        )
    if not (0.0 < flags.ppr_damping < 1.0):
        raise ValueError(
            f"{_flag_name('ppr-damping', naming_style)} must be in (0,1), got {flags.ppr_damping}"
        )
    if flags.passage_node_weight < 0:
        raise ValueError(
            f"{_flag_name('passage-node-weight', naming_style)} must be >= 0, got {flags.passage_node_weight}"
        )
    if flags.enable_synonym_linking and not flags.enable_entity_disambiguation:
        raise ValueError(
            f"{_flag_name('enable-synonym-linking', naming_style)} requires "
            f"{_flag_name('enable-entity-disambiguation', naming_style)}"
        )
    if flags.enable_multi_hop and (
        not flags.enable_entity_disambiguation or not flags.enable_synonym_linking
    ):
        raise ValueError(
            f"{_flag_name('enable-multi-hop', naming_style)} requires "
            f"{_flag_name('enable-entity-disambiguation', naming_style)} and "
            f"{_flag_name('enable-synonym-linking', naming_style)}"
        )
    return flags


def apply_ablation_flags_to_settings(settings: Any, flags: AblationFlags) -> None:
    settings.enable_entity_disambiguation = flags.enable_entity_disambiguation
    settings.enable_synonym_linking = flags.enable_synonym_linking
    settings.enable_multi_hop = flags.enable_multi_hop
    settings.multi_hop_depth = flags.multi_hop_depth
    settings.ppr_damping = flags.ppr_damping
    settings.ppr_top_k = flags.ppr_top_k
    settings.passage_node_weight = flags.passage_node_weight


def validate_workspace_env_isolation(*, workspace_id: str) -> None:
    normalized_workspace = str(workspace_id or "").strip()
    if not normalized_workspace:
        return

    conflicts: list[str] = []
    for env_name in ("NEO4J_WORKSPACE", "QDRANT_WORKSPACE"):
        env_value_raw = os.getenv(env_name)
        if env_value_raw is None:
            continue
        env_value = str(env_value_raw).strip()
        if not env_value:
            continue
        if env_value != normalized_workspace:
            conflicts.append(
                f"{env_name}={env_value!r} conflicts with workspace_id={normalized_workspace!r}"
            )

    if conflicts:
        raise ValueError(
            "Workspace isolation check failed. "
            "Unset conflicting NEO4J_WORKSPACE/QDRANT_WORKSPACE overrides or make them "
            f"equal to workspace_id. Details: {'; '.join(conflicts)}"
        )


def _flag_name(flag: str, naming_style: str) -> str:
    style = naming_style.strip().lower()
    if style == "underscore":
        return f"--{flag.replace('-', '_')}"
    return f"--{flag}"


def build_index_profile(flags: AblationFlags, *, settings: Any | None = None) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "profile_version": 1,
        "enable_entity_disambiguation": bool(flags.enable_entity_disambiguation),
        "enable_synonym_linking": bool(flags.enable_synonym_linking),
    }
    if settings is not None:
        if hasattr(settings, "synonymy_threshold"):
            profile["synonymy_threshold"] = float(getattr(settings, "synonymy_threshold"))
        if hasattr(settings, "synonymy_topk"):
            profile["synonymy_topk"] = int(getattr(settings, "synonymy_topk"))
        if hasattr(settings, "synonymy_min_entity_len"):
            profile["synonymy_min_entity_len"] = int(
                getattr(settings, "synonymy_min_entity_len")
            )
    return profile


def build_query_profile(flags: AblationFlags) -> dict[str, Any]:
    profile = dict(flags.to_query_kwargs())
    profile["profile_version"] = 1
    return profile


def ensure_workspace_index_profile(
    *,
    working_dir_root: str | Path,
    workspace_id: str,
    index_profile: dict[str, Any],
    allow_legacy_adoption: bool = False,
) -> dict[str, Any]:
    workspace_dir = Path(working_dir_root) / str(workspace_id)
    profile_path = workspace_dir / INDEX_PROFILE_FILE
    current_profile = _normalize_profile(index_profile)

    existing_profile = _load_json(profile_path)
    if isinstance(existing_profile, dict):
        existing_normalized = _normalize_profile(existing_profile)
        if existing_normalized != current_profile:
            detail = _profile_diff(existing_normalized, current_profile)
            raise ValueError(
                "Workspace ablation index profile mismatch. "
                "Use a new workspace_id for DB-only/DB+V1/DB+V1+V2 isolation or rebuild the workspace. "
                f"Details: {detail}"
            )
        return existing_normalized

    if _workspace_has_artifacts(workspace_dir) and not allow_legacy_adoption:
        raise ValueError(
            "Workspace contains existing index artifacts but has no ablation profile file. "
            "Refusing to continue to avoid mixed ablation states. "
            "Use a new workspace_id (recommended) or rebuild/clean this workspace."
        )

    workspace_dir.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        json.dumps(current_profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return current_profile


def _workspace_has_artifacts(workspace_dir: Path) -> bool:
    if not workspace_dir.exists():
        return False
    for entry in workspace_dir.rglob("*"):
        if not entry.is_file():
            continue
        if entry.name == INDEX_PROFILE_FILE:
            continue
        return True
    return False


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _normalize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "profile_version": int(profile.get("profile_version", 1)),
        "enable_entity_disambiguation": _coerce_bool(
            profile.get("enable_entity_disambiguation", True)
        ),
        "enable_synonym_linking": _coerce_bool(
            profile.get("enable_synonym_linking", False)
        ),
    }
    if "synonymy_threshold" in profile:
        normalized["synonymy_threshold"] = float(profile["synonymy_threshold"])
    if "synonymy_topk" in profile:
        normalized["synonymy_topk"] = int(profile["synonymy_topk"])
    if "synonymy_min_entity_len" in profile:
        normalized["synonymy_min_entity_len"] = int(profile["synonymy_min_entity_len"])
    return normalized


def _profile_diff(existing: dict[str, Any], current: dict[str, Any]) -> str:
    keys = sorted(set(existing.keys()) | set(current.keys()))
    diffs: list[str] = []
    for key in keys:
        if existing.get(key) != current.get(key):
            diffs.append(f"{key}: existing={existing.get(key)!r}, current={current.get(key)!r}")
    return "; ".join(diffs) if diffs else "unknown"
