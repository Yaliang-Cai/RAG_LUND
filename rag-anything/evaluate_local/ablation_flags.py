#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

DEFAULT_ENABLE_ENTITY_DISAMBIGUATION = True
DEFAULT_ENABLE_SYNONYM_LINKING = False
DEFAULT_ENABLE_MULTI_HOP = False
DEFAULT_MULTI_HOP_DEPTH = 2
DEFAULT_PPR_DAMPING = 0.5
DEFAULT_PPR_TOP_K = 50
DEFAULT_PASSAGE_NODE_WEIGHT = 0.05

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
    def from_namespace(cls, args: argparse.Namespace) -> "AblationFlags":
        return cls(
            enable_entity_disambiguation=bool(args.enable_entity_disambiguation),
            enable_synonym_linking=bool(args.enable_synonym_linking),
            enable_multi_hop=bool(args.enable_multi_hop),
            multi_hop_depth=int(args.multi_hop_depth),
            ppr_damping=float(args.ppr_damping),
            ppr_top_k=int(args.ppr_top_k),
            passage_node_weight=float(args.passage_node_weight),
        )

    def to_query_kwargs(self) -> dict[str, Any]:
        return {
            "enable_multi_hop": self.enable_multi_hop,
            "multi_hop_depth": self.multi_hop_depth,
            "ppr_damping": self.ppr_damping,
            "ppr_top_k": self.ppr_top_k,
            "passage_node_weight": self.passage_node_weight,
        }


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
    return flags


def apply_ablation_flags_to_settings(settings: Any, flags: AblationFlags) -> None:
    settings.enable_entity_disambiguation = flags.enable_entity_disambiguation
    settings.enable_synonym_linking = flags.enable_synonym_linking
    settings.enable_multi_hop = flags.enable_multi_hop
    settings.multi_hop_depth = flags.multi_hop_depth
    settings.ppr_damping = flags.ppr_damping
    settings.ppr_top_k = flags.ppr_top_k
    settings.passage_node_weight = flags.passage_node_weight


def _flag_name(flag: str, naming_style: str) -> str:
    style = naming_style.strip().lower()
    if style == "underscore":
        return f"--{flag.replace('-', '_')}"
    return f"--{flag}"
