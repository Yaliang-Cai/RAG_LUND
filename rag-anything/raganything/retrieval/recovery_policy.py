from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RecoveryAction:
    action_type: str
    profile: str
    query_kind: str

    @property
    def signature(self) -> tuple[str, str, str]:
        return (self.action_type, self.profile, self.query_kind)


_MULTIHOP_TERMS = re.compile(
    r"\b(compare|comparison|connect|connected|bridge|between|both|shared|common|cause|caused|causal|"
    r"lead to|led to|influence|relationship between|how did|why did)\b",
    re.IGNORECASE,
)


class RecoveryPolicy:
    def choose(
        self,
        *,
        failure_type: str,
        original_profile: str,
        original_query: str,
        tried_profiles: set[str],
        tried_signatures: set[tuple[str, str, str]],
    ) -> RecoveryAction | None:
        candidates = self._candidates(
            failure_type=failure_type,
            original_profile=original_profile,
            original_query=original_query,
            tried_profiles=tried_profiles,
        )
        for action in candidates:
            if action.signature not in tried_signatures:
                return action
        for action in self._fallback_candidates(tried_profiles):
            if action.signature not in tried_signatures:
                return action
        return None

    def _candidates(
        self,
        *,
        failure_type: str,
        original_profile: str,
        original_query: str,
        tried_profiles: set[str],
    ) -> list[RecoveryAction]:
        rewrite_profile = "multihop" if _MULTIHOP_TERMS.search(original_query) else "semantic"
        if failure_type in {"missing_relation", "partial_evidence"}:
            actions: list[RecoveryAction] = []
            if "multihop" not in tried_profiles:
                actions.append(RecoveryAction("retrieve", "multihop", "original"))
            actions.extend([
                RecoveryAction("retrieve", "full_v2", "original"),
                RecoveryAction("decompose", "full_v2", "decompose"),
            ])
            return actions
        if failure_type == "needs_decomposition":
            return [
                RecoveryAction("decompose", "full_v2", "decompose"),
                RecoveryAction("retrieve", "full_v2", "original"),
            ]
        if failure_type in {"empty", "off_topic"}:
            return [
                RecoveryAction("rewrite", rewrite_profile, "rewrite"),
                RecoveryAction("retrieve", "full_v2", "original"),
                RecoveryAction("decompose", "full_v2", "decompose"),
            ]
        if failure_type == "missing_entity":
            return [
                RecoveryAction("rewrite", "semantic", "rewrite"),
                RecoveryAction("retrieve", "multihop", "original"),
                RecoveryAction("retrieve", "full_v2", "original"),
                RecoveryAction("decompose", "full_v2", "decompose"),
            ]
        if failure_type == "ambiguous_query":
            adjacent = _adjacent_profile(original_profile)
            return [
                RecoveryAction("rewrite", original_profile, "rewrite"),
                RecoveryAction("retrieve", adjacent, "original"),
                RecoveryAction("retrieve", "full_v2", "original"),
            ]
        if failure_type == "unanswerable_candidate":
            return [
                RecoveryAction("retrieve", "full_v2", "original"),
                RecoveryAction("decompose", "full_v2", "decompose"),
            ]
        return [
            RecoveryAction("retrieve", "full_v2", "original"),
            RecoveryAction("decompose", "full_v2", "decompose"),
        ]

    def _fallback_candidates(self, tried_profiles: set[str]) -> list[RecoveryAction]:
        actions = []
        if "full_v2" not in tried_profiles:
            actions.append(RecoveryAction("retrieve", "full_v2", "original"))
        if "multihop" not in tried_profiles:
            actions.append(RecoveryAction("retrieve", "multihop", "original"))
        actions.append(RecoveryAction("decompose", "full_v2", "decompose"))
        return actions


def _adjacent_profile(profile: str) -> str:
    if profile in {"local", "global", "semantic"}:
        return "multihop"
    if profile == "multihop":
        return "full_v2"
    return "semantic"
