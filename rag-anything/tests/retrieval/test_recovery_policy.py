from raganything.retrieval.recovery_policy import RecoveryAction, RecoveryPolicy


def test_missing_relation_escalates_to_multihop_then_full_v2_then_decompose():
    policy = RecoveryPolicy()
    action = policy.choose(
        failure_type="missing_relation",
        original_profile="semantic",
        original_query="How is A connected to B?",
        tried_profiles={"semantic"},
        tried_signatures=set(),
    )
    assert action == RecoveryAction("retrieve", "multihop", "original")

    action = policy.choose(
        failure_type="missing_relation",
        original_profile="semantic",
        original_query="How is A connected to B?",
        tried_profiles={"semantic", "multihop"},
        tried_signatures={("retrieve", "multihop", "original")},
    )
    assert action == RecoveryAction("retrieve", "full_v2", "original")

    action = policy.choose(
        failure_type="missing_relation",
        original_profile="semantic",
        original_query="How is A connected to B?",
        tried_profiles={"semantic", "multihop", "full_v2"},
        tried_signatures={
            ("retrieve", "multihop", "original"),
            ("retrieve", "full_v2", "original"),
        },
    )
    assert action == RecoveryAction("decompose", "full_v2", "decompose")


def test_empty_prefers_rewrite_then_full_v2_without_repeating_actions():
    policy = RecoveryPolicy()
    action = policy.choose(
        failure_type="empty",
        original_profile="semantic",
        original_query="plain question",
        tried_profiles={"semantic"},
        tried_signatures=set(),
    )
    assert action == RecoveryAction("rewrite", "semantic", "rewrite")

    action = policy.choose(
        failure_type="empty",
        original_profile="semantic",
        original_query="plain question",
        tried_profiles={"semantic"},
        tried_signatures={("rewrite", "semantic", "rewrite")},
    )
    assert action == RecoveryAction("retrieve", "full_v2", "original")


def test_unanswerable_candidate_does_not_terminate_before_recovery():
    policy = RecoveryPolicy()
    action = policy.choose(
        failure_type="unanswerable_candidate",
        original_profile="local",
        original_query="question",
        tried_profiles={"local"},
        tried_signatures=set(),
    )
    assert action == RecoveryAction("retrieve", "full_v2", "original")

