import pytest

from raganything.retrieval.profiles import (
    RetrievalProfile,
    PROFILE_REGISTRY,
    KNOWN_PATHS,
)


def test_all_builtin_profiles_present():
    assert set(PROFILE_REGISTRY.keys()) == {
        "precise",
        "semantic",
        "local",
        "global",
        "multihop",
        "full",
        "full_v2",
        "hybrid_experimental",
        "gfm_multihop",
    }


def test_profile_paths_match_router_design():
    assert PROFILE_REGISTRY["precise"].paths == ["qdrant_sparse"]
    assert PROFILE_REGISTRY["semantic"].paths == ["qdrant_chunks_hybrid"]
    assert PROFILE_REGISTRY["local"].paths == ["local_kg"]
    assert PROFILE_REGISTRY["global"].paths == ["global_kg"]
    assert PROFILE_REGISTRY["hybrid_experimental"].paths == ["hybrid"]


def test_multihop_profile_protects_ppr():
    profile = PROFILE_REGISTRY["multihop"]
    assert profile.paths == ["ppr"]
    assert profile.rrf_weights == {"ppr": 1.0}
    assert profile.enable_rerank is False
    assert profile.min_rrf_score == 0.0
    assert "hybrid" not in profile.paths


def test_full_profile_is_failure_recovery_mix_without_hybrid():
    profile = PROFILE_REGISTRY["full"]
    assert profile.paths == ["ppr", "qdrant_chunks_hybrid", "qdrant_sparse"]
    assert profile.rrf_weights == {
        "ppr": 1.3,
        "qdrant_chunks_hybrid": 0.8,
        "qdrant_sparse": 0.5,
    }
    assert profile.enable_rerank is False
    assert "hybrid" not in profile.paths


def test_full_v2_profile_prefers_ppr_more_strongly():
    profile = PROFILE_REGISTRY["full_v2"]
    assert profile.paths == ["ppr", "qdrant_chunks_hybrid", "qdrant_sparse"]
    assert profile.rrf_weights == {
        "ppr": 2.0,
        "qdrant_chunks_hybrid": 0.7,
        "qdrant_sparse": 0.4,
    }
    assert profile.enable_rerank is False
    assert profile.min_rrf_score == 0.0
    assert profile.path_overrides["ppr"]["exclude_synonym_edges"] is False
    assert profile.path_overrides["qdrant_chunks_hybrid"]["exclude_synonym_edges"] is True
    assert profile.path_overrides["qdrant_sparse"]["exclude_synonym_edges"] is True
    assert profile.path_floors == {"ppr": 3}


def test_non_sparse_profiles_use_qdrant_hybrid_retrieval():
    expectations = {
        "semantic": ("qdrant_chunks_hybrid", "hybrid"),
        "local": ("local_kg", "hybrid"),
        "global": ("global_kg", "hybrid"),
        "multihop": ("ppr", "hybrid"),
        "hybrid_experimental": ("hybrid", "hybrid"),
    }
    for profile_name, (path_name, mode) in expectations.items():
        profile = PROFILE_REGISTRY[profile_name]
        assert profile.path_overrides[path_name]["qdrant_retrieval_mode"] == mode


def test_only_ppr_profiles_allow_synonym_edges():
    assert PROFILE_REGISTRY["multihop"].path_overrides["ppr"]["exclude_synonym_edges"] is False
    assert PROFILE_REGISTRY["gfm_multihop"].path_overrides["ppr"]["exclude_synonym_edges"] is False
    assert PROFILE_REGISTRY["full"].path_overrides["ppr"]["exclude_synonym_edges"] is False
    assert PROFILE_REGISTRY["full_v2"].path_overrides["ppr"]["exclude_synonym_edges"] is False

    non_ppr = [
        ("precise", "qdrant_sparse"),
        ("semantic", "qdrant_chunks_hybrid"),
        ("local", "local_kg"),
        ("global", "global_kg"),
        ("hybrid_experimental", "hybrid"),
    ]
    for profile_name, path_name in non_ppr:
        assert PROFILE_REGISTRY[profile_name].path_overrides[path_name]["exclude_synonym_edges"] is True


def test_profile_paths_are_known():
    for profile in PROFILE_REGISTRY.values():
        for path in profile.paths:
            assert path in KNOWN_PATHS, f"Profile '{profile.name}' uses unknown path '{path}'"


def test_profile_rrf_weights_cover_all_paths():
    for profile in PROFILE_REGISTRY.values():
        for path in profile.paths:
            assert path in profile.rrf_weights, (
                f"Profile '{profile.name}' path '{path}' missing from rrf_weights"
            )


def test_full_profile_has_no_semaphore():
    assert PROFILE_REGISTRY["full"].max_concurrent_paths is None


def test_simple_profiles_have_no_semaphore():
    for name in ("precise", "local", "global", "multihop", "semantic"):
        assert PROFILE_REGISTRY[name].max_concurrent_paths is None


def test_profile_defaults():
    p = RetrievalProfile(
        name="test",
        description="test",
        paths=["naive"],
        rrf_weights={"naive": 1.0},
    )
    assert p.rrf_k == 60
    assert p.enable_rerank is True
    assert p.min_rerank_score == 0.3
    assert p.rerank_candidate_cap == 30
    assert p.min_rrf_score == 0.01
    assert p.max_concurrent_paths is None
    assert p.path_overrides == {}
    assert p.path_floors == {}


def test_profile_raises_on_missing_rrf_weight():
    with pytest.raises(ValueError, match="missing from rrf_weights"):
        RetrievalProfile(
            name="bad",
            description="test",
            paths=["naive", "hybrid"],
            rrf_weights={"naive": 1.0},  # missing "hybrid"
        )


def test_profile_raises_on_extra_rrf_weight():
    with pytest.raises(ValueError, match="rrf_weights keys not in paths"):
        RetrievalProfile(
            name="bad",
            description="test",
            paths=["naive"],
            rrf_weights={"naive": 1.0, "hybrid": 0.5},  # "hybrid" not in paths
        )


def test_profile_raises_on_extra_path_floor():
    with pytest.raises(ValueError, match="path_floors keys not in paths"):
        RetrievalProfile(
            name="bad",
            description="test",
            paths=["naive"],
            rrf_weights={"naive": 1.0},
            path_floors={"ppr": 3},
        )


def test_profile_raises_on_negative_path_floor():
    with pytest.raises(ValueError, match="path_floors must be non-negative"):
        RetrievalProfile(
            name="bad",
            description="test",
            paths=["naive"],
            rrf_weights={"naive": 1.0},
            path_floors={"naive": -1},
        )


def test_profile_raises_on_unknown_path():
    with pytest.raises(ValueError, match="unknown path names"):
        RetrievalProfile(
            name="bad",
            description="test",
            paths=["nonexistent_backend"],
            rrf_weights={"nonexistent_backend": 1.0},
        )
