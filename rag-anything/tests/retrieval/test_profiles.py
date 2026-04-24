from raganything.retrieval.profiles import (
    RetrievalProfile,
    PROFILE_REGISTRY,
    KNOWN_PATHS,
)


def test_all_builtin_profiles_present():
    assert set(PROFILE_REGISTRY.keys()) == {"precise", "local", "multihop", "descriptive", "full"}


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


def test_full_profile_has_semaphore():
    assert PROFILE_REGISTRY["full"].max_concurrent_paths == 3


def test_simple_profiles_have_no_semaphore():
    for name in ("precise", "local", "multihop", "descriptive"):
        assert PROFILE_REGISTRY[name].max_concurrent_paths is None


def test_descriptive_path_overrides():
    profile = PROFILE_REGISTRY["descriptive"]
    assert "mix" in profile.path_overrides
    assert profile.path_overrides["mix"]["kg_chunk_selection_source"] == "untruncated"


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
    assert p.rerank_candidate_cap == 60
    assert p.max_concurrent_paths is None
    assert p.path_overrides == {}
