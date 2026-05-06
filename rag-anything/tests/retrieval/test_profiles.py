import pytest

from raganything.retrieval.profiles import (
    RetrievalProfile,
    PROFILE_REGISTRY,
    KNOWN_PATHS,
)


def test_all_builtin_profiles_present():
    assert set(PROFILE_REGISTRY.keys()) == {"precise", "semantic", "local", "multihop", "full", "gfm_multihop"}


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
    for name in ("precise", "local", "multihop", "semantic"):
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


def test_profile_raises_on_unknown_path():
    with pytest.raises(ValueError, match="unknown path names"):
        RetrievalProfile(
            name="bad",
            description="test",
            paths=["nonexistent_backend"],
            rrf_weights={"nonexistent_backend": 1.0},
        )
