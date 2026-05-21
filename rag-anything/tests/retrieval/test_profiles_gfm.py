class TestGFMPath:
    """GFM path remains a known retrieval path even though no built-in
    profile currently uses it (the gfm_multihop profile was removed in
    the 2026-05-21 profile registry cleanup). Keeping the path name in
    KNOWN_PATHS lets ad-hoc profile configs still reference it."""

    def test_gfm_in_known_paths(self):
        from raganything.retrieval.profiles import KNOWN_PATHS
        assert "gfm" in KNOWN_PATHS

    def test_gfm_multihop_profile_removed(self):
        from raganything.retrieval.profiles import PROFILE_REGISTRY
        assert "gfm_multihop" not in PROFILE_REGISTRY
