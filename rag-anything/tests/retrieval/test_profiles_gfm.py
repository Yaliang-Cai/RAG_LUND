class TestGFMMultihopProfile:
    def test_gfm_in_known_paths(self):
        from raganything.retrieval.profiles import KNOWN_PATHS
        assert "gfm" in KNOWN_PATHS

    def test_gfm_multihop_profile_exists(self):
        from raganything.retrieval.profiles import PROFILE_REGISTRY
        assert "gfm_multihop" in PROFILE_REGISTRY

    def test_gfm_multihop_default_excludes_gfm(self):
        from raganything.retrieval.profiles import PROFILE_REGISTRY
        profile = PROFILE_REGISTRY["gfm_multihop"]
        # GFM is commented out by default
        assert "gfm" not in profile.paths
        assert "ppr" in profile.paths
        assert "hybrid" not in profile.paths

    def test_gfm_multihop_rrf_weights_match_paths(self):
        from raganything.retrieval.profiles import PROFILE_REGISTRY
        profile = PROFILE_REGISTRY["gfm_multihop"]
        assert set(profile.paths) == set(profile.rrf_weights.keys())
