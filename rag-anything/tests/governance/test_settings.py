import os
from unittest.mock import patch

from raganything.governance.settings import GovernanceSettings


def test_defaults_loaded_when_env_unset():
    with patch.dict(os.environ, {}, clear=False):
        for key in list(os.environ):
            if key.startswith("RAGANYTHING_PG_") or key.startswith("RAGANYTHING_JOB_"):
                os.environ.pop(key, None)
        s = GovernanceSettings.from_env()
        assert s.pg_dsn == "postgresql://localhost:5432/raganything"
        assert s.pg_pool_min == 2
        assert s.pg_pool_max == 10
        assert s.job_max_concurrent == 1


def test_env_overrides_applied():
    with patch.dict(os.environ, {
        "RAGANYTHING_PG_DSN": "postgresql://test:5433/db",
        "RAGANYTHING_PG_POOL_MAX": "20",
        "RAGANYTHING_JOB_MAX_CONCURRENT": "4",
    }):
        s = GovernanceSettings.from_env()
        assert s.pg_dsn == "postgresql://test:5433/db"
        assert s.pg_pool_max == 20
        assert s.job_max_concurrent == 4
