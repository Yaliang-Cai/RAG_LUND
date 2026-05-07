"""Governance layer settings loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass

from raganything.constants import (
    DEFAULT_PG_DSN,
    DEFAULT_PG_POOL_MIN,
    DEFAULT_PG_POOL_MAX,
    DEFAULT_PG_COMMAND_TIMEOUT_SECONDS,
    DEFAULT_JOB_MAX_CONCURRENT,
    DEFAULT_JOB_PROGRESS_INTERVAL_SECONDS,
    DEFAULT_JOB_PROGRESS_CHUNK_INTERVAL,
    DEFAULT_JOB_SHUTDOWN_GRACE_SECONDS,
)


@dataclass(frozen=True)
class GovernanceSettings:
    pg_dsn: str = DEFAULT_PG_DSN
    pg_pool_min: int = DEFAULT_PG_POOL_MIN
    pg_pool_max: int = DEFAULT_PG_POOL_MAX
    pg_command_timeout: int = DEFAULT_PG_COMMAND_TIMEOUT_SECONDS
    job_max_concurrent: int = DEFAULT_JOB_MAX_CONCURRENT
    job_progress_interval: int = DEFAULT_JOB_PROGRESS_INTERVAL_SECONDS
    job_progress_chunk_interval: int = DEFAULT_JOB_PROGRESS_CHUNK_INTERVAL
    job_shutdown_grace: int = DEFAULT_JOB_SHUTDOWN_GRACE_SECONDS

    @classmethod
    def from_env(cls) -> "GovernanceSettings":
        return cls(
            pg_dsn=os.getenv("RAGANYTHING_PG_DSN", DEFAULT_PG_DSN),
            pg_pool_min=int(os.getenv("RAGANYTHING_PG_POOL_MIN", DEFAULT_PG_POOL_MIN)),
            pg_pool_max=int(os.getenv("RAGANYTHING_PG_POOL_MAX", DEFAULT_PG_POOL_MAX)),
            pg_command_timeout=int(os.getenv(
                "RAGANYTHING_PG_COMMAND_TIMEOUT", DEFAULT_PG_COMMAND_TIMEOUT_SECONDS
            )),
            job_max_concurrent=int(os.getenv(
                "RAGANYTHING_JOB_MAX_CONCURRENT", DEFAULT_JOB_MAX_CONCURRENT
            )),
            job_progress_interval=int(os.getenv(
                "RAGANYTHING_JOB_PROGRESS_INTERVAL", DEFAULT_JOB_PROGRESS_INTERVAL_SECONDS
            )),
            job_progress_chunk_interval=int(os.getenv(
                "RAGANYTHING_JOB_PROGRESS_CHUNK_INTERVAL", DEFAULT_JOB_PROGRESS_CHUNK_INTERVAL
            )),
            job_shutdown_grace=int(os.getenv(
                "RAGANYTHING_JOB_SHUTDOWN_GRACE", DEFAULT_JOB_SHUTDOWN_GRACE_SECONDS
            )),
        )
