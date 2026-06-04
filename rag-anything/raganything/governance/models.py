"""Pydantic schemas for the governance API surface."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field

DocumentStatus = Literal[
    "pending", "parsing", "indexing", "done", "failed", "deleting", "deleted",
]
JobStatus = Literal["queued", "running", "done", "failed", "crashed"]


class WorkspaceRow(BaseModel):
    workspace_id: str
    frozen: bool
    owner: Optional[str] = None
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentRow(BaseModel):
    doc_id: UUID
    workspace_id: str
    filename: str
    file_hash: str
    size_bytes: int
    status: DocumentStatus
    ingested_at: datetime
    finished_at: Optional[datetime] = None
    error: Optional[str] = None


class JobRow(BaseModel):
    job_id: UUID
    workspace_id: str
    doc_ids: list[UUID] = Field(default_factory=list)
    status: JobStatus
    progress: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    started_at: datetime
    finished_at: Optional[datetime] = None


class AuditRow(BaseModel):
    id: int
    workspace_id: str
    doc_id: Optional[UUID] = None
    action: str
    actor: Optional[str] = None
    timestamp: datetime
    details: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    job_id: Optional[UUID]
    doc_id: UUID
    status: str
    duplicate: bool
