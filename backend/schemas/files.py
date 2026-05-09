from typing import Optional

from pydantic import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
# POST /v1/files — INPUT (multipart/form-data)
# ─────────────────────────────────────────────────────────────────────────────

# File upload uses FastAPI UploadFile + Form — no Pydantic request body needed.


# ─────────────────────────────────────────────────────────────────────────────
# GET /v1/files — OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

class FileObject(BaseModel):
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str
    status: Optional[str] = "processed"
    status_details: Optional[str] = None


class FileListResponse(BaseModel):
    object: str = "list"
    data: list[FileObject]
    has_more: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# DELETE /v1/files/{file_id} — OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

class FileDeleteResponse(BaseModel):
    id: str
    object: str = "file"
    deleted: bool