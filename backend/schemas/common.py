from typing import Optional

from pydantic import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
# Error Schemas (Shared)
# ─────────────────────────────────────────────────────────────────────────────


class OpenAIError(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class OpenAIErrorResponse(BaseModel):
    error: OpenAIError