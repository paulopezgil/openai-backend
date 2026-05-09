from typing import Optional

from pydantic import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
# GET /health — OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: Optional[str] = None
    model: Optional[str] = None