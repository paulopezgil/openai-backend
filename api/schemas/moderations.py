from typing import Optional, Union

from pydantic import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
# POST /v1/moderations — INPUT
# ─────────────────────────────────────────────────────────────────────────────

class ModerationRequest(BaseModel):
    input: Union[str, list[str]]
    model: Optional[str] = "omni-moderation-latest"


# ─────────────────────────────────────────────────────────────────────────────
# POST /v1/moderations — OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

class ModerationCategories(BaseModel):
    hate: bool = False
    hate_threatening: bool = False
    harassment: bool = False
    harassment_threatening: bool = False
    self_harm: bool = False
    self_harm_intent: bool = False
    self_harm_instructions: bool = False
    sexual: bool = False
    sexual_minors: bool = False
    violence: bool = False
    violence_graphic: bool = False


class ModerationResult(BaseModel):
    flagged: bool
    categories: ModerationCategories
    category_scores: dict[str, float]


class ModerationResponse(BaseModel):
    id: str
    model: str
    results: list[ModerationResult]