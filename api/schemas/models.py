from typing import Literal

from pydantic import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
# GET /v1/models
# ─────────────────────────────────────────────────────────────────────────────


class Model(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[Model]