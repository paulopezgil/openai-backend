from typing import Optional, Union

from pydantic import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
# POST /v1/embeddings — INPUT
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, list[str], list[int], list[list[int]]]
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# POST /v1/embeddings — OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingObject(BaseModel):
    object: str = "embedding"
    index: int
    embedding: Union[list[float], str]


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingObject]
    model: str
    usage: EmbeddingUsage