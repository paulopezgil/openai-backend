from typing import Any, Optional, Union

from pydantic import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
# POST /v1/completions — INPUT (legacy)
# ─────────────────────────────────────────────────────────────────────────────

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, list[str], list[int], list[list[int]]]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, list[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[dict[str, float]] = None
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    suffix: Optional[str] = None
    seed: Optional[int] = None
    user: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# POST /v1/completions — OUTPUT (legacy)
# ─────────────────────────────────────────────────────────────────────────────

class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage