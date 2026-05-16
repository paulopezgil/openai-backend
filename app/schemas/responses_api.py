from typing import Any, Optional, Union

from pydantic import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
# POST /v1/responses — INPUT
# ─────────────────────────────────────────────────────────────────────────────

class ResponseInputText(BaseModel):
    type: str = "input_text"
    text: str


class ResponseInputImage(BaseModel):
    type: str = "input_image"
    image_url: Optional[str] = None
    file_id: Optional[str] = None
    detail: Optional[str] = "auto"


class ResponseInputFile(BaseModel):
    type: str = "input_file"
    file_id: str


class ResponseTextPart(BaseModel):
    type: str = "output_text"
    text: str
    annotations: list = []
    logprobs: list = []


class ResponseMessageContent(BaseModel):
    type: str = "message"
    role: str
    content: list


class ResponseItem(BaseModel):
    id: str
    type: str
    status: Optional[str] = None
    role: Optional[str] = None
    content: Optional[list] = None


# ─────────────────────────────────────────────────────────────────────────────
# POST /v1/responses — OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

class ResponseUsageDetails(BaseModel):
    reasoning_tokens: int = 0


class ResponseUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    output_tokens_details: Optional[ResponseUsageDetails] = None


class ResponseObject(BaseModel):
    id: str
    object: str = "response"
    status: str = "completed"
    created_at: int
    model: str
    output: list
    usage: ResponseUsage
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = None
    previous_response_id: Optional[str] = None
    reasoning: dict = {}
    store: bool = True
    text: dict = {}
    tool_choice: str = "auto"
    tools: list = []
    top_p: float = 1.0
    truncation: str = "disabled"
    parallel_tool_calls: bool = True
    user: Optional[str] = None
    metadata: dict = {}