from typing import Any, Optional, Union, Literal

from pydantic import BaseModel, Field


class TextContentPart(BaseModel):
    type: Literal["text"]
    text: str


class ImageURL(BaseModel):
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = "auto"


class ImageContentPart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageURL


ContentPart = Union[TextContentPart, ImageContentPart]


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"]
    function: FunctionCall


class SystemMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, list[TextContentPart]]
    name: Optional[str] = None


class UserMessage(BaseModel):
    role: Literal["user"]
    content: Union[str, list[ContentPart]]
    name: Optional[str] = None


class AssistantMessage(BaseModel):
    role: Literal["assistant"]
    content: Optional[Union[str, list[TextContentPart]]] = None
    name: Optional[str] = None
    refusal: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None


class ToolMessage(BaseModel):
    role: Literal["tool"]
    content: Union[str, list[TextContentPart]]
    tool_call_id: str


Message = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]


class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None
    strict: Optional[bool] = None


class FunctionTool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


Tool = FunctionTool


class ResponseFormatText(BaseModel):
    type: Literal["text"]


class ResponseFormatJSON(BaseModel):
    type: Literal["json_object"]


class JSONSchemaConfig(BaseModel):
    name: str
    description: Optional[str] = None
    schema_: Optional[dict[str, Any]] = None
    strict: Optional[bool] = None

    class Config:
        populate_by_name = True
        alias_priority = -1

    def __init__(self, **data):
        if "schema" in data and "schema_" not in data:
            data["schema_"] = data.pop("schema")
        super().__init__(**data)


class ResponseFormatJSONSchema(BaseModel):
    type: Literal["json_schema"]
    json_schema: JSONSchemaConfig


ResponseFormat = Union[ResponseFormatText, ResponseFormatJSON, ResponseFormatJSONSchema]


class ToolChoiceFunction(BaseModel):
    name: str


class ToolChoiceSpecific(BaseModel):
    type: Literal["function"]
    function: ToolChoiceFunction


ToolChoice = Union[Literal["none", "auto", "required"], ToolChoiceSpecific]


class ChatCompletionRequest(BaseModel):
    # Required parameters
    model: str
    messages: list[Message]

    # Llama-cpp supported parameters (Optional with defaults)
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stop: Optional[Union[str, list[str]]] = None
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[dict[str, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    seed: Optional[int] = None
    response_format: Optional[ResponseFormat] = None
    tools: Optional[list[Tool]] = None
    tool_choice: Optional[ToolChoice] = None

    # Unsupported llama-cpp parameters (filtered by our app)
    n: Optional[int] = 1
    stream_options: Optional[dict[str, Any]] = None
    user: Optional[str] = None
    parallel_tool_calls: Optional[bool] = True
    metadata: Optional[dict[str, str]] = None
    service_tier: Optional[Literal["auto", "default", "flex"]] = None
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None
    store: Optional[bool] = None

    # Advanced llama-cpp sampling parameters
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    typical_p: Optional[float] = None
    repeat_penalty: Optional[float] = None
    tfs_z: Optional[float] = None
    mirostat_mode: Optional[int] = None
    mirostat_tau: Optional[float] = None
    mirostat_eta: Optional[float] = None


class UsageDetails(BaseModel):
    cached_tokens: int = 0
    audio_tokens: int = 0
    reasoning_tokens: int = 0


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[UsageDetails] = None
    completion_tokens_details: Optional[UsageDetails] = None


class AssistantResponseMessage(BaseModel):
    role: Literal["assistant"]
    content: Optional[str] = None
    refusal: Optional[str] = None
    tool_calls: Optional[list] = None
    annotations: list = Field(default_factory=list)


class Choice(BaseModel):
    index: int
    message: AssistantResponseMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "function_call"]] = None
    logprobs: Optional[Any] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    system_fingerprint: Optional[str] = None
    choices: list[Choice]
    usage: CompletionUsage
    service_tier: Optional[str] = None


class ChunkDelta(BaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None
    tool_calls: Optional[list] = None
    refusal: Optional[str] = None


class StreamChoice(BaseModel):
    index: int
    delta: ChunkDelta
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    system_fingerprint: Optional[str] = None
    choices: list[StreamChoice]
    usage: Optional[CompletionUsage] = None