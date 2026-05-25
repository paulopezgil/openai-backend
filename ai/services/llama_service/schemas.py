from typing import Optional, Union, Dict
from pydantic import BaseModel
from llama_cpp import (
    ChatCompletionRequestMessage, ChatCompletionFunction,
    ChatCompletionRequestFunctionCall, ChatCompletionTool,
    ChatCompletionToolChoiceOption, ChatCompletionRequestResponseFormat,
    StoppingCriteriaList, LogitsProcessorList, LlamaGrammar,
)


class LlamaCppChatCompletionInput(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    # Required parameters
    messages: list[ChatCompletionRequestMessage]

    # Function/Tool parameters
    functions: Optional[list[ChatCompletionFunction]] = None
    function_call: Optional[ChatCompletionRequestFunctionCall] = None
    tools: Optional[list[ChatCompletionTool]] = None
    tool_choice: Optional[ChatCompletionToolChoiceOption] = None

    # Sampling parameters
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40
    min_p: Optional[float] = 0.05
    typical_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    stop: Optional[Union[str, list[str]]] = None
    seed: Optional[int] = None
    
    # Generation parameters
    response_format: Optional[ChatCompletionRequestResponseFormat] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    repeat_penalty: Optional[float] = 1.0
    
    # Advanced sampling parameters
    tfs_z: Optional[float] = 1.0
    mirostat_mode: Optional[int] = 0
    mirostat_tau: Optional[float] = 5.0
    mirostat_eta: Optional[float] = 0.1
    
    # Advanced parameters
    model: Optional[str] = None
    logits_processor: Optional[LogitsProcessorList] = None
    grammar: Optional[LlamaGrammar] = None
    logit_bias: Optional[dict[int, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None


class LlamaCppCompletionInput(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    # Required parameters
    prompt: Union[str, list[int]]
    
    # Text generation parameters
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16
    temperature: float = 0.8
    top_p: float = 0.95
    min_p: float = 0.05
    typical_p: float = 1.0
    logprobs: Optional[int] = None
    echo: bool = False
    stop: Optional[Union[str, list[str]]] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repeat_penalty: float = 1.0
    top_k: int = 40
    stream: bool = False
    seed: Optional[int] = None
    tfs_z: float = 1.0
    mirostat_mode: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    
    # Advanced parameters
    model: Optional[str] = None
    stopping_criteria: Optional[StoppingCriteriaList] = None
    logits_processor: Optional[LogitsProcessorList] = None
    grammar: Optional[LlamaGrammar] = None
    logit_bias: Optional[dict[int, float]] = None


class LlamaCppEmbeddingInput(BaseModel):
    # Required parameters
    input: Union[str, list[str]]
    
    # Model selection
    model: Optional[str] = None
