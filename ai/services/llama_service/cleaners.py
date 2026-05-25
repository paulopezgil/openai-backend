UNSUPPORTED_CHAT_PARAMS = {
    "user", "service_tier", "metadata", "reasoning_effort",
    "stream_options", "parallel_tool_calls",
}

UNSUPPORTED_EMBEDDING_PARAMS = {"user"}

UNSUPPORTED_COMPLETION_PARAMS: set[str] = set()

PARAMETER_MAPPINGS = {
    "max_completion_tokens": "max_tokens",
}


def clean_chat_params(body: dict) -> dict:
    cleaned = {}
    for k, v in body.items():
        if k in UNSUPPORTED_CHAT_PARAMS or v is None:
            continue
        if k == "n":
            v = 1
            continue
        mapped = PARAMETER_MAPPINGS.get(k, k)
        cleaned[mapped] = v
    return cleaned


def clean_embedding_params(body: dict) -> dict:
    return {k: v for k, v in body.items() if k not in UNSUPPORTED_EMBEDDING_PARAMS}


def clean_completion_params(body: dict) -> dict:
    return {k: v for k, v in body.items() if k not in UNSUPPORTED_COMPLETION_PARAMS}
