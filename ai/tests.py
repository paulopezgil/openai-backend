import json
from unittest.mock import patch

from django.test import TestCase, Client

from ai.exceptions import (
    BadRequest,
    ModelNotFoundError,
    ModelNotLoadedError,
    ServiceError,
    handle_exception,
)
from ai.services.llama_service.cleaners import (
    clean_chat_params,
    clean_embedding_params,
    clean_completion_params,
)


class TestCleanChatParams(TestCase):
    def test_strips_unsupported(self):
        body = {
            "model": "m",
            "messages": [],
            "user": "x",
            "service_tier": "default",
            "metadata": {"k": "v"},
            "reasoning_effort": "high",
            "stream_options": {"include_usage": True},
            "parallel_tool_calls": True,
        }
        cleaned = clean_chat_params(body)
        for param in ("user", "service_tier", "metadata",
                      "reasoning_effort", "stream_options", "parallel_tool_calls"):
            self.assertNotIn(param, cleaned)

    def test_maps_max_completion_tokens(self):
        cleaned = clean_chat_params({"max_completion_tokens": 200})
        self.assertNotIn("max_completion_tokens", cleaned)
        self.assertEqual(cleaned["max_tokens"], 200)

    def test_removes_n(self):
        cleaned = clean_chat_params({"n": 2})
        self.assertNotIn("n", cleaned)

    def test_strips_null_values(self):
        cleaned = clean_chat_params({"model": "m", "temperature": None})
        self.assertNotIn("temperature", cleaned)
        self.assertIn("model", cleaned)

    def test_preserves_valid_params(self):
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.5,
            "max_tokens": 100,
            "stream": False,
        }
        cleaned = clean_chat_params(body)
        self.assertEqual(cleaned["model"], "m")
        self.assertEqual(cleaned["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(cleaned["temperature"], 0.5)
        self.assertEqual(cleaned["max_tokens"], 100)
        self.assertFalse(cleaned["stream"])

    def test_preserves_optional_llama_params(self):
        body = {"top_k": 40, "min_p": 0.05, "mirostat_mode": 2}
        cleaned = clean_chat_params(body)
        self.assertEqual(cleaned["top_k"], 40)
        self.assertEqual(cleaned["min_p"], 0.05)
        self.assertEqual(cleaned["mirostat_mode"], 2)


class TestCleanEmbeddingParams(TestCase):
    def test_strips_user(self):
        cleaned = clean_embedding_params({"input": "hello", "model": "m", "user": "x"})
        self.assertNotIn("user", cleaned)
        self.assertEqual(cleaned["input"], "hello")

    def test_preserves_valid_params(self):
        body = {"input": "hello", "model": "m"}
        self.assertEqual(clean_embedding_params(body), body)


class TestCleanCompletionParams(TestCase):
    def test_passthrough(self):
        body = {"prompt": "hello", "max_tokens": 50}
        self.assertEqual(clean_completion_params(body), body)


class TestHandleException(TestCase):
    def test_json_decode_error(self):
        try:
            json.loads("not-json")
        except json.JSONDecodeError as e:
            response = handle_exception(e)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(json.loads(response.content), {"error": "Invalid JSON body"})

    def test_bad_request(self):
        response = handle_exception(BadRequest("bad param"))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(json.loads(response.content), {"error": "bad param"})

    def test_model_not_found(self):
        response = handle_exception(ModelNotFoundError("not on disk"))
        self.assertEqual(response.status_code, 404)
        self.assertEqual(json.loads(response.content), {"error": "not on disk"})

    def test_model_not_loaded(self):
        response = handle_exception(ModelNotLoadedError("no model"))
        self.assertEqual(response.status_code, 500)
        self.assertEqual(json.loads(response.content), {"error": "no model"})

    def test_service_error(self):
        response = handle_exception(ServiceError("engine crashed"))
        self.assertEqual(response.status_code, 500)
        self.assertEqual(json.loads(response.content), {"error": "engine crashed"})

    def test_file_not_found(self):
        response = handle_exception(FileNotFoundError("missing.gguf"))
        self.assertEqual(response.status_code, 404)
        self.assertIn("Model not found", json.loads(response.content)["error"])

    def test_unhandled_exception(self):
        response = handle_exception(RuntimeError("weird"))
        self.assertEqual(response.status_code, 500)
        self.assertEqual(json.loads(response.content), {"error": "Internal server error"})


class ChatCompletionsViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.valid_body = {
            "model": "Qwen3-0.6B.Q2_K",
            "messages": [{"role": "user", "content": "Hi"}],
        }

    @patch("ai.views.request_manager.execute_chat")
    def test_success_non_streaming(self, mock_exec):
        mock_exec.return_value = {
            "id": "cmpl-1",
            "object": "chat.completion",
            "created": 1000000,
            "model": "Qwen3-0.6B.Q2_K",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "logprobs": None,
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        response = self.client.post(
            "/ai/v1/chat/completions",
            data=json.dumps(self.valid_body),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "Hello!")
        mock_exec.assert_called_once()

    @patch("ai.views.request_manager.execute_chat")
    def test_success_streaming(self, mock_exec):
        chunks = [
            {"id": "cmpl-1", "object": "chat.completion.chunk", "created": 1000000,
             "model": "m", "choices": [{"index": 0, "delta": {"role": "assistant"},
                                        "logprobs": None, "finish_reason": None}]},
            {"id": "cmpl-1", "object": "chat.completion.chunk", "created": 1000000,
             "model": "m", "choices": [{"index": 0, "delta": {"content": "Hello"},
                                        "logprobs": None, "finish_reason": None}]},
            {"id": "cmpl-1", "object": "chat.completion.chunk", "created": 1000000,
             "model": "m", "choices": [{"index": 0, "delta": {},
                                        "logprobs": None, "finish_reason": "stop"}]},
        ]
        mock_exec.return_value = iter(chunks)

        response = self.client.post(
            "/ai/v1/chat/completions",
            data=json.dumps({**self.valid_body, "stream": True}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["content-type"], "text/event-stream")

        content = b"".join(response.streaming_content).decode()
        self.assertIn("data: [DONE]", content)
        self.assertIn("Hello", content)

    def test_missing_model_returns_400(self):
        response = self.client.post(
            "/ai/v1/chat/completions",
            data=json.dumps({"messages": []}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("model is required", response.json()["error"])

    def test_invalid_json_returns_400(self):
        response = self.client.post(
            "/ai/v1/chat/completions",
            data="not-json",
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid JSON", response.json()["error"])

    @patch("ai.views.request_manager.execute_chat")
    def test_model_not_found_returns_404(self, mock_exec):
        mock_exec.side_effect = ModelNotFoundError("Model not on disk")

        response = self.client.post(
            "/ai/v1/chat/completions",
            data=json.dumps(self.valid_body),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 404)


class EmbeddingsViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    @patch("ai.views.request_manager.execute_embedding")
    def test_success(self, mock_exec):
        mock_exec.return_value = {
            "object": "list",
            "data": [{"object": "embedding", "index": 0,
                      "embedding": [0.1, 0.2, 0.3]}],
            "model": "Qwen3-0.6B.Q2_K",
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        }

        response = self.client.post(
            "/ai/v1/embeddings",
            data=json.dumps({"model": "Qwen3-0.6B.Q2_K", "input": "hello world"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["embedding"], [0.1, 0.2, 0.3])
        mock_exec.assert_called_once()

    def test_missing_model_returns_400(self):
        response = self.client.post(
            "/ai/v1/embeddings",
            data=json.dumps({"input": "hello"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("model is required", response.json()["error"])


class ModelsViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    @patch("ai.views.request_manager.model_service.list_ai_models")
    def test_success(self, mock_list):
        mock_list.return_value = ["model-a", "model-b"]

        response = self.client.get("/ai/v1/models")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["object"], "list")
        self.assertEqual(len(data["data"]), 2)
        self.assertEqual(data["data"][0]["id"], "model-a")
        self.assertEqual(data["data"][0]["object"], "model")
        self.assertEqual(data["data"][1]["id"], "model-b")
