"""
URL routing for the AI API.
"""

from django.urls import path

from ai.views import ChatCompletionsView, ModelsView, EmbeddingsView

urlpatterns = [
    path("v1/chat/completions", ChatCompletionsView.as_view(), name="chat_completions"),
    path("v1/models", ModelsView.as_view(), name="models"),
    path("v1/embeddings", EmbeddingsView.as_view(), name="embeddings"),
]