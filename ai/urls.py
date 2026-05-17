from django.urls import path
from . import views

urlpatterns = [
    path("v1/chat/completions", views.ChatCompletionsView.as_view(), name="chat_completions"),
    path("v1/embeddings", views.EmbeddingsView.as_view(), name="embeddings"),
    path("v1/models", views.ModelsListView.as_view(), name="models_list"),
    path("v1/models/<str:model_name>", views.ModelDetailView.as_view(), name="model_detail"),
]