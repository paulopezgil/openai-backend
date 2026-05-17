from django.db import models


class GGUFModel(models.Model):
    class ModelType(models.TextChoices):
        CHAT = "chat", "Chat"
        EMBEDDING = "embedding", "Embedding"

    filename = models.CharField(max_length=255, unique=True)
    created = models.DateTimeField()
    type = models.CharField(max_length=20, choices=ModelType.choices, default=ModelType.CHAT)
    n_ctx = models.PositiveIntegerField(default=2048)
    n_gpu_layers = models.PositiveIntegerField(default=0)
    n_threads = models.PositiveIntegerField(null=True, blank=True)
    seed = models.IntegerField(default=-1)

    class Meta:
        db_table = "gguf_models"

    def __str__(self):
        return self.filename