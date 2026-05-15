from django.db import models

class AIModel(models.Model):
    name = models.CharField(max_length=100, unique=True, help_with="Model ID for API calls")
    local_path = models.FilePathField(path="/models_storage/", recursive=True)
    context_size = models.IntegerField(default=2048)
    n_gpu_layers = models.IntegerField(default=-1, help_text="-1 for all layers on GPU")

    def __str__(self):
        return self.name