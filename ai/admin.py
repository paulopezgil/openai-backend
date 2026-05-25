from django.contrib import admin

from .models import GGUFModel


@admin.register(GGUFModel)
class GGUFModelAdmin(admin.ModelAdmin):
    list_display = ("filename", "type", "n_ctx", "n_gpu_layers")
