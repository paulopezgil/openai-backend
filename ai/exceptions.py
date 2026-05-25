import json
import logging

from django.http import JsonResponse


logger = logging.getLogger(__name__)


class APIError(Exception):
    status_code = 500


class BadRequest(APIError):
    status_code = 400


class NotFound(APIError):
    status_code = 404


class ModelNotFoundError(NotFound):
    """Requested model file does not exist on disk."""


class ModelNotLoadedError(APIError):
    """Inference attempted without a model loaded in memory."""


class ServiceError(APIError):
    """Unexpected error from the underlying inference engine."""


def handle_exception(error: Exception) -> JsonResponse:
    if isinstance(error, json.JSONDecodeError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)
    if isinstance(error, APIError):
        return JsonResponse({"error": str(error)}, status=error.status_code)
    if isinstance(error, FileNotFoundError):
        return JsonResponse({"error": f"Model not found: {error}"}, status=404)
    logger.exception("Unhandled exception")
    return JsonResponse({"error": "Internal server error"}, status=500)
