# Base API Errors
class APIError(Exception):
    pass


# Model Service Errors
class ModelNotFoundError(APIError):
    pass