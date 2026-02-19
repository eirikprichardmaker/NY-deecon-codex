# src/common/errors.py
from __future__ import annotations

class AppError(Exception):
    """Base class for predictable application errors."""
    exit_code: int = 1

class SchemaError(AppError):
    exit_code = 2

class ProviderError(AppError):
    exit_code = 3

class KeyErrorJoin(AppError):
    exit_code = 5

class DataQualityError(AppError):
    exit_code = 6
