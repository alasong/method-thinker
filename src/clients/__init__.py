"""API clients for Layer2 multi-model validation."""

from .base_client import BaseClient, ValidationResult
from .mock_client import MockClient

# Optional imports for real API clients
try:
    from .deepseek_client import DeepSeekClient
except ImportError:
    DeepSeekClient = None

try:
    from .qwen_client import QwenMathClient
except ImportError:
    QwenMathClient = None

try:
    from .openai_client import OpenAIClient
except ImportError:
    OpenAIClient = None

__all__ = [
    'BaseClient',
    'ValidationResult',
    'MockClient',
    'DeepSeekClient',
    'QwenMathClient',
    'OpenAIClient',
]