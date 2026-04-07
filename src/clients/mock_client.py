"""Mock client for testing Layer2 multi-model validation."""

import json
from typing import Dict

from .base_client import BaseClient, ValidationResult


class MockClient(BaseClient):
    """Mock client for testing purposes.

    Returns configurable responses without making actual API calls.
    Useful for unit testing and development.
    """

    def __init__(
        self,
        model_name: str = "MockModel",
        default_score: float = 7.0,
        default_confidence: float = 0.8,
        default_issues: list = None,
        default_reasoning: str = "Mock assessment"
    ):
        """Initialize mock client with configurable defaults.

        Args:
            model_name: Name to return for get_model_name().
            default_score: Default score for validate_method().
            default_confidence: Default confidence for validate_method().
            default_issues: Default issues list.
            default_reasoning: Default reasoning string.
        """
        self._model_name = model_name
        self._default_score = default_score
        self._default_confidence = default_confidence
        self._default_issues = default_issues or []
        self._default_reasoning = default_reasoning
        self._call_count = 0
        self._last_prompt = None

    def validate_method(self, method: Dict) -> ValidationResult:
        """Return mock validation result.

        Args:
            method: Methodology dictionary (stored but not used).

        Returns:
            ValidationResult with configured defaults.
        """
        self._call_count += 1
        return ValidationResult(
            score=self._default_score,
            confidence=self._default_confidence,
            issues=self._default_issues,
            reasoning=self._default_reasoning,
            details={"mock": True, "method_name": method.get("name", "unknown")}
        )

    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        """Return mock JSON response.

        Args:
            prompt: Input prompt (stored for inspection).
            temperature: Ignored in mock.

        Returns:
            JSON string matching expected format.
        """
        self._call_count += 1
        self._last_prompt = prompt

        return json.dumps({
            "scores": {
                "correctness": self._default_score,
                "completeness": self._default_score,
                "applicability": self._default_score,
                "clarity": self._default_score,
                "practicality": self._default_score
            },
            "overall_score": self._default_score,
            "confidence": self._default_confidence,
            "issues": self._default_issues,
            "reasoning": self._default_reasoning
        })

    def get_model_name(self) -> str:
        """Return configured model name."""
        return self._model_name

    def estimate_cost(self, tokens: int) -> float:
        """Return zero cost for mock."""
        return 0.0

    def get_call_count(self) -> int:
        """Get number of calls made to this client.

        Returns:
            Integer count of validate_method and generate calls.
        """
        return self._call_count

    def get_last_prompt(self) -> str:
        """Get the last prompt sent to generate().

        Returns:
            Last prompt string, or None if no calls made.
        """
        return self._last_prompt

    def reset(self):
        """Reset call count and last prompt for fresh testing."""
        self._call_count = 0
        self._last_prompt = None