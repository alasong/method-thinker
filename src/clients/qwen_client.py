"""Qwen-Math API client for methodology validation."""

import json
import os
from typing import Dict

from .base_client import BaseClient, ValidationResult

# Lazy import for optional dependency
try:
    from dashscope import Generation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    Generation = None


class QwenMathClient(BaseClient):
    """Client for Qwen-Math-72B via Alibaba DashScope API.

    Qwen-Math specializes in mathematical reasoning and problem-solving.
    """

    MODEL_NAME = "Qwen-Math-72B"
    MODEL_ID = "qwen-math-plus"  # DashScope model ID
    COST_PER_1K_TOKENS = 0.015  # USD per 1000 tokens (approximate)

    def __init__(self, api_key: str = None):
        """Initialize Qwen-Math client.

        Args:
            api_key: DashScope API key. If not provided, reads from
                     DASHSCOPE_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DashScope API key required. Set DASHSCOPE_API_KEY env var or pass api_key parameter.")
        if not DASHSCOPE_AVAILABLE:
            raise ImportError("dashscope package not installed. Run: pip install dashscope")
        # DashScope uses environment variable internally

    def validate_method(self, method: Dict) -> ValidationResult:
        """Validate methodology using Qwen-Math.

        Args:
            method: Methodology dictionary.

        Returns:
            ValidationResult with assessment.
        """
        prompt = self._build_validation_prompt(method)
        response = self.generate(prompt, temperature=0.3)
        return self._parse_response(response)

    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        """Generate response from Qwen-Math.

        Args:
            prompt: Input prompt.
            temperature: Sampling temperature.

        Returns:
            Model response string.
        """
        response = Generation.call(
            model=self.MODEL_ID,
            prompt=prompt,
            temperature=temperature,
            max_tokens=1024,
            result_format='message'
        )

        if response.status_code != 200:
            raise RuntimeError(f"Qwen API error: {response.status_code} - {response.message}")

        return response.output.choices[0].message.content

    def get_model_name(self) -> str:
        """Return model name."""
        return self.MODEL_NAME

    def estimate_cost(self, tokens: int) -> float:
        """Estimate cost based on token count.

        Args:
            tokens: Number of tokens.

        Returns:
            Estimated cost in USD.
        """
        return (tokens / 1000) * self.COST_PER_1K_TOKENS

    def _parse_response(self, response: str) -> ValidationResult:
        """Parse JSON response into ValidationResult.

        Args:
            response: Raw model response string.

        Returns:
            ValidationResult object.
        """
        try:
            # Extract JSON from response (may have markdown wrapper)
            content = response.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            data = json.loads(content)
            return ValidationResult(
                score=float(data.get("overall_score", 5)),
                confidence=float(data.get("confidence", 0.5)),
                issues=data.get("issues", []),
                reasoning=data.get("reasoning", ""),
                details={"scores": data.get("scores", {})}
            )
        except (json.JSONDecodeError, KeyError) as e:
            return ValidationResult(
                score=5.0,
                confidence=0.3,
                issues=["Response parsing failed"],
                reasoning=f"Parse error: {e}",
                details={"raw_response": response}
            )