"""OpenAI GPT-4o-mini API client for methodology validation."""

import json
import os
from typing import Dict

from .base_client import BaseClient, ValidationResult

# Lazy import for optional dependency
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


class OpenAIClient(BaseClient):
    """Client for OpenAI GPT-4o-mini API.

    GPT-4o-mini is a fast, cost-effective model for general reasoning tasks.
    """

    MODEL_NAME = "GPT-4o-mini"
    MODEL_ID = "gpt-4o-mini"
    COST_PER_1K_INPUT_TOKENS = 0.00015  # USD per 1000 input tokens
    COST_PER_1K_OUTPUT_TOKENS = 0.0006  # USD per 1000 output tokens

    def __init__(self, api_key: str = None):
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key. If not provided, reads from
                     OPENAI_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter.")
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Run: pip install openai")
        self.client = OpenAI(api_key=self.api_key)

    def validate_method(self, method: Dict) -> ValidationResult:
        """Validate methodology using GPT-4o-mini.

        Args:
            method: Methodology dictionary.

        Returns:
            ValidationResult with assessment.
        """
        prompt = self._build_validation_prompt(method)
        response = self.generate(prompt, temperature=0.3)
        return self._parse_response(response)

    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        """Generate response from GPT-4o-mini.

        Args:
            prompt: Input prompt.
            temperature: Sampling temperature.

        Returns:
            Model response string.
        """
        response = self.client.chat.completions.create(
            model=self.MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1024
        )

        return response.choices[0].message.content

    def get_model_name(self) -> str:
        """Return model name."""
        return self.MODEL_NAME

    def estimate_cost(self, tokens: int) -> float:
        """Estimate cost based on token count.

        Uses average of input/output token costs.

        Args:
            tokens: Number of tokens.

        Returns:
            Estimated cost in USD.
        """
        # Average cost assuming 50/50 input/output split
        avg_cost_per_1k = (self.COST_PER_1K_INPUT_TOKENS + self.COST_PER_1K_OUTPUT_TOKENS) / 2
        return (tokens / 1000) * avg_cost_per_1k

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