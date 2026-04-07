"""DeepSeek-V3 API client for methodology validation."""

import json
import os
from typing import Dict
import requests

from .base_client import BaseClient, ValidationResult


class DeepSeekClient(BaseClient):
    """Client for DeepSeek-V3 API.

    DeepSeek-V3 excels at reasoning, mathematics, and programming tasks.
    """

    MODEL_NAME = "DeepSeek-V3"
    API_URL = "https://api.deepseek.com/v1/chat/completions"
    COST_PER_1K_TOKENS = 0.02  # USD per 1000 tokens (approximate)

    def __init__(self, api_key: str = None):
        """Initialize DeepSeek client.

        Args:
            api_key: DeepSeek API key. If not provided, reads from
                     DEEPSEEK_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key required. Set DEEPSEEK_API_KEY env var or pass api_key parameter.")

    def validate_method(self, method: Dict) -> ValidationResult:
        """Validate methodology using DeepSeek-V3.

        Args:
            method: Methodology dictionary.

        Returns:
            ValidationResult with assessment.
        """
        prompt = self._build_validation_prompt(method)
        response = self.generate(prompt, temperature=0.3)
        return self._parse_response(response)

    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        """Generate response from DeepSeek-V3.

        Args:
            prompt: Input prompt.
            temperature: Sampling temperature.

        Returns:
            Model response string.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 1024
        }

        response = requests.post(
            self.API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            raise RuntimeError(f"DeepSeek API error: {response.status_code} - {response.text}")

        data = response.json()
        return data["choices"][0]["message"]["content"]

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