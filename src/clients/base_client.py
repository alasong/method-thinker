"""Base client interface for Layer2 multi-model validation."""

from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result from validate_method call."""
    score: float  # 0-10
    confidence: float  # 0-1
    issues: list
    reasoning: str
    details: Dict = None


class BaseClient(ABC):
    """Abstract base class for API clients.

    All clients used in Layer2 multi-model validation must implement
    this interface.
    """

    @abstractmethod
    def validate_method(self, method: Dict) -> ValidationResult:
        """Validate a methodology using this model.

        Args:
            method: Methodology dictionary containing name, description,
                    applicability, template with steps, etc.

        Returns:
            ValidationResult with score, confidence, issues, reasoning.
        """
        pass

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        """Generate a response from the model.

        Args:
            prompt: The prompt to send to the model.
            temperature: Sampling temperature (0-1).

        Returns:
            Model response as string.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name/identifier.

        Returns:
            Model name string (e.g., 'DeepSeek-V3', 'Qwen-Math-72B').
        """
        pass

    @abstractmethod
    def estimate_cost(self, tokens: int) -> float:
        """Estimate cost for given token count.

        Args:
            tokens: Estimated number of tokens.

        Returns:
            Estimated cost in USD.
        """
        pass

    def _build_validation_prompt(self, method: Dict) -> str:
        """Build standard validation prompt for methodology.

        Args:
            method: Methodology dictionary.

        Returns:
            Formatted prompt string.
        """
        return f"""你是一位数学方法论评审专家，请评估以下方法论的质量。

方法名称：{method.get('name', '')}
方法描述：{method.get('description', '')}
适用条件：{method.get('applicability', [])}
执行步骤：{method.get('template', {}).get('steps', [])}

请从以下维度评分（每项0-10分）：
1. 正确性：方法原理是否正确？
2. 完整性：步骤是否完整？
3. 适用性：适用条件是否合理？
4. 清晰度：描述是否清晰？
5. 实用性：这个方法有实际价值吗？

输出JSON格式：
{
    "scores": {"correctness": X, "completeness": X, "applicability": X, "clarity": X, "practicality": X},
    "overall_score": X,
    "confidence": 0.X,
    "issues": ["问题1", "问题2"],
    "reasoning": "评估理由"
}
"""