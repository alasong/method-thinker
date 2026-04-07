"""Layer 0: 快速过滤器

负责快速识别明显问题，包括：
- 语法检查：JSON/YAML格式是否正确
- 字段完整性：必要字段是否齐全
- 值域检查：分数、难度等是否在合理范围
- 去重检查：是否与现有方法重复
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
import re
import json


@dataclass
class ValidationResult:
    """验证结果"""
    passed: bool
    layer: int
    confidence: float
    issues: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)


class Layer0FastFilter:
    """Layer 0: 快速过滤器

    用最小成本快速识别明显问题，避免后续资源浪费。

    Attributes:
        existing_kb: 已有的知识库
        existing_names: 已存在的方法名称集合
        existing_ids: 已存在的方法ID集合
    """

    REQUIRED_FIELDS = [
        'method_id',
        'name',
        'description',
        'applicability',
        'template'
    ]

    VALUE_CONSTRAINTS = {
        'difficulty': (1, 5),
        'frequency': (0, 1),
        'success_rate': (0, 1)
    }

    def __init__(self, existing_kb: Optional[Dict] = None):
        """初始化快速过滤器

        Args:
            existing_kb: 已有的知识库，用于去重检查
        """
        self.existing_kb = existing_kb or {}
        self.existing_names: Set[str] = set()
        self.existing_ids: Set[str] = set()

        if existing_kb and 'methods' in existing_kb:
            for method_id, method in existing_kb['methods'].items():
                # 支持Method对象和字典两种格式
                if hasattr(method, 'name'):
                    self.existing_names.add(method.name.lower())
                    self.existing_ids.add(method.method_id)
                else:
                    self.existing_names.add(method.get('name', '').lower())
                    self.existing_ids.add(method.get('method_id', ''))

    def validate(self, method: Dict) -> ValidationResult:
        """验证方法

        Args:
            method: 待验证的方法字典

        Returns:
            ValidationResult: 验证结果
        """
        issues = []

        # 1. 字段完整性检查
        missing_fields = self._check_required_fields(method)
        if missing_fields:
            issues.append(f"缺少必要字段: {missing_fields}")

        # 2. 字段格式检查
        format_issues = self._check_field_formats(method)
        issues.extend(format_issues)

        # 3. 值域检查
        value_issues = self._check_value_constraints(method)
        issues.extend(value_issues)

        # 4. 去重检查
        duplicate_issues = self._check_duplicates(method)
        issues.extend(duplicate_issues)

        # 5. 描述质量检查
        quality_issues = self._check_description_quality(method)
        issues.extend(quality_issues)

        passed = len(issues) == 0
        confidence = 1.0 if passed else 0.0

        return ValidationResult(
            passed=passed,
            layer=0,
            confidence=confidence,
            issues=issues,
            details={'fast_filter': 'passed' if passed else 'failed'}
        )

    def _check_required_fields(self, method: Dict) -> List[str]:
        """检查必要字段"""
        return [f for f in self.REQUIRED_FIELDS if f not in method]

    def _check_field_formats(self, method: Dict) -> List[str]:
        """检查字段格式"""
        issues = []

        if 'method_id' in method:
            if not re.match(r'^[A-Z]{3}_\d{3}$', str(method['method_id'])):
                issues.append(f"method_id格式错误: {method['method_id']}")

        if 'name' in method:
            name_len = len(str(method['name']))
            if name_len < 2 or name_len > 50:
                issues.append(f"name长度异常: {name_len}")

        if 'description' in method:
            desc_len = len(str(method['description']))
            if desc_len < 20:
                issues.append("description过短，可能不够详细")

        if 'applicability' in method:
            app = method['applicability']
            if not app or (isinstance(app, list) and len(app) == 0):
                issues.append("applicability为空")

        if 'template' in method:
            template = method['template']
            steps = template.get('steps', []) if isinstance(template, dict) else []
            if not steps or len(steps) < 2:
                issues.append("template.steps过少，方法步骤不完整")

        return issues

    def _check_value_constraints(self, method: Dict) -> List[str]:
        """检查值域约束"""
        issues = []

        for field_name, (min_val, max_val) in self.VALUE_CONSTRAINTS.items():
            if field_name in method:
                value = method[field_name]
                if isinstance(value, (int, float)):
                    if not (min_val <= value <= max_val):
                        issues.append(f"{field_name}值{value}超出范围[{min_val}, {max_val}]")

        return issues

    def _check_duplicates(self, method: Dict) -> List[str]:
        """检查重复"""
        issues = []

        method_id = method.get('method_id')
        if method_id and method_id in self.existing_ids:
            issues.append(f"method_id重复: {method_id}")

        name = str(method.get('name', '')).lower()
        for existing_name in self.existing_names:
            if self._compute_similarity(name, existing_name) > 0.9:
                issues.append(f"方法名称与现有方法'{existing_name}'高度相似")
                break

        return issues

    def _check_description_quality(self, method: Dict) -> List[str]:
        """检查描述质量"""
        issues = []
        description = str(method.get('description', ''))

        vague_phrases = ['通用的方法', '一般的方法', '常规方法', '正常方法']
        if any(phrase in description for phrase in vague_phrases):
            issues.append("描述过于泛化，缺乏具体性")

        return issues

    @staticmethod
    def _compute_similarity(s1: str, s2: str) -> float:
        """计算字符串相似度（Jaccard）"""
        if not s1 or not s2:
            return 0.0
        set1, set2 = set(s1), set(s2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0