"""方法论提取器

从成功解答中提炼方法论。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json


@dataclass
class Method:
    """方法论定义"""
    method_id: str
    name: str
    category: str
    description: str
    applicability: List[Dict] = field(default_factory=list)
    template: Dict = field(default_factory=dict)
    difficulty: int = 3
    frequency: float = 0.5
    related_methods: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)


class MethodologyExtractor:
    """方法论提取器

    从模型的成功解答中提炼方法论模式。

    Attributes:
        assistant_model: 辅助提炼的大模型
        min_samples: 提炼所需的最小样本数
    """

    def __init__(self, assistant_model, min_samples: int = 3):
        """初始化提取器

        Args:
            assistant_model: 用于辅助提炼的大模型
            min_samples: 提炼所需的最小成功样本数
        """
        self.assistant_model = assistant_model
        self.min_samples = min_samples

    def extract_from_solutions(self, solutions: List[Dict]) -> List[Method]:
        """从解答集提炼方法论

        Args:
            solutions: 解答列表，每个解答包含problem, solution, correct等字段

        Returns:
            List[Method]: 提炼出的方法论列表
        """
        # 只从成功解答中提炼
        successful = [s for s in solutions if s.get('correct', False)]

        if len(successful) < self.min_samples:
            return []

        # 聚类相似解答
        clusters = self._cluster_solutions(successful)

        # 从每个簇提炼方法
        methods = []
        for cluster in clusters:
            if len(cluster) >= self.min_samples:
                method = self._extract_method_from_cluster(cluster)
                if method:
                    methods.append(method)

        return methods

    def _cluster_solutions(self, solutions: List[Dict]) -> List[List[Dict]]:
        """聚类相似解答"""
        # 简化实现：按问题类型聚类
        clusters = {}

        for solution in solutions:
            problem_type = solution.get('problem_type', 'unknown')
            if problem_type not in clusters:
                clusters[problem_type] = []
            clusters[problem_type].append(solution)

        return list(clusters.values())

    def _extract_method_from_cluster(self, cluster: List[Dict]) -> Optional[Method]:
        """从解答簇提炼方法论

        Args:
            cluster: 相似解答的列表

        Returns:
            Optional[Method]: 提炼出的方法论，失败返回None
        """
        prompt = self._build_extraction_prompt(cluster)

        try:
            response = self.assistant_model.generate(prompt, temperature=0.3)
            method_data = json.loads(response)

            return Method(
                method_id=method_data.get('method_id', 'AUTO_001'),
                name=method_data.get('name', '未命名方法'),
                category=method_data.get('category', 'GENERAL'),
                description=method_data.get('description', ''),
                applicability=method_data.get('applicability', []),
                template=method_data.get('template', {}),
                difficulty=method_data.get('difficulty', 3),
                frequency=method_data.get('frequency', 0.5),
                related_methods=method_data.get('related_methods', []),
                examples=[s.get('problem_id', '') for s in cluster[:3]]
            )
        except Exception as e:
            print(f"提取失败: {e}")
            return None

    def _build_extraction_prompt(self, cluster: List[Dict]) -> str:
        """构建提取提示"""
        examples = "\n\n".join([
            f"问题{i+1}: {s.get('problem', '')}\n解答: {s.get('solution', '')[:500]}..."
            for i, s in enumerate(cluster[:5])
        ])

        return f"""分析以下成功解答，提炼出通用的解题方法论：

{examples}

请输出JSON格式的方法论定义：
{{
    "method_id": "AUTO_XXX",
    "name": "方法名称",
    "category": "ALGEBRA/GEOMETRY/NUMBER_THEORY/COMBINATORICS/GENERAL",
    "description": "详细描述方法原理和适用场景（≥50字）",
    "applicability": [
        {{
            "condition": "适用条件",
            "keywords": ["关键词1", "关键词2"],
            "problem_types": ["题型1", "题型2"]
        }}
    ],
    "template": {{
        "steps": ["步骤1", "步骤2", ...],
        "common_tricks": ["技巧1", "技巧2"],
        "pitfall_warnings": ["陷阱1"]
    }},
    "difficulty": 1-5,
    "frequency": 0.0-1.0,
    "related_methods": ["相关方法ID"]
}}
"""

    def validate_extracted_method(self, method: Method) -> bool:
        """验证提取的方法论是否有效

        Args:
            method: 待验证的方法论

        Returns:
            bool: 是否有效
        """
        if len(method.description) < 20:
            return False
        if not method.template.get('steps'):
            return False
        if len(method.template['steps']) < 2:
            return False
        return True