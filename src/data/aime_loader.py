"""AIME数据加载器

加载和处理AIME数学竞赛题目数据。
支持YAML和JSON格式，提供筛选和转换功能。
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml


@dataclass
class AIMEProblem:
    """AIME题目数据结构"""
    id: str
    year: int
    number: int
    statement: str
    answer: str
    difficulty: int  # 1-5
    category: str  # ALGEBRA, GEOMETRY, NUMBER_THEORY, COMBINATORICS
    subcategory: str = ""
    methods: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    def to_training_sample(self) -> Dict[str, Any]:
        """转换为训练样本格式

        Returns:
            Dict: 符合MethodologyDataset格式的训练样本
        """
        return {
            'problem_id': self.id,
            'problem': self.statement,
            'problem_type': self.category,
            'difficulty': self.difficulty,
            'candidate_methods': [
                {'method_id': m, 'confidence': 0.8}
                for m in self.methods
            ],
            'selected_method': self.methods[0] if self.methods else '',
            'selection_reasoning': f"Based on keywords: {', '.join(self.keywords)}",
            'solution_steps': [],  # 需要后续填充
            'solution_annotations': [],
            'reflection': '',
            'source': f'AIME_{self.year}',
            'verified': False,
            'metadata': {
                'year': self.year,
                'number': self.number,
                'answer': self.answer,
                'subcategory': self.subcategory,
                'keywords': self.keywords
            }
        }


class AIMELoader:
    """AIME数据加载器

    支持从YAML/JSON文件加载AIME题目，并提供筛选功能。

    Attributes:
        problems: 加载的题目列表
        metadata: 数据集元信息

    Example:
        >>> loader = AIMELoader('data/test_sets/aime_samples.yaml')
        >>> algebra_problems = loader.filter_by_category('ALGEBRA')
        >>> hard_problems = loader.filter_by_difficulty(4, 5)
    """

    def __init__(self, path: Optional[str] = None):
        """初始化加载器

        Args:
            path: 数据文件路径（支持.yaml和.json）
        """
        self.problems: List[AIMEProblem] = []
        self.metadata: Dict[str, Any] = {}
        self.method_coverage: Dict[str, int] = {}

        if path:
            self.load(path)

    def load(self, path: str) -> None:
        """从文件加载数据

        Args:
            path: 数据文件路径

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不支持
        """
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        suffix = file_path.suffix.lower()

        if suffix == '.yaml' or suffix == '.yml':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        elif suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .yaml or .json")

        self._parse_data(data)

    def _parse_data(self, data: Dict[str, Any]) -> None:
        """解析数据结构

        Args:
            data: 原始数据字典
        """
        # 解析题目列表
        problems_data = data.get('problems', [])
        for item in problems_data:
            problem = AIMEProblem(
                id=item.get('id', ''),
                year=item.get('year', 0),
                number=item.get('number', 0),
                statement=item.get('statement', ''),
                answer=item.get('answer', ''),
                difficulty=item.get('difficulty', 3),
                category=item.get('category', 'GENERAL'),
                subcategory=item.get('subcategory', ''),
                methods=item.get('methods', []),
                keywords=item.get('keywords', [])
            )
            self.problems.append(problem)

        # 解析元数据
        self.metadata = data.get('metadata', {})
        self.method_coverage = data.get('method_coverage', {})

    def save(self, path: str, format: str = 'yaml') -> None:
        """保存数据到文件

        Args:
            path: 目标文件路径
            format: 输出格式 ('yaml' 或 'json')
        """
        data = self._to_dict()
        file_path = Path(path)

        if format == 'yaml':
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, allow_unicode=True, sort_keys=False)
        elif format == 'json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _to_dict(self) -> Dict[str, Any]:
        """转换为字典格式

        Returns:
            Dict: 包含所有数据的字典
        """
        problems_data = [
            {
                'id': p.id,
                'year': p.year,
                'number': p.number,
                'statement': p.statement,
                'answer': p.answer,
                'difficulty': p.difficulty,
                'category': p.category,
                'subcategory': p.subcategory,
                'methods': p.methods,
                'keywords': p.keywords
            }
            for p in self.problems
        ]

        return {
            'problems': problems_data,
            'metadata': self.metadata,
            'method_coverage': self.method_coverage
        }

    def filter_by_category(self, category: str) -> List[AIMEProblem]:
        """按类别筛选题目

        Args:
            category: 题目类别

        Returns:
            List[AIMEProblem]: 筛选后的题目列表
        """
        return [p for p in self.problems if p.category == category]

    def filter_by_difficulty(
        self,
        min_diff: int,
        max_diff: int = 5
    ) -> List[AIMEProblem]:
        """按难度筛选题目

        Args:
            min_diff: 最小难度（1-5）
            max_diff: 最大难度（1-5）

        Returns:
            List[AIMEProblem]: 筛选后的题目列表
        """
        return [
            p for p in self.problems
            if min_diff <= p.difficulty <= max_diff
        ]

    def filter_by_year(self, year: int) -> List[AIMEProblem]:
        """按年份筛选题目

        Args:
            year: 题目年份

        Returns:
            List[AIMEProblem]: 筛选后的题目列表
        """
        return [p for p in self.problems if p.year == year]

    def filter_by_method(self, method_id: str) -> List[AIMEProblem]:
        """按方法筛选题目

        Args:
            method_id: 方法ID（如ALG_001）

        Returns:
            List[AIMEProblem]: 筛选后的题目列表
        """
        return [p for p in self.problems if method_id in p.methods]

    def filter_by_keyword(self, keyword: str) -> List[AIMEProblem]:
        """按关键词筛选题目

        Args:
            keyword: 关键词

        Returns:
            List[AIMEProblem]: 筛选后的题目列表
        """
        return [
            p for p in self.problems
            if keyword.lower() in [k.lower() for k in p.keywords]
        ]

    def get_training_samples(self) -> List[Dict[str, Any]]:
        """将所有题目转换为训练样本格式

        Returns:
            List[Dict]: 训练样本列表
        """
        return [p.to_training_sample() for p in self.problems]

    def get_filtered_samples(
        self,
        category: Optional[str] = None,
        min_difficulty: Optional[int] = None,
        max_difficulty: Optional[int] = None,
        year: Optional[int] = None,
        method: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取经过筛选的训练样本

        Args:
            category: 题目类别（可选）
            min_difficulty: 最小难度（可选）
            max_difficulty: 最大难度（可选）
            year: 年份（可选）
            method: 方法ID（可选）

        Returns:
            List[Dict]: 筛选后的训练样本列表
        """
        filtered = self.problems

        if category:
            filtered = self.filter_by_category(category)
        if min_difficulty is not None:
            filtered = [
                p for p in filtered
                if p.difficulty >= min_difficulty
            ]
        if max_difficulty is not None:
            filtered = [
                p for p in filtered
                if p.difficulty <= max_difficulty
            ]
        if year:
            filtered = [p for p in filtered if p.year == year]
        if method:
            filtered = [p for p in filtered if method in p.methods]

        return [p.to_training_sample() for p in filtered]

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息

        Returns:
            Dict: 统计信息字典
        """
        if not self.problems:
            return {'total': 0}

        stats = {
            'total': len(self.problems),
            'by_category': {},
            'by_difficulty': {},
            'by_year': {},
            'methods_used': set()
        }

        for p in self.problems:
            # 类别统计
            stats['by_category'][p.category] = \
                stats['by_category'].get(p.category, 0) + 1

            # 难度统计
            diff_key = f'level_{p.difficulty}'
            stats['by_difficulty'][diff_key] = \
                stats['by_difficulty'].get(diff_key, 0) + 1

            # 年份统计
            stats['by_year'][p.year] = \
                stats['by_year'].get(p.year, 0) + 1

            # 方法统计
            for m in p.methods:
                stats['methods_used'].add(m)

        stats['methods_used'] = sorted(list(stats['methods_used']))

        return stats

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(self, idx: int) -> AIMEProblem:
        return self.problems[idx]

    def __iter__(self):
        return iter(self.problems)


def create_aime_dataset(
    path: str,
    category: Optional[str] = None,
    min_difficulty: Optional[int] = None,
    max_difficulty: Optional[int] = None
) -> AIMELoader:
    """创建AIME数据集的便捷函数

    Args:
        path: 数据文件路径
        category: 筛选类别（可选）
        min_difficulty: 最小难度（可选）
        max_difficulty: 最大难度（可选）

    Returns:
        AIMELoader: 加载并筛选后的数据集
    """
    loader = AIMELoader(path)

    if category:
        loader.problems = loader.filter_by_category(category)
    if min_difficulty is not None or max_difficulty is not None:
        loader.problems = loader.filter_by_difficulty(
            min_difficulty or 1,
            max_difficulty or 5
        )

    return loader


# 预定义的数据集路径
DEFAULT_AIME_PATH = 'data/test_sets/aime_samples.yaml'