"""知识库管理"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml
import json
import os


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


class KnowledgeBase:
    """方法论知识库

    管理所有方法论的定义、索引和查询。

    Attributes:
        methods: 方法论字典
        category_index: 按类别索引
        keyword_index: 关键词倒排索引
    """

    def __init__(self):
        self.methods: Dict[str, Method] = {}
        self.category_index: Dict[str, List[str]] = {}
        self.keyword_index: Dict[str, List[str]] = {}
        self.problem_type_index: Dict[str, List[str]] = {}

    def add_method(self, method: Method):
        """添加方法到知识库"""
        self.methods[method.method_id] = method
        self._update_indices(method)

    def _update_indices(self, method: Method):
        """更新索引"""
        # 类别索引
        cat = method.category
        if cat not in self.category_index:
            self.category_index[cat] = []
        if method.method_id not in self.category_index[cat]:
            self.category_index[cat].append(method.method_id)

        # 关键词索引
        for app in method.applicability:
            for keyword in app.get('keywords', []):
                kw = keyword.lower()
                if kw not in self.keyword_index:
                    self.keyword_index[kw] = []
                if method.method_id not in self.keyword_index[kw]:
                    self.keyword_index[kw].append(method.method_id)

            # 题型索引
            for ptype in app.get('problem_types', []):
                if ptype not in self.problem_type_index:
                    self.problem_type_index[ptype] = []
                if method.method_id not in self.problem_type_index[ptype]:
                    self.problem_type_index[ptype].append(method.method_id)

    def get_method(self, method_id: str) -> Optional[Method]:
        """获取方法"""
        return self.methods.get(method_id)

    def get_applicable_methods(self, problem: str, problem_type: str) -> List[tuple]:
        """获取适用于给定问题的方法

        Args:
            problem: 问题描述
            problem_type: 问题类型

        Returns:
            List[tuple]: (Method, score) 列表，按分数降序
        """
        scores = {}

        # 题型匹配
        if problem_type in self.problem_type_index:
            for method_id in self.problem_type_index[problem_type]:
                scores[method_id] = scores.get(method_id, 0) + 0.4

        # 关键词匹配
        problem_lower = problem.lower()
        for keyword, method_ids in self.keyword_index.items():
            if keyword in problem_lower:
                for method_id in method_ids:
                    scores[method_id] = scores.get(method_id, 0) + 0.1

        # 生成结果列表
        results = []
        for method_id, score in scores.items():
            if method_id in self.methods:
                results.append((self.methods[method_id], min(score, 1.0)))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def get_methods_by_category(self, category: str) -> List[Method]:
        """获取某类别的所有方法"""
        method_ids = self.category_index.get(category, [])
        return [self.methods[mid] for mid in method_ids if mid in self.methods]

    def find_similar_methods(self, method: Method, threshold: float = 0.8) -> List[Method]:
        """查找相似方法"""
        similar = []
        for existing in self.methods.values():
            if existing.method_id == method.method_id:
                continue
            similarity = self._compute_similarity(method, existing)
            if similarity >= threshold:
                similar.append(existing)
        return similar

    def _compute_similarity(self, m1: Method, m2: Method) -> float:
        """计算两个方法的相似度"""
        # 名称相似度
        name_sim = self._jaccard_similarity(
            set(m1.name), set(m2.name)
        )
        # 描述相似度
        desc_sim = self._jaccard_similarity(
            set(m1.description.split()),
            set(m2.description.split())
        )
        return 0.6 * name_sim + 0.4 * desc_sim

    @staticmethod
    def _jaccard_similarity(s1: set, s2: set) -> float:
        """Jaccard相似度"""
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)

    def save(self, path: str):
        """保存知识库"""
        data = {
            'methods': {
                mid: {
                    'method_id': m.method_id,
                    'name': m.name,
                    'category': m.category,
                    'description': m.description,
                    'applicability': m.applicability,
                    'template': m.template,
                    'difficulty': m.difficulty,
                    'frequency': m.frequency,
                    'related_methods': m.related_methods,
                    'examples': m.examples
                }
                for mid, m in self.methods.items()
            }
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'KnowledgeBase':
        """加载知识库"""
        kb = cls()

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for mid, m_data in data.get('methods', {}).items():
            method = Method(
                method_id=m_data['method_id'],
                name=m_data['name'],
                category=m_data['category'],
                description=m_data['description'],
                applicability=m_data.get('applicability', []),
                template=m_data.get('template', {}),
                difficulty=m_data.get('difficulty', 3),
                frequency=m_data.get('frequency', 0.5),
                related_methods=m_data.get('related_methods', []),
                examples=m_data.get('examples', [])
            )
            kb.add_method(method)

        return kb

    @classmethod
    def from_yaml(cls, path: str) -> 'KnowledgeBase':
        """从YAML文件加载知识库"""
        kb = cls()

        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        for m_data in data.get('methods', []):
            method = Method(
                method_id=m_data['method_id'],
                name=m_data['name'],
                category=m_data['category'],
                description=m_data['description'],
                applicability=m_data.get('applicability', []),
                template=m_data.get('template', {}),
                difficulty=m_data.get('difficulty', 3),
                frequency=m_data.get('frequency', 0.5),
                related_methods=m_data.get('related_methods', []),
                examples=m_data.get('examples', [])
            )
            kb.add_method(method)

        return kb