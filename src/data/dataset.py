"""方法论数据集"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import json


@dataclass
class MethodologySample:
    """方法论数据样本"""
    problem_id: str
    problem: str
    problem_type: str
    difficulty: int

    # 方法选择
    candidate_methods: List[Dict]
    selected_method: str
    selection_reasoning: str

    # 解答
    solution_steps: List[str]
    solution_annotations: List[str]

    # 反思
    reflection: str

    # 元数据
    source: str
    verified: bool = False


class MethodologyDataset:
    """方法论数据集

    处理方法论训练数据的加载和处理。

    Attributes:
        samples: 样本列表
        kb: 方法论知识库
    """

    def __init__(self, data_path: Optional[str] = None, kb=None):
        """初始化数据集

        Args:
            data_path: 数据文件路径
            kb: 方法论知识库
        """
        self.samples: List[MethodologySample] = []
        self.kb = kb

        if data_path:
            self.load(data_path)

    def load(self, path: str):
        """加载数据"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            sample = MethodologySample(
                problem_id=item.get('problem_id', ''),
                problem=item.get('problem', ''),
                problem_type=item.get('problem_type', ''),
                difficulty=item.get('difficulty', 3),
                candidate_methods=item.get('candidate_methods', []),
                selected_method=item.get('selected_method', ''),
                selection_reasoning=item.get('selection_reasoning', ''),
                solution_steps=item.get('solution_steps', []),
                solution_annotations=item.get('solution_annotations', []),
                reflection=item.get('reflection', ''),
                source=item.get('source', ''),
                verified=item.get('verified', False)
            )
            self.samples.append(sample)

    def save(self, path: str):
        """保存数据"""
        data = [
            {
                'problem_id': s.problem_id,
                'problem': s.problem,
                'problem_type': s.problem_type,
                'difficulty': s.difficulty,
                'candidate_methods': s.candidate_methods,
                'selected_method': s.selected_method,
                'selection_reasoning': s.selection_reasoning,
                'solution_steps': s.solution_steps,
                'solution_annotations': s.solution_annotations,
                'reflection': s.reflection,
                'source': s.source,
                'verified': s.verified
            }
            for s in self.samples
        ]

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def filter_by_type(self, problem_type: str) -> 'MethodologyDataset':
        """按题型过滤"""
        filtered = MethodologyDataset()
        filtered.samples = [
            s for s in self.samples
            if s.problem_type == problem_type
        ]
        return filtered

    def filter_by_difficulty(self, min_diff: int, max_diff: int) -> 'MethodologyDataset':
        """按难度过滤"""
        filtered = MethodologyDataset()
        filtered.samples = [
            s for s in self.samples
            if min_diff <= s.difficulty <= max_diff
        ]
        return filtered

    def split(self, ratios: List[float] = [0.8, 0.1, 0.1]) -> List['MethodologyDataset']:
        """划分数据集

        Args:
            ratios: 划分比例 [train, val, test]

        Returns:
            List[MethodologyDataset]: 划分后的数据集列表
        """
        total = len(self.samples)
        train_end = int(total * ratios[0])
        val_end = train_end + int(total * ratios[1])

        train_set = MethodologyDataset()
        val_set = MethodologyDataset()
        test_set = MethodologyDataset()

        train_set.samples = self.samples[:train_end]
        val_set.samples = self.samples[train_end:val_end]
        test_set.samples = self.samples[val_end:]

        return [train_set, val_set, test_set]