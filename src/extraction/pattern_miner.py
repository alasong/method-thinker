"""模式挖掘器

从解答中发现隐藏的解题模式。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import re
from collections import Counter


@dataclass
class Pattern:
    """解题模式"""
    name: str
    description: str
    frequency: int
    examples: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)


class PatternMiner:
    """模式挖掘器

    从解答文本中挖掘常见的解题模式。
    """

    # 常见解题关键词模式
    KEYWORD_PATTERNS = {
        '变量替换': ['设', '令', '替换', '换元', 't ='],
        '配方法': ['配方', '完全平方', '平方项'],
        '归纳': ['归纳', '假设', '基础步骤', '归纳步骤'],
        '反证': ['假设不成立', '矛盾', '反证'],
        '分类讨论': ['分类', '情况', '分情况', '分别'],
        '构造': ['构造', '定义', '设...满足'],
        '放缩': ['放缩', '放大', '缩小', '不等式'],
        '递推': ['递推', '递归', 'f(n)', '递推式'],
    }

    def __init__(self):
        self.discovered_patterns = []

    def mine_patterns(self, solutions: List[Dict]) -> List[Pattern]:
        """从解答中挖掘模式

        Args:
            solutions: 解答列表

        Returns:
            List[Pattern]: 发现的模式列表
        """
        patterns = []

        # 统计关键词出现频率
        keyword_counts = Counter()
        keyword_examples = {}

        for solution in solutions:
            text = solution.get('solution', '')

            for pattern_name, keywords in self.KEYWORD_PATTERNS.items():
                for keyword in keywords:
                    if keyword in text:
                        keyword_counts[pattern_name] += 1
                        if pattern_name not in keyword_examples:
                            keyword_examples[pattern_name] = []
                        keyword_examples[pattern_name].append(
                            solution.get('problem_id', '')[:20]
                        )

        # 创建模式对象
        for pattern_name, count in keyword_counts.most_common(10):
            if count >= 3:  # 至少出现3次才算模式
                patterns.append(Pattern(
                    name=pattern_name,
                    description=f"在解答中频繁使用{pattern_name}技巧",
                    frequency=count,
                    examples=keyword_examples.get(pattern_name, [])[:5],
                    conditions=self._infer_conditions(pattern_name)
                ))

        return patterns

    def _infer_conditions(self, pattern_name: str) -> List[str]:
        """推断模式的适用条件"""
        condition_map = {
            '变量替换': ['表达式有重复结构', '表达式复杂需要简化'],
            '配方法': ['涉及二次式', '需要利用非负性'],
            '归纳': ['命题涉及正整数', '有递推结构'],
            '反证': ['直接证明困难', '需要证明不存在'],
            '分类讨论': ['有多种情况', '参数影响结果'],
            '构造': ['需要证明存在性', '需要构造反例'],
            '放缩': ['证明不等式', '估计范围'],
            '递推': ['有递推关系', '数列问题'],
        }
        return condition_map.get(pattern_name, [])

    def find_step_patterns(self, solutions: List[Dict]) -> List[Dict]:
        """发现步骤模式

        分析解答中的常见步骤序列
        """
        step_sequences = []

        for solution in solutions:
            text = solution.get('solution', '')
            # 提取步骤标记
            steps = re.findall(r'(?:步骤|Step|\d+[\.\、])[^\n]+', text)
            if len(steps) >= 2:
                step_sequences.append(steps)

        # 寻找共同步骤
        if not step_sequences:
            return []

        common_steps = Counter()
        for seq in step_sequences:
            for step in seq:
                # 标准化步骤描述
                normalized = self._normalize_step(step)
                common_steps[normalized] += 1

        return [
            {'step': step, 'frequency': count}
            for step, count in common_steps.most_common(20)
            if count >= 3
        ]

    def _normalize_step(self, step: str) -> str:
        """标准化步骤描述"""
        # 移除编号
        step = re.sub(r'^\d+[\.\、]\s*', '', step)
        # 移除"步骤"前缀
        step = re.sub(r'^步骤\s*', '', step)
        return step.strip()[:50]