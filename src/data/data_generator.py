"""训练数据生成器

将方法论知识库转换为训练数据格式，支持多样性数据生成。
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import random
import os

from ..kb.knowledge_base import KnowledgeBase, Method


@dataclass
class TrainingSample:
    """训练数据样本"""
    problem_id: str
    problem: str
    problem_type: str
    solution: str
    method_used: str
    method_steps: List[str]
    difficulty: int
    annotations: List[str]
    source: str = "generated"
    verified: bool = False


class DataGenerator:
    """训练数据生成器

    将方法论KB转换为训练数据格式，支持多样性生成。

    Attributes:
        kb: 方法论知识库
        templates: 问题模板库
        difficulty_range: 难度范围
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        problem_templates: Optional[Dict] = None,
        difficulty_range: Tuple[int, int] = (1, 5)
    ):
        """初始化数据生成器

        Args:
            kb: 方法论知识库
            problem_templates: 问题模板库（可选）
            difficulty_range: 难度范围 (min, max)
        """
        self.kb = kb
        self.problem_templates = problem_templates or {}
        self.difficulty_range = difficulty_range
        self.generated_count = 0

        # 问题模板（内置）
        self._builtin_templates = {
            "方程求解": [
                "求解方程 {equation}，其中 {conditions}",
                "已知 {variable} 满足 {constraint}，求 {target}",
                "解方程组：{system}，并求 {additional}"
            ],
            "不等式证明": [
                "证明对于所有 {domain}，{inequality} 成立",
                "设 {params}，证明 {inequality} 当 {condition} 时成立",
                "求 {expression} 的最值，并证明最优性"
            ],
            "函数最值": [
                "求函数 {function} 在 {domain} 上的最大值和最小值",
                "给定 {constraint}，求 {expression} 的极值",
                "求 {sequence} 的最值项"
            ],
            "代数恒等式": [
                "证明恒等式：{identity}",
                "化简表达式 {expression}，并证明简化结果",
                "证明 {left} = {right} 对所有 {domain} 成立"
            ],
            "整除性": [
                "证明 {expression} 被 {divisor} 整除",
                "求 {number} 除以 {divisor} 的余数",
                "确定 {sequence} 中被 {divisor} 整除的项"
            ],
            "计数问题": [
                "计算满足 {condition} 的 {object} 的数量",
                "求 {set} 中满足 {property} 的元素个数",
                "有多少种方式可以 {action}？"
            ],
            "几何证明": [
                "在 {figure} 中，证明 {statement}",
                "已知 {geometry_conditions}，求证 {conclusion}",
                "证明 {points} 具有 {property}"
            ],
            "模运算": [
                "计算 {expression} mod {modulus}",
                "证明 {congruence} 成立",
                "求满足 {congruence_condition} 的最小正整数"
            ]
        }

    def generate_sample(
        self,
        method_id: str,
        problem_type: Optional[str] = None,
        difficulty: Optional[int] = None
    ) -> Optional[TrainingSample]:
        """生成单个训练样本

        Args:
            method_id: 方法ID
            problem_type: 问题类型（可选，自动选择）
            difficulty: 难度等级（可选，使用方法默认难度）

        Returns:
            TrainingSample: 生成的样本，或None如果方法不存在
        """
        method = self.kb.get_method(method_id)
        if method is None:
            return None

        # 确定难度
        if difficulty is None:
            difficulty = method.difficulty
        difficulty = max(self.difficulty_range[0],
                        min(difficulty, self.difficulty_range[1]))

        # 确定问题类型
        if problem_type is None:
            problem_type = self._select_problem_type(method)

        # 生成问题描述
        problem = self._generate_problem(method, problem_type, difficulty)

        # 生成解题过程（含方法标注）
        solution, steps, annotations = self._generate_solution(
            method, problem, problem_type, difficulty
        )

        # 创建样本ID
        sample_id = f"{method_id}_{self.generated_count:04d}"
        self.generated_count += 1

        return TrainingSample(
            problem_id=sample_id,
            problem=problem,
            problem_type=problem_type,
            solution=solution,
            method_used=method_id,
            method_steps=steps,
            difficulty=difficulty,
            annotations=annotations,
            source="generated"
        )

    def generate_batch(
        self,
        method_ids: Optional[List[str]] = None,
        count_per_method: int = 3,
        difficulty_distribution: Optional[Dict[int, float]] = None
    ) -> List[TrainingSample]:
        """批量生成训练样本

        Args:
            method_ids: 方法ID列表（可选，使用所有方法）
            count_per_method: 每个方法生成的样本数
            difficulty_distribution: 难度分布（可选）

        Returns:
            List[TrainingSample]: 生成的样本列表
        """
        if method_ids is None:
            method_ids = list(self.kb.methods.keys())

        samples = []
        for method_id in method_ids:
            for i in range(count_per_method):
                # 根据分布选择难度
                difficulty = self._select_difficulty(difficulty_distribution)

                sample = self.generate_sample(method_id, difficulty=difficulty)
                if sample:
                    samples.append(sample)

        return samples

    def generate_diverse_dataset(
        self,
        total_samples: int = 100,
        balance_by_category: bool = True,
        balance_by_difficulty: bool = True
    ) -> List[TrainingSample]:
        """生成多样性数据集

        Args:
            total_samples: 总样本数
            balance_by_category: 按类别平衡
            balance_by_difficulty: 按难度平衡

        Returns:
            List[TrainingSample]: 多样化的样本列表
        """
        samples = []

        if balance_by_category:
            # 按类别分配样本
            categories = list(self.kb.category_index.keys())
            samples_per_category = total_samples // len(categories)

            for category in categories:
                method_ids = self.kb.category_index.get(category, [])
                count_per_method = max(1, samples_per_category // len(method_ids))

                for method_id in method_ids:
                    for diff in range(self.difficulty_range[0],
                                     self.difficulty_range[1] + 1):
                        if balance_by_difficulty:
                            sample = self.generate_sample(
                                method_id, difficulty=diff
                            )
                            if sample:
                                samples.append(sample)

                        if len(samples) >= total_samples:
                            break

                    if len(samples) >= total_samples:
                        break

                if len(samples) >= total_samples:
                    break
        else:
            # 随机生成
            method_ids = list(self.kb.methods.keys())
            for _ in range(total_samples):
                method_id = random.choice(method_ids)
                difficulty = random.randint(
                    self.difficulty_range[0],
                    self.difficulty_range[1]
                )
                sample = self.generate_sample(method_id, difficulty=difficulty)
                if sample:
                    samples.append(sample)

        return samples[:total_samples]

    def _select_problem_type(self, method: Method) -> str:
        """选择问题类型"""
        if method.applicability:
            types = []
            for app in method.applicability:
                types.extend(app.get('problem_types', []))
            if types:
                return random.choice(types)

        # 默认问题类型
        default_types = ["方程求解", "不等式证明", "函数最值"]
        return random.choice(default_types)

    def _select_difficulty(
        self,
        distribution: Optional[Dict[int, float]] = None
    ) -> int:
        """根据分布选择难度"""
        if distribution is None:
            # 默认分布：中等难度居多
            distribution = {1: 0.1, 2: 0.2, 3: 0.4, 4: 0.2, 5: 0.1}

        difficulties = list(distribution.keys())
        weights = list(distribution.values())

        return random.choices(difficulties, weights=weights)[0]

    def _generate_problem(
        self,
        method: Method,
        problem_type: str,
        difficulty: int
    ) -> str:
        """生成问题描述"""
        # 获取模板
        templates = self._builtin_templates.get(problem_type, [])
        if not templates:
            templates = ["求解问题：{expression}"]

        template = random.choice(templates)

        # 根据难度填充模板
        problem = self._fill_template(template, method, difficulty)

        return problem

    def _fill_template(
        self,
        template: str,
        method: Method,
        difficulty: int
    ) -> str:
        """填充问题模板"""
        # 简化的填充逻辑
        # 实际应用中可以更复杂，生成具体数学表达式

        placeholders = {
            "{equation}": f"x^2 + {difficulty*2}x + {difficulty*3} = 0",
            "{conditions}": "x > 0",
            "{variable}": "x",
            "{constraint}": f"x + y = {difficulty*10}",
            "{target}": "x 和 y 的值",
            "{system}": f"x + y = {difficulty*10}, xy = {difficulty*5}",
            "{additional}": "x^2 + y^2",
            "{domain}": f"x ∈ [{difficulty}, {difficulty*10}]",
            "{inequality}": f"x^2 + y^2 ≥ {difficulty*5}",
            "{params}": f"a = {difficulty}, b = {difficulty*2}",
            "{condition}": "a + b > 0",
            "{expression}": f"{difficulty}x^2 + {difficulty*2}xy",
            "{function}": f"f(x) = x^2 - {difficulty*2}x + {difficulty*5}",
            "{sequence}": f"a_n = n^2 + {difficulty}",
            "{identity}": f"(a+b)^2 = a^2 + 2ab + b^2",
            "{left}": "(a+b)^2",
            "{right}": "a^2 + 2ab + b^2",
            "{divisor}": str(difficulty + 1),
            "{number}": str(difficulty * 100 + 7),
            "{set}": f"{difficulty}到{difficulty*10}的正整数",
            "{property}": "是完全平方数",
            "{object}": "排列",
            "{action}": f"将{difficulty*2}个不同元素排列",
            "{figure}": "三角形ABC",
            "{statement}": f"AB = AC = {difficulty}",
            "{geometry_conditions}": f"AB = {difficulty}, BC = {difficulty*2}",
            "{conclusion}": "三角形ABC是直角三角形",
            "{points}": "A, B, C",
            "{modulus}": str(difficulty + 2),
            "{congruence}": f"a ≡ {difficulty} (mod {difficulty+2})",
            "{congruence_condition}": f"x ≡ 1 (mod {difficulty})"
        }

        result = template
        for key, value in placeholders.items():
            if key in result:
                result = result.replace(key, str(value))

        return result

    def _generate_solution(
        self,
        method: Method,
        problem: str,
        problem_type: str,
        difficulty: int
    ) -> Tuple[str, List[str], List[str]]:
        """生成解题过程

        Returns:
            Tuple[str, List[str], List[str]]: (完整解, 步骤列表, 注释列表)
        """
        steps = []
        annotations = []

        # 从方法模板获取步骤
        method_steps = method.template.get('steps', [])
        common_tricks = method.template.get('common_tricks', [])
        pitfalls = method.template.get('pitfall_warnings', [])

        # 构建解题过程
        solution_parts = []

        # 添加方法选择说明
        annotation = f"[方法选择] 根据问题特征，选择{method.name}"
        annotations.append(annotation)
        solution_parts.append(annotation)

        # 添加方法描述
        solution_parts.append(f"\n【{method.name}】\n{method.description.strip()}\n")

        # 添加解题步骤
        solution_parts.append("\n解题步骤：")
        for i, step in enumerate(method_steps):
            step_text = f"步骤{i+1}: {step}"
            steps.append(step_text)

            # 添加具体操作（根据难度调整复杂度）
            step_detail = self._generate_step_detail(step, difficulty)
            solution_parts.append(f"  {step_text}")
            if step_detail:
                solution_parts.append(f"    {step_detail}")

        # 添加技巧提示
        if common_tricks:
            annotation = "[技巧提示] " + random.choice(common_tricks)
            annotations.append(annotation)
            solution_parts.append(f"\n技巧提示：{annotation}")

        # 添加注意事项
        if pitfalls:
            annotation = "[注意事项] " + random.choice(pitfalls)
            annotations.append(annotation)
            solution_parts.append(f"\n注意事项：{annotation}")

        # 添加最终答案
        answer = self._generate_answer(difficulty)
        solution_parts.append(f"\n答案：{answer}")

        solution = "\n".join(solution_parts)
        return solution, steps, annotations

    def _generate_step_detail(self, step: str, difficulty: int) -> str:
        """生成步骤的具体细节"""
        # 根据步骤类型生成具体操作
        if "识别" in step or "观察" in step:
            return f"分析给定条件，发现关键特征"
        elif "选择" in step or "设" in step:
            return f"令 t = {difficulty}x，简化表达式"
        elif "转化" in step or "变换" in step:
            return f"将原式转化为关于 t 的表达式"
        elif "求解" in step:
            return f"计算得到 t = {difficulty * 2}"
        elif "回代" in step or "返回" in step:
            return f"代回得到最终结果"
        elif "验证" in step or "检验" in step:
            return f"检查结果满足所有条件"
        else:
            return ""

    def _generate_answer(self, difficulty: int) -> str:
        """生成答案"""
        return f"x = {difficulty}, y = {difficulty * 2}"

    def save_samples(
        self,
        samples: List[TrainingSample],
        path: str
    ):
        """保存样本到文件"""
        data = [asdict(s) for s in samples]

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_samples(cls, path: str) -> List[TrainingSample]:
        """从文件加载样本"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return [TrainingSample(**item) for item in data]