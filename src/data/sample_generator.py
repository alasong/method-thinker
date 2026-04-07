"""训练样本生成器

从题目+方法论KB生成训练样本，支持方法注入和Pass@K多样本生成。
"""

from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
import json
import random
import os
import hashlib

from ..kb.knowledge_base import KnowledgeBase, Method
from .method_injector import MethodInjector, MethodAnnotation


@dataclass
class TrainingSampleV2:
    """训练样本格式V2 - 支持方法论注入"""
    problem: str
    method_selection: str  # 方法选择推理过程
    solution_steps: List[str]  # 解题步骤列表
    final_answer: str
    method_id: str
    method_name: str
    problem_type: Optional[str] = None
    difficulty: int = 3
    annotations: Optional[List[Dict]] = None
    sample_id: Optional[str] = None

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return asdict(self)


class SampleGenerator:
    """训练样本生成器

    从方法论KB生成训练样本，支持方法注入和Pass@K多样本生成。

    Attributes:
        kb: 方法论知识库
        injector: 方法注入器
        llm_client: LLM客户端（用于生成具体解）
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        injector: Optional[MethodInjector] = None,
        llm_client: Optional[Callable] = None,
        config: Optional[Dict] = None
    ):
        """初始化样本生成器

        Args:
            kb: 方法论知识库
            injector: 方法注入器（可选，自动创建）
            llm_client: LLM生成函数（可选，用于生成具体解）
            config: 配置参数
        """
        self.kb = kb
        self.injector = injector or MethodInjector(kb)
        self.llm_client = llm_client
        self.config = config or {
            'include_method_description': True,
            'include_common_tricks': True,
            'include_pitfall_warnings': True,
            'max_steps': 10,
            'answer_format': 'detailed'
        }
        self._generated_count = 0

    def generate_sample(
        self,
        problem: str,
        method_id: str,
        problem_type: Optional[str] = None,
        difficulty: int = 3,
        raw_solution: Optional[str] = None
    ) -> TrainingSampleV2:
        """生成单个训练样本

        Args:
            problem: 题目描述
            method_id: 方法ID
            problem_type: 问题类型
            difficulty: 难度等级
            raw_solution: 原始解（可选，无则生成）

        Returns:
            TrainingSampleV2: 生成的训练样本
        """
        method = self.kb.get_method(method_id)
        if method is None:
            raise ValueError(f"方法不存在: {method_id}")

        # 生成方法选择推理
        method_selection = self._generate_method_selection(
            problem, method, problem_type
        )

        # 生成解题步骤
        if raw_solution:
            solution_steps = self._extract_steps_from_solution(raw_solution)
        else:
            solution_steps = self._generate_solution_steps(
                problem, method, difficulty
            )

        # 注入方法标注
        injected_steps, annotations = self._inject_method_into_steps(
            problem, solution_steps, method
        )

        # 生成最终答案
        final_answer = self._generate_final_answer(
            injected_steps, method, difficulty
        )

        # 生成样本ID
        sample_id = self._generate_sample_id(problem, method_id, self._generated_count)
        self._generated_count += 1

        return TrainingSampleV2(
            problem=problem,
            method_selection=method_selection,
            solution_steps=injected_steps,
            final_answer=final_answer,
            method_id=method_id,
            method_name=method.name,
            problem_type=problem_type,
            difficulty=difficulty,
            annotations=[self._annotation_to_dict(a) for a in annotations],
            sample_id=sample_id
        )

    def generate_pass_k_samples(
        self,
        problem: str,
        k: int = 8,
        problem_type: Optional[str] = None,
        difficulty: int = 3,
        diversity_mode: str = 'method'
    ) -> List[TrainingSampleV2]:
        """生成Pass@K多样本

        为同一问题生成K个不同的解，用于训练多样性。

        Args:
            problem: 题目描述
            k: 样本数量
            problem_type: 问题类型
            difficulty: 难度等级
            diversity_mode: 多样性模式
                - 'method': 使用不同方法
                - 'step': 同方法不同步骤表述
                - 'random': 随机多样性

        Returns:
            List[TrainingSampleV2]: K个训练样本
        """
        samples = []

        if diversity_mode == 'method':
            # 获取适用方法
            applicable_methods = self.kb.get_applicable_methods(
                problem, problem_type or 'general'
            )

            # 选择K个不同方法
            method_ids = self._select_k_methods(applicable_methods, k)

            for i, method_id in enumerate(method_ids):
                # 每个方法生成一个样本
                sample = self.generate_sample(
                    problem, method_id, problem_type, difficulty
                )
                # 标注样本序号
                sample.sample_id = f"{sample.sample_id}_k{i}"
                samples.append(sample)

            # 如果方法数不足K，用随机多样性补充
            remaining = k - len(method_ids)
            if remaining > 0:
                base_method = method_ids[0] if method_ids else 'ALG_001'
                for i in range(remaining):
                    sample = self._generate_diverse_variant(
                        problem, base_method, problem_type, difficulty,
                        variant_idx=len(samples)
                    )
                    samples.append(sample)

        elif diversity_mode == 'step':
            # 使用同一方法，但生成不同步骤表述
            base_method = self._select_best_method(problem, problem_type)
            for i in range(k):
                sample = self._generate_step_variant(
                    problem, base_method, problem_type, difficulty, variant_idx=i
                )
                samples.append(sample)

        else:  # random
            base_method = self._select_best_method(problem, problem_type)
            for i in range(k):
                sample = self.generate_sample(
                    problem, base_method, problem_type, difficulty
                )
                sample.sample_id = f"{sample.sample_id}_rand{i}"
                samples.append(sample)

        return samples[:k]

    def generate_batch(
        self,
        problems: List[Dict],
        samples_per_problem: int = 1,
        pass_k: Optional[int] = None,
        difficulty_distribution: Optional[Dict[int, float]] = None
    ) -> List[TrainingSampleV2]:
        """批量生成训练样本

        Args:
            problems: 问题列表，每项包含 problem, problem_type, difficulty
            samples_per_problem: 每题样本数（非Pass@K模式）
            pass_k: Pass@K数量（启用则忽略samples_per_problem）
            difficulty_distribution: 难度分布

        Returns:
            List[TrainingSampleV2]: 生成的样本列表
        """
        all_samples = []

        for prob_item in problems:
            problem = prob_item.get('problem', '')
            problem_type = prob_item.get('problem_type', 'general')

            # 选择难度
            if difficulty_distribution:
                difficulty = self._select_from_distribution(difficulty_distribution)
            else:
                difficulty = prob_item.get('difficulty', 3)

            if pass_k:
                # Pass@K模式
                samples = self.generate_pass_k_samples(
                    problem, k=pass_k, problem_type=problem_type,
                    difficulty=difficulty, diversity_mode='method'
                )
                all_samples.extend(samples)
            else:
                # 单样本模式
                method_id = self._select_best_method(problem, problem_type)
                for _ in range(samples_per_problem):
                    sample = self.generate_sample(
                        problem, method_id, problem_type, difficulty
                    )
                    all_samples.append(sample)

        return all_samples

    def generate_from_kb(
        self,
        total_samples: int = 100,
        balance_by_category: bool = True,
        balance_by_difficulty: bool = True,
        problem_types: Optional[List[str]] = None
    ) -> List[TrainingSampleV2]:
        """从KB自动生成训练样本

        Args:
            total_samples: 总样本数
            balance_by_category: 按类别平衡
            balance_by_difficulty: 按难度平衡
            problem_types: 问题类型列表（可选）

        Returns:
            List[TrainingSampleV2]: 生成的样本列表
        """
        samples = []

        if balance_by_category:
            categories = list(self.kb.category_index.keys())
            samples_per_category = total_samples // len(categories) if categories else total_samples

            for category in categories:
                method_ids = self.kb.category_index.get(category, [])

                for method_id in method_ids:
                    method = self.kb.get_method(method_id)
                    if method is None:
                        continue

                    # 为每个难度生成样本
                    if balance_by_difficulty:
                        for diff in range(1, 6):
                            if len(samples) >= total_samples:
                                break
                            problem = self._generate_problem_for_method(method, diff)
                            sample = self.generate_sample(
                                problem, method_id, None, diff
                            )
                            samples.append(sample)
                    else:
                        problem = self._generate_problem_for_method(method, method.difficulty)
                        sample = self.generate_sample(
                            problem, method_id, None, method.difficulty
                        )
                        samples.append(sample)

                    if len(samples) >= total_samples:
                        break

                if len(samples) >= total_samples:
                    break
        else:
            # 随机生成
            method_ids = list(self.kb.methods.keys())
            for _ in range(total_samples):
                method_id = random.choice(method_ids)
                difficulty = random.randint(1, 5)
                method = self.kb.get_method(method_id)
                problem = self._generate_problem_for_method(method, difficulty)
                sample = self.generate_sample(problem, method_id, None, difficulty)
                samples.append(sample)

        return samples[:total_samples]

    def _generate_method_selection(
        self,
        problem: str,
        method: Method,
        problem_type: Optional[str]
    ) -> str:
        """生成方法选择推理过程"""
        # 提取特征
        features = []
        for app in method.applicability:
            keywords = app.get('keywords', [])
            for kw in keywords:
                if kw.lower() in problem.lower():
                    features.append(kw)

        if features:
            feature_str = f"包含{'、'.join(features[:3])}等特征"
        else:
            feature_str = method.applicability[0].get('condition', '符合方法适用条件') if method.applicability else '符合方法适用条件'

        # 构建选择推理
        if problem_type:
            selection = f"本题属于{problem_type}类型，观察到{feature_str}，选择使用{method.name}（{method.method_id}）。"
        else:
            selection = f"观察到题目{feature_str}，符合{method.name}的适用条件，因此选择该方法。"

        # 添加方法描述
        if self.config.get('include_method_description', True):
            selection += f"\n【{method.name}】{method.description.strip()}"

        return selection

    def _generate_solution_steps(
        self,
        problem: str,
        method: Method,
        difficulty: int
    ) -> List[str]:
        """生成解题步骤"""
        # 从方法模板获取步骤
        template_steps = method.template.get('steps', [])
        steps = []

        for i, step in enumerate(template_steps):
            # 生成具体步骤内容
            step_content = self._generate_step_content(step, difficulty, i)
            steps.append(step_content)

        # 添加技巧提示
        if self.config.get('include_common_tricks', True):
            tricks = method.template.get('common_tricks', [])
            if tricks:
                steps.append(f"[技巧提示] {random.choice(tricks)}")

        # 添加注意事项
        if self.config.get('include_pitfall_warnings', True):
            pitfalls = method.template.get('pitfall_warnings', [])
            if pitfalls:
                steps.append(f"[注意事项] {random.choice(pitfalls)}")

        return steps

    def _generate_step_content(
        self,
        step_template: str,
        difficulty: int,
        step_index: int
    ) -> str:
        """生成步骤的具体内容"""
        # 根据步骤关键词生成具体内容
        templates = {
            '识别': f"分析题目，发现关键特征，难度系数{difficulty}",
            '观察': f"观察表达式结构，发现{['对称', '重复', '周期'][step_index % 3]}模式",
            '选择': f"根据特征，选择合适的{['变量替换', '参数设定', '方法组合'][step_index % 3]}",
            '设': f"设 t = {difficulty}x，引入辅助变量简化表达式",
            '引入': f"引入新参数{['a', 'b', 't'][step_index % 3]}，建立辅助关系",
            '转化': f"将原表达式转化为关于新变量的形式",
            '变换': f"利用{['代数变换', '几何变换', '三角变换'][step_index % 3]}，简化问题",
            '求解': f"求解变换后的方程，得到中间结果 t = {difficulty * 2}",
            '计算': f"计算{['数值', '表达式', '关系'][step_index % 3]}，得到关键数据",
            '回代': f"将中间结果回代，得到原问题的解",
            '验证': f"检验结果是否满足所有条件，确认解的有效性",
            '检验': f"检查边界条件和特殊情况，排除无效解"
        }

        for keyword, template in templates.items():
            if keyword in step_template:
                return f"{step_template}: {template}"

        # 默认内容
        return f"{step_template}: 执行第{step_index + 1}步操作"

    def _inject_method_into_steps(
        self,
        problem: str,
        steps: List[str],
        method: Method
    ) -> Tuple[List[str], List[MethodAnnotation]]:
        """将方法注入到步骤中"""
        annotations = []
        injected_steps = []

        # 方法选择标注
        selection_annotation = MethodAnnotation(
            method_id=method.method_id,
            method_name=method.name,
            step_index=-1,
            step_description='',
            annotation_type='selection',
            reasoning=f"根据问题特征选择{method.name}"
        )
        annotations.append(selection_annotation)

        # 步骤注入
        for i, step in enumerate(steps):
            # 应用标注
            annotation = MethodAnnotation(
                method_id=method.method_id,
                method_name=method.name,
                step_index=i,
                step_description=step,
                annotation_type='application',
                reasoning=f"执行{method.name}第{i+1}步"
            )
            annotations.append(annotation)

            # 添加方法标注
            step_type = self._classify_step(step)
            injected_step = f"[{step_type}] {step}"
            injected_steps.append(injected_step)

        return injected_steps, annotations

    def _extract_steps_from_solution(self, solution: str) -> List[str]:
        """从原始解中提取步骤"""
        import re
        # 按换行或句号分割
        paragraphs = re.split(r'\n+|(?<=[。！？])', solution)
        return [p.strip() for p in paragraphs if p.strip()]

    def _generate_final_answer(
        self,
        steps: List[str],
        method: Method,
        difficulty: int
    ) -> str:
        """生成最终答案"""
        # 从步骤中查找答案
        for step in steps:
            if '答案' in step or '结果' in step or '解为' in step:
                return step

        # 生成默认答案
        if self.config.get('answer_format') == 'detailed':
            return f"最终答案：x = {difficulty}, y = {difficulty * 2}（经{method.name}求解得到）"
        else:
            return f"x = {difficulty}, y = {difficulty * 2}"

    def _select_k_methods(
        self,
        applicable_methods: List[Tuple[Method, float]],
        k: int
    ) -> List[str]:
        """选择K个不同方法"""
        if not applicable_methods:
            # 使用所有方法
            all_methods = list(self.kb.methods.keys())
            return random.sample(all_methods, min(k, len(all_methods)))

        # 按分数排序，选择前N个
        sorted_methods = sorted(applicable_methods, key=lambda x: x[1], reverse=True)
        method_ids = [m.method_id for m, _ in sorted_methods[:k]]

        # 补充随机方法
        if len(method_ids) < k:
            remaining_ids = list(set(self.kb.methods.keys()) - set(method_ids))
            additional = random.sample(remaining_ids, min(k - len(method_ids), len(remaining_ids)))
            method_ids.extend(additional)

        return method_ids

    def _select_best_method(
        self,
        problem: str,
        problem_type: Optional[str]
    ) -> str:
        """选择最佳方法"""
        applicable = self.kb.get_applicable_methods(problem, problem_type or 'general')

        if applicable:
            best_method, score = applicable[0]
            return best_method.method_id

        # 随机选择
        method_ids = list(self.kb.methods.keys())
        return random.choice(method_ids) if method_ids else 'ALG_001'

    def _generate_diverse_variant(
        self,
        problem: str,
        method_id: str,
        problem_type: Optional[str],
        difficulty: int,
        variant_idx: int
    ) -> TrainingSampleV2:
        """生成多样性变体样本"""
        # 调整难度
        adjusted_diff = max(1, min(5, difficulty + (variant_idx % 3 - 1)))

        sample = self.generate_sample(problem, method_id, problem_type, adjusted_diff)
        sample.sample_id = f"{sample.sample_id}_v{variant_idx}"
        return sample

    def _generate_step_variant(
        self,
        problem: str,
        method_id: str,
        problem_type: Optional[str],
        difficulty: int,
        variant_idx: int
    ) -> TrainingSampleV2:
        """生成步骤表述变体"""
        method = self.kb.get_method(method_id)
        if method is None:
            method_id = 'ALG_001'
            method = self.kb.get_method(method_id)

        # 生成方法选择（带变体标记）
        method_selection = self._generate_method_selection(problem, method, problem_type)
        method_selection += f"\n[变体{variant_idx}] 采用不同的步骤表述方式"

        # 生成不同表述的步骤
        base_steps = method.template.get('steps', [])
        variant_steps = []

        # 步骤表述变体模板
        variant_templates = [
            ['首先', '然后', '接着', '最后'],
            ['第一步', '第二步', '第三步', '第四步'],
            ['分析阶段', '处理阶段', '求解阶段', '验证阶段'],
            ['初步', '深入', '核心', '收尾']
        ]
        template_idx = variant_idx % len(variant_templates)

        for i, step in enumerate(base_steps[:4]):
            prefix = variant_templates[template_idx][i] if i < 4 else f"第{i+1}步"
            variant_steps.append(f"{prefix}: {step}")

        # 添加技巧和注意事项
        if self.config.get('include_common_tricks', True):
            tricks = method.template.get('common_tricks', [])
            if tricks:
                variant_steps.append(f"[技巧] {tricks[variant_idx % len(tricks)]}")

        # 生成答案
        final_answer = self._generate_final_answer(variant_steps, method, difficulty)

        sample_id = self._generate_sample_id(problem, method_id, self._generated_count)
        self._generated_count += 1

        return TrainingSampleV2(
            problem=problem,
            method_selection=method_selection,
            solution_steps=variant_steps,
            final_answer=final_answer,
            method_id=method_id,
            method_name=method.name,
            problem_type=problem_type,
            difficulty=difficulty,
            sample_id=f"{sample_id}_stepv{variant_idx}"
        )

    def _classify_step(self, step: str) -> str:
        """分类步骤类型"""
        step_lower = step.lower()

        if any(kw in step_lower for kw in ['识别', '观察', '分析']):
            return '分析'
        elif any(kw in step_lower for kw in ['选择', '设', '引入', '构造']):
            return '构造'
        elif any(kw in step_lower for kw in ['转化', '变换', '化简']):
            return '转换'
        elif any(kw in step_lower for kw in ['求解', '计算', '推导']):
            return '计算'
        elif any(kw in step_lower for kw in ['回代', '返回', '得出']):
            return '回代'
        elif any(kw in step_lower for kw in ['验证', '检验', '确认']):
            return '验证'
        elif any(kw in step_lower for kw in ['技巧', '提示']):
            return '技巧'
        elif any(kw in step_lower for kw in ['注意', '警告']):
            return '注意'
        else:
            return '操作'

    def _generate_sample_id(self, problem: str, method_id: str, count: int) -> str:
        """生成样本ID"""
        # 使用问题哈希确保唯一性
        problem_hash = hashlib.md5(problem.encode()).hexdigest()[:8]
        return f"{method_id}_{problem_hash}_{count:04d}"

    def _annotation_to_dict(self, annotation: MethodAnnotation) -> Dict:
        """将标注转换为字典"""
        return {
            'method_id': annotation.method_id,
            'method_name': annotation.method_name,
            'step_index': annotation.step_index,
            'step_description': annotation.step_description,
            'annotation_type': annotation.annotation_type,
            'reasoning': annotation.reasoning
        }

    def _generate_problem_for_method(self, method: Method, difficulty: int) -> str:
        """为方法生成对应问题"""
        # 内置问题模板
        templates = {
            '方程求解': [
                f"求解方程 x^2 + {difficulty*2}x + {difficulty*3} = 0",
                f"已知x + 1/x = {difficulty+3}, 求x的值"
            ],
            '不等式证明': [
                f"证明对于所有x > 0, x + 1/x ≥ 2",
                f"设a, b > 0, 证明 a^2 + b^2 ≥ {difficulty}ab"
            ],
            '函数最值': [
                f"求函数 f(x) = x^2 - {difficulty*2}x + {difficulty*5} 的最小值",
                f"求 {difficulty}x + {difficulty*2}y 在约束 x + y = {difficulty*10} 下的最大值"
            ],
            '几何证明': [
                f"在三角形ABC中, AB = AC = {difficulty}, 证明∠B = ∠C",
                f"证明面积为{difficulty*difficulty}的正方形边长为{difficulty}"
            ],
            '整除性': [
                f"证明 {difficulty*100 + 7} 被 {difficulty + 1} 整除",
                f"求{difficulty*100 + 37}除以{difficulty + 3}的余数"
            ],
            '计数问题': [
                f"计算将{difficulty*2}个不同元素排列的方法数",
                f"从{difficulty*10}个元素中选择{difficulty}个的组合数"
            ]
        }

        # 从方法适用条件获取问题类型
        problem_types = []
        for app in method.applicability:
            problem_types.extend(app.get('problem_types', []))

        if problem_types:
            problem_type = problem_types[0]
            if problem_type in templates:
                return random.choice(templates[problem_type])

        # 默认问题
        return f"使用{method.name}求解问题，难度等级{difficulty}"

    def _select_from_distribution(self, distribution: Dict[int, float]) -> int:
        """根据分布选择数值"""
        values = list(distribution.keys())
        weights = list(distribution.values())
        return random.choices(values, weights=weights)[0]

    def save_samples(
        self,
        samples: List[TrainingSampleV2],
        path: str,
        format: str = 'json'
    ):
        """保存样本到文件

        Args:
            samples: 样本列表
            path: 输出路径
            format: 输出格式 ('json', 'jsonl')
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        data = [s.to_dict() for s in samples]

        if format == 'jsonl':
            with open(path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"保存样本: {len(samples)} 条 -> {path}")

    @classmethod
    def load_samples(cls, path: str) -> List[TrainingSampleV2]:
        """从文件加载样本"""
        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)

        return [TrainingSampleV2(**item) for item in data]


def create_training_samples(
    problems: List[Dict],
    kb: KnowledgeBase,
    pass_k: Optional[int] = None,
    config: Optional[Dict] = None
) -> List[TrainingSampleV2]:
    """便捷函数：创建训练样本

    Args:
        problems: 问题列表
        kb: 知识库
        pass_k: Pass@K数量
        config: 配置

    Returns:
        List[TrainingSampleV2]: 训练样本列表
    """
    generator = SampleGenerator(kb, config=config)

    if pass_k:
        samples = generator.generate_batch(
            problems, pass_k=pass_k
        )
    else:
        samples = generator.generate_batch(problems, samples_per_problem=1)

    return samples