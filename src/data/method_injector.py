"""方法论注入器

将方法论步骤嵌入到解题过程中，生成带方法标注的训练数据。
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re

from ..kb.knowledge_base import KnowledgeBase, Method


@dataclass
class MethodAnnotation:
    """方法标注"""
    method_id: str
    method_name: str
    step_index: int
    step_description: str
    annotation_type: str  # 'selection', 'application', 'transition', 'verification'
    reasoning: str


class MethodInjector:
    """方法论注入器

    将方法论步骤嵌入解题过程，生成结构化的训练数据。

    Attributes:
        kb: 方法论知识库
        injection_config: 注入配置
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        injection_config: Optional[Dict] = None
    ):
        """初始化方法注入器

        Args:
            kb: 方法论知识库
            injection_config: 注入配置（可选）
        """
        self.kb = kb
        self.injection_config = injection_config or {
            'include_method_description': True,
            'include_common_tricks': True,
            'include_pitfall_warnings': True,
            'annotate_step_type': True,
            'max_reasoning_length': 200
        }

        # 标注模板
        self._annotation_templates = {
            'selection': [
                "【方法选择】{reasoning}",
                "根据{feature}，选择使用{method_name}",
                "观察到{observation}，因此采用{method_name}"
            ],
            'application': [
                "【方法应用】{step}",
                "执行{method_name}的步骤{step_num}: {step}",
                "{step} （{method_name}核心步骤）"
            ],
            'transition': [
                "【关键转换】{transformation}",
                "利用{method_name}的性质，{transformation}",
                "通过{method_name}，将{from_state}转化为{to_state}"
            ],
            'verification': [
                "【结果验证】{verification}",
                "检验{result}是否满足{condition}",
                "验证{method_name}的应用正确性"
            ]
        }

    def inject_method(
        self,
        problem: str,
        raw_solution: str,
        target_method_id: str,
        problem_type: Optional[str] = None
    ) -> Tuple[str, List[MethodAnnotation]]:
        """将方法注入到解题过程中

        Args:
            problem: 问题描述
            raw_solution: 原始解题过程（未标注）
            target_method_id: 目标方法ID
            problem_type: 问题类型（可选）

        Returns:
            Tuple[str, List[MethodAnnotation]]: (注入后的解, 标注列表)
        """
        method = self.kb.get_method(target_method_id)
        if method is None:
            return raw_solution, []

        # 获取方法的步骤和配置
        steps = method.template.get('steps', [])
        common_tricks = method.template.get('common_tricks', [])
        pitfalls = method.template.get('pitfall_warnings', [])

        annotations = []
        injected_solution_parts = []

        # 1. 方法选择标注
        selection_annotation = self._create_selection_annotation(
            method, problem, problem_type
        )
        annotations.append(selection_annotation)

        if self.injection_config.get('include_method_description', True):
            injected_solution_parts.append(
                f"\n【{method.name}】\n{method.description.strip()}\n"
            )

        # 2. 步骤注入
        injected_solution_parts.append("\n解题过程：")

        # 将原始解分割成段落
        solution_paragraphs = self._split_solution(raw_solution)

        # 按步骤注入
        for i, (step, paragraph) in enumerate(
            zip(steps, solution_paragraphs[:len(steps)])
        ):
            # 应用标注
            app_annotation = MethodAnnotation(
                method_id=method.method_id,
                method_name=method.name,
                step_index=i,
                step_description=step,
                annotation_type='application',
                reasoning=f"执行{method.name}第{i+1}步"
            )
            annotations.append(app_annotation)

            # 注入步骤文本
            step_injected = self._inject_step(
                step, paragraph, i, method
            )
            injected_solution_parts.append(step_injected)

        # 处理剩余段落
        remaining = solution_paragraphs[len(steps):]
        if remaining:
            injected_solution_parts.append("\n补充说明：")
            injected_solution_parts.extend(remaining)

        # 3. 技巧提示注入
        if common_tricks and self.injection_config.get('include_common_tricks', True):
            trick_annotation = MethodAnnotation(
                method_id=method.method_id,
                method_name=method.name,
                step_index=-1,
                step_description='',
                annotation_type='transition',
                reasoning=random_choice(common_tricks) if common_tricks else ''
            )
            annotations.append(trick_annotation)
            injected_solution_parts.append(
                f"\n【技巧提示】{trick_annotation.reasoning}"
            )

        # 4. 注意事项注入
        if pitfalls and self.injection_config.get('include_pitfall_warnings', True):
            pitfall_annotation = MethodAnnotation(
                method_id=method.method_id,
                method_name=method.name,
                step_index=-1,
                step_description='',
                annotation_type='verification',
                reasoning=random_choice(pitfalls) if pitfalls else ''
            )
            annotations.append(pitfall_annotation)
            injected_solution_parts.append(
                f"\n【注意事项】{pitfall_annotation.reasoning}"
            )

        injected_solution = "\n".join(injected_solution_parts)
        return injected_solution, annotations

    def inject_methods_chain(
        self,
        problem: str,
        raw_solution: str,
        method_chain: List[str]
    ) -> Tuple[str, List[MethodAnnotation]]:
        """注入方法链（多方法组合）

        Args:
            problem: 问题描述
            raw_solution: 原始解题过程
            method_chain: 方法ID链（按应用顺序）

        Returns:
            Tuple[str, List[MethodAnnotation]]: (注入后的解, 标注列表)
        """
        all_annotations = []
        all_parts = []

        # 添加问题分析
        all_parts.append("【问题分析】")
        all_parts.append(f"问题：{problem}\n")

        # 按方法链分段注入
        solution_parts = self._split_by_methods(raw_solution, method_chain)

        for i, (method_id, part) in enumerate(zip(method_chain, solution_parts)):
            method = self.kb.get_method(method_id)
            if method is None:
                all_parts.append(part)
                continue

            # 方法标记
            all_parts.append(f"\n【阶段{i+1}：{method.name}】")

            # 方法选择说明
            if i == 0:
                reasoning = f"首先观察到{self._extract_feature(problem, method)}"
            else:
                prev_method = self.kb.get_method(method_chain[i-1])
                reasoning = f"在上一步{prev_method.name if prev_method else ''}的基础上，发现{self._extract_feature(part, method)}"

            annotation = MethodAnnotation(
                method_id=method_id,
                method_name=method.name,
                step_index=0,
                step_description='',
                annotation_type='selection',
                reasoning=reasoning
            )
            all_annotations.append(annotation)

            all_parts.append(f"选择理由：{reasoning}\n")

            # 注入方法步骤
            injected_part, part_annotations = self._inject_method_part(
                part, method
            )
            all_parts.append(injected_part)
            all_annotations.extend(part_annotations)

            # 方法过渡标注
            if i < len(method_chain) - 1:
                next_method = self.kb.get_method(method_chain[i+1])
                transition_annotation = MethodAnnotation(
                    method_id=method_id,
                    method_name=method.name,
                    step_index=-1,
                    step_description='',
                    annotation_type='transition',
                    reasoning=f"当前方法得到{self._extract_result(injected_part)}, 为下一步{next_method.name if next_method else ''}做准备"
                )
                all_annotations.append(transition_annotation)

        injected_solution = "\n".join(all_parts)
        return injected_solution, all_annotations

    def _create_selection_annotation(
        self,
        method: Method,
        problem: str,
        problem_type: Optional[str]
    ) -> MethodAnnotation:
        """创建方法选择标注"""
        # 生成选择理由
        feature = self._extract_feature(problem, method)

        if problem_type:
            reasoning = f"本题属于{problem_type}类型，观察到{feature}，适合使用{method.name}"
        else:
            reasoning = f"观察到{feature}，符合{method.name}的适用条件"

        reasoning = self._truncate_reasoning(reasoning)

        return MethodAnnotation(
            method_id=method.method_id,
            method_name=method.name,
            step_index=-1,
            step_description='',
            annotation_type='selection',
            reasoning=reasoning
        )

    def _inject_step(
        self,
        step: str,
        paragraph: str,
        step_index: int,
        method: Method
    ) -> str:
        """注入单个步骤"""
        # 步骤标题
        header = f"  步骤{step_index+1}: {step}"

        # 步骤内容（从原文提取或生成）
        content = paragraph if paragraph else self._generate_step_content(step)

        # 步骤类型标注
        if self.injection_config.get('annotate_step_type', True):
            step_type = self._classify_step(step)
            content = f"    [{step_type}] {content}"

        return f"{header}\n{content}"

    def _inject_method_part(
        self,
        content: str,
        method: Method
    ) -> Tuple[str, List[MethodAnnotation]]:
        """注入方法部分"""
        steps = method.template.get('steps', [])
        annotations = []
        parts = []

        paragraphs = self._split_solution(content)

        for i, step in enumerate(steps):
            paragraph = paragraphs[i] if i < len(paragraphs) else ''

            annotation = MethodAnnotation(
                method_id=method.method_id,
                method_name=method.name,
                step_index=i,
                step_description=step,
                annotation_type='application',
                reasoning=f"执行{method.name}核心步骤"
            )
            annotations.append(annotation)

            step_text = self._inject_step(step, paragraph, i, method)
            parts.append(step_text)

        return "\n".join(parts), annotations

    def _split_solution(self, solution: str) -> List[str]:
        """分割解题过程"""
        # 按换行或关键词分割
        paragraphs = re.split(r'\n+|(?<=[。！？])', solution)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_by_methods(
        self,
        solution: str,
        method_chain: List[str]
    ) -> List[str]:
        """按方法链分割解题过程"""
        total_methods = len(method_chain)
        paragraphs = self._split_solution(solution)

        # 简单平均分割
        parts_count = len(paragraphs) // total_methods if total_methods > 0 else len(paragraphs)
        if parts_count < 1:
            parts_count = 1

        parts = []
        for i in range(total_methods):
            start = i * parts_count
            end = start + parts_count if i < total_methods - 1 else len(paragraphs)
            parts.append("\n".join(paragraphs[start:end]))

        return parts

    def _extract_feature(self, text: str, method: Method) -> str:
        """提取方法适用特征"""
        # 从适用条件提取关键词
        features = []
        for app in method.applicability:
            keywords = app.get('keywords', [])
            for kw in keywords:
                if kw.lower() in text.lower():
                    features.append(kw)

        if features:
            return f"包含{'、'.join(features[:3])}等特征"

        # 使用通用特征
        condition = method.applicability[0].get('condition', '') if method.applicability else ''
        return condition if condition else f"符合{method.name}的适用范围"

    def _extract_result(self, solution: str) -> str:
        """提取当前结果"""
        # 查找答案或结果
        patterns = [
            r'答案[是为：]+\s*(.+)',
            r'结果[是为：]+\s*(.+)',
            r'得到\s*(.+)',
            r'求得\s*(.+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, solution)
            if match:
                return match.group(1).strip()[:50]

        return "中间结果"

    def _generate_step_content(self, step: str) -> str:
        """生成步骤内容"""
        # 根据步骤关键词生成
        if "识别" in step or "观察" in step:
            return "分析给定条件，识别关键特征"
        elif "选择" in step or "设" in step:
            return "选择合适的变量替换，设新变量"
        elif "转化" in step or "变换" in step:
            return "将原表达式转化为新形式"
        elif "求解" in step or "计算" in step:
            return "计算并求解问题"
        elif "回代" in step:
            return "将结果回代，得到最终解"
        elif "验证" in step:
            return "验证结果的正确性"
        else:
            return "执行相关操作"

    def _classify_step(self, step: str) -> str:
        """分类步骤类型"""
        if any(kw in step for kw in ['识别', '观察', '分析']):
            return '分析'
        elif any(kw in step for kw in ['选择', '设', '引入', '构造']):
            return '构造'
        elif any(kw in step for kw in ['转化', '变换', '化简']):
            return '转换'
        elif any(kw in step for kw in ['求解', '计算', '推导']):
            return '计算'
        elif any(kw in step for kw in ['回代', '返回', '得出']):
            return '回代'
        elif any(kw in step for kw in ['验证', '检验', '确认']):
            return '验证'
        else:
            return '操作'

    def _truncate_reasoning(self, reasoning: str) -> str:
        """截断推理文本"""
        max_len = self.injection_config.get('max_reasoning_length', 200)
        if len(reasoning) > max_len:
            return reasoning[:max_len-3] + "..."
        return reasoning


def random_choice(items: List[str]) -> str:
    """随机选择（避免导入random的额外开销）"""
    import random
    return random.choice(items) if items else ''


def create_annotated_dataset(
    problems: List[Dict],
    kb: KnowledgeBase,
    injector: Optional[MethodInjector] = None
) -> List[Dict]:
    """创建标注数据集

    Args:
        problems: 问题列表，每项包含 problem, solution, method_used
        kb: 方法论知识库
        injector: 方法注入器（可选）

    Returns:
        List[Dict]: 标注后的数据集
    """
    if injector is None:
        injector = MethodInjector(kb)

    annotated_data = []

    for item in problems:
        problem = item.get('problem', '')
        raw_solution = item.get('solution', '')
        method_id = item.get('method_used', '')
        problem_type = item.get('problem_type', '')

        # 注入方法
        injected_solution, annotations = injector.inject_method(
            problem, raw_solution, method_id, problem_type
        )

        # 构建标注数据
        annotation_dicts = [
            {
                'method_id': a.method_id,
                'method_name': a.method_name,
                'step_index': a.step_index,
                'step_description': a.step_description,
                'annotation_type': a.annotation_type,
                'reasoning': a.reasoning
            }
            for a in annotations
        ]

        annotated_item = {
            'problem_id': item.get('problem_id', ''),
            'problem': problem,
            'problem_type': problem_type,
            'solution': injected_solution,
            'method_used': method_id,
            'annotations': annotation_dicts,
            'difficulty': item.get('difficulty', 3),
            'source': item.get('source', 'annotated'),
            'verified': item.get('verified', False)
        }

        annotated_data.append(annotated_item)

    return annotated_data