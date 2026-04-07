"""数据整理器"""

from typing import List, Dict
import torch


class MethodologyCollator:
    """方法论数据整理器

    将样本整理成模型输入格式。
    """

    def __init__(self, tokenizer, max_length: int = 4096):
        """初始化整理器

        Args:
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, samples: List[Dict]) -> Dict:
        """整理样本"""
        batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': []
        }

        for sample in samples:
            # 构建输入文本
            input_text = self._build_input_text(sample)
            target_text = self._build_target_text(sample)

            # Tokenize
            inputs = self.tokenizer(
                input_text,
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            )

            targets = self.tokenizer(
                target_text,
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            )

            # 合并
            input_ids = torch.cat([
                inputs['input_ids'][0],
                targets['input_ids'][0]
            ])

            attention_mask = torch.cat([
                inputs['attention_mask'][0],
                targets['attention_mask'][0]
            ])

            # 标签：输入部分为-100
            labels = torch.cat([
                torch.full_like(inputs['input_ids'][0], -100),
                targets['input_ids'][0]
            ])

            batch['input_ids'].append(input_ids)
            batch['attention_mask'].append(attention_mask)
            batch['labels'].append(labels)

        # Padding
        max_len = max(len(ids) for ids in batch['input_ids'])

        for i in range(len(samples)):
            pad_length = max_len - len(batch['input_ids'][i])

            batch['input_ids'][i] = torch.cat([
                batch['input_ids'][i],
                torch.zeros(pad_length, dtype=torch.long)
            ])
            batch['attention_mask'][i] = torch.cat([
                batch['attention_mask'][i],
                torch.zeros(pad_length, dtype=torch.long)
            ])
            batch['labels'][i] = torch.cat([
                batch['labels'][i],
                torch.full((pad_length,), -100, dtype=torch.long)
            ])

        # Stack
        batch['input_ids'] = torch.stack(batch['input_ids'])
        batch['attention_mask'] = torch.stack(batch['attention_mask'])
        batch['labels'] = torch.stack(batch['labels'])

        return batch

    def _build_input_text(self, sample: Dict) -> str:
        """构建输入文本"""
        candidates_str = "\n".join([
            f"{i+1}. {m.get('method_name', '')}（适用性：{m.get('applicability_score', 0):.2f}）"
            for i, m in enumerate(sample.get('candidate_methods', []))
        ])

        return f"""【问题】
{sample.get('problem', '')}

【题型】
{sample.get('problem_type', '')}

【候选方法】
{candidates_str}

请分析各方法的适用性，选择最合适的方法并给出理由，然后用该方法解答问题。
"""

    def _build_target_text(self, sample: Dict) -> str:
        """构建目标文本"""
        steps_str = "\n".join(sample.get('solution_steps', []))

        return f"""【方法选择】
选中的方法：{sample.get('selected_method', '')}

【选择理由】
{sample.get('selection_reasoning', '')}

【解答】
{steps_str}

【反思】
{sample.get('reflection', '')}
"""