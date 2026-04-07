"""知识库增量更新"""

from typing import Dict, List, Optional
from .knowledge_base import KnowledgeBase, Method


class IncrementalKBUpdater:
    """知识库增量更新器

    处理新增方法的合并、替换和冲突解决。
    """

    def __init__(self, kb: KnowledgeBase):
        """初始化更新器

        Args:
            kb: 要更新的知识库
        """
        self.kb = kb
        self.update_history = []

    def update(self, new_methods: List[Method]) -> Dict:
        """增量更新知识库

        Args:
            new_methods: 新方法列表

        Returns:
            Dict: 更新统计
        """
        stats = {
            'added': 0,
            'replaced': 0,
            'merged': 0,
            'skipped': 0
        }

        for new_method in new_methods:
            # 查找相似方法
            similar = self.kb.find_similar_methods(new_method, threshold=0.8)

            if similar:
                existing = similar[0]
                # 比较质量决定替换还是合并
                if self._should_replace(new_method, existing):
                    self._replace_method(existing.method_id, new_method)
                    stats['replaced'] += 1
                else:
                    self._merge_method(existing, new_method)
                    stats['merged'] += 1
            else:
                # 新方法，直接添加
                self.kb.add_method(new_method)
                stats['added'] += 1

            self.update_history.append({
                'method_id': new_method.method_id,
                'action': 'replace' if similar else 'add'
            })

        return stats

    def _should_replace(self, new_method: Method, existing: Method) -> bool:
        """判断是否应该替换"""
        # 比较频率和描述完整性
        new_score = new_method.frequency + (0.1 if len(new_method.description) > 100 else 0)
        existing_score = existing.frequency + (0.1 if len(existing.description) > 100 else 0)
        return new_score > existing_score

    def _replace_method(self, old_id: str, new_method: Method):
        """替换方法"""
        # 删除旧方法的索引
        old_method = self.kb.methods.get(old_id)
        if old_method:
            del self.kb.methods[old_id]

        # 添加新方法
        new_method.method_id = old_id  # 保持ID
        self.kb.add_method(new_method)

    def _merge_method(self, existing: Method, new_method: Method):
        """合并方法"""
        # 合并适用条件 - 使用JSON字符串作为key避免hash问题
        import json
        existing_conditions = {
            json.dumps(app, sort_keys=True) for app in existing.applicability
        }
        for app in new_method.applicability:
            app_key = json.dumps(app, sort_keys=True)
            if app_key not in existing_conditions:
                existing.applicability.append(app)

        # 合并例子
        for ex in new_method.examples:
            if ex not in existing.examples:
                existing.examples.append(ex)

        # 合并相关方法
        for rm in new_method.related_methods:
            if rm not in existing.related_methods:
                existing.related_methods.append(rm)

        # 更新频率（取平均）
        existing.frequency = (existing.frequency + new_method.frequency) / 2

    def detect_conflicts(self) -> List[Dict]:
        """检测知识库中的冲突"""
        conflicts = []

        for method in self.kb.methods.values():
            # 检查相关方法是否存在
            for related_id in method.related_methods:
                if related_id not in self.kb.methods:
                    conflicts.append({
                        'type': 'missing_related',
                        'method_id': method.method_id,
                        'missing': related_id
                    })

        return conflicts

    def prune_low_quality(self, threshold: float = 0.3) -> int:
        """清理低质量方法

        Args:
            threshold: 频率阈值

        Returns:
            int: 删除的方法数量
        """
        to_remove = [
            mid for mid, m in self.kb.methods.items()
            if m.frequency < threshold and len(m.examples) < 2
        ]

        for mid in to_remove:
            del self.kb.methods[mid]

        return len(to_remove)

    def get_update_summary(self) -> Dict:
        """获取更新摘要"""
        return {
            'total_methods': len(self.kb.methods),
            'total_updates': len(self.update_history),
            'categories': {
                cat: len(methods)
                for cat, methods in self.kb.category_index.items()
            }
        }