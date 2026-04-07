"""知识库管理模块"""
from .knowledge_base import KnowledgeBase
from .incremental_updater import IncrementalKBUpdater

__all__ = ['KnowledgeBase', 'IncrementalKBUpdater']