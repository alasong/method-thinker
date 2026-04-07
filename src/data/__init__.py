"""数据处理模块"""

from .dataset import MethodologyDataset
from .data_generator import DataGenerator, TrainingSample
from .method_injector import MethodInjector, MethodAnnotation, create_annotated_dataset
from .sample_generator import SampleGenerator, TrainingSampleV2, create_training_samples
from .aime_loader import AIMELoader, AIMEProblem, create_aime_dataset, DEFAULT_AIME_PATH

# Collator requires torch, import lazily
try:
    from .collator import MethodologyCollator
    _collator_available = True
except ImportError:
    _collator_available = False

__all__ = [
    'MethodologyDataset',
    'MethodologyCollator',
    'DataGenerator',
    'TrainingSample',
    'MethodInjector',
    'MethodAnnotation',
    'create_annotated_dataset',
    'SampleGenerator',
    'TrainingSampleV2',
    'create_training_samples',
    'AIMELoader',
    'AIMEProblem',
    'create_aime_dataset',
    'DEFAULT_AIME_PATH'
]

def __getattr__(name):
    if name == 'MethodologyCollator':
        if not _collator_available:
            raise ImportError(
                "MethodologyCollator requires torch. "
                "Install torch with: pip install torch"
            )
        return MethodologyCollator
    raise AttributeError(f"module {__name__} has no attribute {name}")