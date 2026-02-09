"""RAM-streamed MoE inference for Qwen3-Coder-30B-A3B on consumer hardware."""

from .loader import load_model, LayerPlacement
from .cache import ExpertCache, ExpertCacheManager
from .model import StreamingMoEModel

__all__ = [
    "load_model",
    "LayerPlacement",
    "ExpertCache",
    "ExpertCacheManager",
    "StreamingMoEModel",
]

__version__ = "0.1.0"
