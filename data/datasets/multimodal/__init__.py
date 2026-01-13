"""
CapibaraGPT-v2 Multimodal Datasets
Multimodal datasets: audio, vision, conversation
"""

from . import vision_datasets
from . import emotional_audio_datasets
from . import multimodal_conversation_datasets

__all__ = [
    'multimodal_conversation_datasets',
    'emotional_audio_datasets',
    'vision_datasets'
]
