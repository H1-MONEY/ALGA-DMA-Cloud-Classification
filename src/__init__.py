"""
ALGA-DMA Cloud Classification Package

This package implements the ALGA (Adaptive Local-Global Attention) and 
DMA (Dynamic Multi-Head Attention) model for cloud classification.

Author: Your Name
License: MIT
"""

from .model import ALGADMAModel, create_model
from .data_utils import (
    get_data_transforms, 
    create_data_loaders, 
    generate_augmented_images,
    CLOUD_CATEGORIES
)

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    'ALGADMAModel',
    'create_model', 
    'get_data_transforms',
    'create_data_loaders',
    'generate_augmented_images',
    'CLOUD_CATEGORIES'
]