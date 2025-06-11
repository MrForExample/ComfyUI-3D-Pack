import os
import logging
from typing import Literal
from trellis.backend_config import (
    get_attention_backend,
    get_debug_mode,
)
import logging

logger = logging.getLogger(__name__)

#ATTN = get_attention_backend()
BACKEND = get_attention_backend()
DEBUG = get_debug_mode()

def __from_env():
    """Read current backend configuration"""
    #global ATTN
    global BACKEND
    global DEBUG
    
    # Get current settings from central config
    #ATTN = 
    BACKEND = get_attention_backend()
    DEBUG = get_debug_mode()
    
    print(f"[ATTENTION] Using backend: {BACKEND}")

from .modules import MultiHeadAttention, RotaryPositionEmbedder
from .full_attn import scaled_dot_product_attention

__all__ = [
    'scaled_dot_product_attention',
    'BACKEND',
    'DEBUG',
    'MultiHeadAttention',
    'RotaryPositionEmbedder'
]
