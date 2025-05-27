from .full_attn import *
from .serialized_attn import *
from .windowed_attn import *
from .modules import *
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
ATTN = get_attention_backend()
DEBUG = get_debug_mode()

def __from_env():
    """Read current backend configuration"""
    #global ATTN
    global ATTN
    global DEBUG
    
    # Get current settings from central config
    #ATTN = 
    ATTN = get_attention_backend()
    DEBUG = get_debug_mode()
    
    print(f"[ATTENTION] sparse backend: {ATTN}")