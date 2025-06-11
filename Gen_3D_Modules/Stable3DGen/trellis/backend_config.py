# trellis/backend_config.py
from typing import *
import os
import logging
import importlib

# Global variables
BACKEND = 'spconv'  # Default sparse backend
DEBUG = False       # Debug mode flag
ATTN = 'xformers' # Default attention backend
SPCONV_ALGO = 'implicit_gemm'  # Default algorithm

def get_spconv_algo() -> str:
    """Get current spconv algorithm."""
    global SPCONV_ALGO
    return SPCONV_ALGO

def set_spconv_algo(algo: Literal['implicit_gemm', 'native', 'auto']) -> bool:
    """Set spconv algorithm with validation."""
    global SPCONV_ALGO
    
    if algo not in ['implicit_gemm', 'native', 'auto']:
        logger.warning(f"Invalid spconv algorithm: {algo}. Must be 'implicit_gemm', 'native', or 'auto'")
        return False
        
    SPCONV_ALGO = algo
    os.environ['SPCONV_ALGO'] = algo
    logger.info(f"Set spconv algorithm to: {algo}")
    return True

logger = logging.getLogger(__name__)

def _try_import_xformers() -> bool:
    try:
        import xformers.ops
        return True
    except ImportError:
        return False

def _try_import_flash_attn() -> bool:
    try:
        import flash_attn
        return True
    except ImportError:
        return False

def _try_import_sageattention() -> bool:
    try:
        import torch.nn.functional as F
        from sageattention import sageattn
        F.scaled_dot_product_attention = sageattn
        #import sageattention
        return True
    except ImportError:
        return False

def _try_import_spconv() -> bool:
    try:
        import spconv
        return True
    except ImportError:
        return False

def _try_import_torchsparse() -> bool:
    try:
        import torchsparse
        return True
    except ImportError:
        return False

def get_available_backends() -> Dict[str, bool]:
    """Return dict of available attention backends and their status"""
    return {
        'xformers': _try_import_xformers(),
        'flash_attn': _try_import_flash_attn(),
        'sage': _try_import_sageattention(),
        'naive': True,
        'sdpa': True  # Always available with PyTorch >= 2.0
    }

def get_available_sparse_backends() -> Dict[str, bool]:
    """Return dict of available sparse backends and their status"""
    return {
        'spconv': _try_import_spconv(),
        'torchsparse': _try_import_torchsparse()
    }

def get_attention_backend() -> str:
    """Get current attention backend"""
    global ATTN
    return ATTN

def get_sparse_backend() -> str:
    """Get current sparse backend"""
    global BACKEND
    return BACKEND

def get_debug_mode() -> bool:
    """Get current debug mode status"""
    global DEBUG
    return DEBUG

def __from_env():
    """Initialize settings from environment variables"""
    global BACKEND
    global DEBUG
    global ATTN
    
    env_sparse_backend = os.environ.get('SPARSE_BACKEND')
    env_sparse_debug = os.environ.get('SPARSE_DEBUG')
    env_sparse_attn = os.environ.get('SPARSE_ATTN_BACKEND')
    
    if env_sparse_backend is not None and env_sparse_backend in ['spconv', 'torchsparse']:
        BACKEND = env_sparse_backend
    if env_sparse_debug is not None:
        DEBUG = env_sparse_debug == '1'
    if env_sparse_attn is not None and env_sparse_attn in ['xformers', 'flash_attn', 'sage', 'sdpa', 'naive']:
        ATTN = env_sparse_attn
        os.environ['SPARSE_ATTN_BACKEND'] = env_sparse_attn
        os.environ['ATTN_BACKEND'] = env_sparse_attn
        
    logger.info(f"[SPARSE] Backend: {BACKEND}, Attention: {ATTN}")

def set_backend(backend: Literal['spconv', 'torchsparse']) -> bool:
    """Set sparse backend with validation"""
    global BACKEND
    
    backend = backend.lower().strip()
    logger.info(f"Setting sparse backend to: {backend}")

    if backend == 'spconv':
        try:
            import spconv
            BACKEND = 'spconv'
            os.environ['SPARSE_BACKEND'] = 'spconv'
            return True
        except ImportError:
            logger.warning("spconv not available")
            return False
            
    elif backend == 'torchsparse':
        try:
            import torchsparse
            BACKEND = 'torchsparse'
            os.environ['SPARSE_BACKEND'] = 'torchsparse'
            return True
        except ImportError:
            logger.warning("torchsparse not available")
            return False
    
    return False

def set_sparse_backend(backend: Literal['spconv', 'torchsparse'], algo: str = None) -> bool:
    """Alias for set_backend for backwards compatibility
    
    Parameters:
        backend: The sparse backend to use
        algo: The algorithm to use (only relevant for spconv backend)
    """
    # Call set_backend first
    result = set_backend(backend)
    
    # If algorithm is provided and backend was set successfully
    if algo is not None and result:
        set_spconv_algo(algo)
        
    return result

def set_debug(debug: bool):
    """Set debug mode"""
    global DEBUG
    DEBUG = debug
    if debug:
        os.environ['SPARSE_DEBUG'] = '1'
    else:
        os.environ['SPARSE_DEBUG'] = '0'

def set_attn(attn: Literal['xformers', 'flash_attn', 'sage', 'sdpa', 'naive']) -> bool:
    """Set attention backend with validation"""
    global ATTN
    
    attn = attn.lower().strip()
    logger.info(f"Setting attention backend to: {attn}")

    if attn == 'xformers' and _try_import_xformers():
        ATTN = 'xformers'
        os.environ['SPARSE_ATTN_BACKEND'] = 'xformers'
        os.environ['ATTN_BACKEND'] = 'xformers'
        return True
        
    elif attn == 'flash_attn' and _try_import_flash_attn():
        ATTN = 'flash_attn'
        os.environ['SPARSE_ATTN_BACKEND'] = 'flash_attn'
        os.environ['ATTN_BACKEND'] = 'flash_attn'
        return True
        
    elif attn == 'sage' and _try_import_sageattention():
        ATTN = 'sage'
        os.environ['SPARSE_ATTN_BACKEND'] = 'sage'
        os.environ['ATTN_BACKEND'] = 'sage'
        return True
        
    elif attn == 'sdpa':
        ATTN = 'sdpa'
        os.environ['SPARSE_ATTN_BACKEND'] = 'sdpa'
        os.environ['ATTN_BACKEND'] = 'sdpa'
        return True
    
    elif attn == 'naive':
        ATTN = 'naive'
        os.environ['SPARSE_ATTN_BACKEND'] = 'naive'
        os.environ['ATTN_BACKEND'] = 'naive'
        return True
        

    logger.warning(f"Attention backend {attn} not available")
    return False

# Add alias for backwards compatibility 
def set_attention_backend(backend: Literal['xformers', 'flash_attn', 'sage', 'sdpa']) -> bool:
    """Alias for set_attn for backwards compatibility"""
    return set_attn(backend)

# Initialize from environment variables on module import
__from_env()
