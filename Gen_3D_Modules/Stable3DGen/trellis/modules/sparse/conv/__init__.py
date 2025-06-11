import os
import logging
from trellis.backend_config import get_sparse_backend, get_spconv_algo

BACKEND = get_sparse_backend()
SPCONV_ALGO = get_spconv_algo()

def __from_env():
    import os
        
    global SPCONV_ALGO
    env_spconv_algo = os.environ.get('SPCONV_ALGO')
    if env_spconv_algo is not None and env_spconv_algo in ['auto', 'implicit_gemm', 'native']:
        SPCONV_ALGO = env_spconv_algo
    print(f"[SPARSE][CONV] spconv algo: {SPCONV_ALGO}")
        

__from_env()

if BACKEND == 'torchsparse':
    from .conv_torchsparse import *
elif BACKEND == 'spconv':
    from .conv_spconv import *
    
__all__ = [
    "SparseConv3d",
    "SparseInverseConv3d",
]

