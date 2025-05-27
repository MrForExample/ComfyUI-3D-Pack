#sage_attn.py
import os
from typing import Optional
import torch
import torch.nn.functional as F
from sageattention import sageattn
import math

__all__ = ['SageAttention', 'sage_attention']


def enable_sage_attention():
    """
    Enable SageAttention by replacing PyTorch's scaled_dot_product_attention
    with sageattn from the SageAttention library.
    """
    F.scaled_dot_product_attention = sageattn
    return True

def disable_sage_attention():
    """
    Restore PyTorch's original scaled_dot_product_attention function.
    """
    F.scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
    return True