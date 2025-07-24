#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

import torch
import numpy as np
from partcrafter_src.models.autoencoders.autoencoder_kl_triposg import TripoSGVAEModel

def test_both_sampling_versions():
    """Quick test of both sampling versions"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Testing on device: {device}")
    
    # Create model
    model = TripoSGVAEModel(
        in_channels=3,
        latent_channels=64,
        num_attention_heads=8,
        width_encoder=512,
        width_decoder=1024,
        num_layers_encoder=8,
        num_layers_decoder=16,
    ).to(device)
    
    # Create realistic test data (simulating robot point cloud)
    batch_size = 2
    num_points = 8000
    
    # Create test data with different scales (like real scenarios)
    test_data = torch.zeros(batch_size, num_points, 6, device=device)
    
    # Batch 0: centered robot-like object
    test_data[0, :, :3] = torch.randn(num_points, 3, device=device) * 0.8
    test_data[0, :, 3:] = torch.randn(num_points, 3, device=device) * 0.1
    
    # Batch 1: offset robot with different scale  
    test_data[1, :, :3] = torch.randn(num_points, 3, device=device) * 1.5 + torch.tensor([2, 1, 0], device=device)
    test_data[1, :, 3:] = torch.randn(num_points, 3, device=device) * 0.2
    
    print(f"📊 Test data shape: {test_data.shape}")
    print(f"📊 Coordinate ranges:")
    for b in range(batch_size):
        xyz = test_data[b, :, :3]
        print(f"   Batch {b}: [{xyz.min():.2f}, {xyz.max():.2f}], std={xyz.std():.3f}")
    
    # Compare both versions
    sampled_v1, sampled_v2 = model.compare_sampling_versions(test_data, num_tokens=2048, seed=42)
    
    print(f"\n🎯 FINAL RECOMMENDATION:")
    v1_zeros = sum([(sampled_v1[b].abs().sum(-1) < 1e-6).sum().item() for b in range(batch_size)])
    v2_zeros = sum([(sampled_v2[b].abs().sum(-1) < 1e-6).sum().item() for b in range(batch_size)])
    
    if v1_zeros > v2_zeros:
        print(f"✅ Version 2 is BETTER (fewer zero artifacts: {v2_zeros} vs {v1_zeros})")
        recommended_version = 2
    elif v2_zeros > v1_zeros:
        print(f"❌ Version 1 is better (fewer zero artifacts: {v1_zeros} vs {v2_zeros})")
        recommended_version = 1  
    else:
        print(f"⚖️  Both versions have similar zero padding: {v1_zeros}")
        recommended_version = 2  # Default to fixed version
    
    return recommended_version

if __name__ == "__main__":
    recommended = test_both_sampling_versions()
    print(f"\n🎯 Use version {recommended} for your robot inference!") 