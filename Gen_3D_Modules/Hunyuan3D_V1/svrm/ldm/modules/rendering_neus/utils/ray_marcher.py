# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Modified by Zexin He
# The modifications are subject to the same license as the original.


"""
The ray marcher takes the raw output of the implicit representation and uses the volume rendering equation to produce composited colors and depths.
Based off of the implementation in MipNeRF (this one doesn't do any cone tracing though!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnedVariance(nn.Module):
    def __init__(self, init_val):
        super(LearnedVariance, self).__init__()
        self.register_parameter("_inv_std", nn.Parameter(torch.tensor(init_val)))

    @property
    def inv_std(self):
        val = torch.exp(self._inv_std * 10.0)
        return val

    def forward(self, x):
        return torch.ones_like(x) * self.inv_std.clamp(1.0e-6, 1.0e6)


class MipRayMarcher2(nn.Module):
    def __init__(self, activation_factory):
        super().__init__()
        self.activation_factory = activation_factory
        self.variance = LearnedVariance(0.3)
        self.cos_anneal_ratio = 1.0
    def get_alpha(self, sdf, normal, dirs, dists):
        # sdf: [N 1]  normal: [N 3]   dirs: [N 3] dists: [N 1]
        # import ipdb; ipdb.set_trace()

        inv_std = self.variance(sdf)

        true_cos = (dirs * normal).sum(-1, keepdim=True)
        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio)
            + F.relu(-true_cos) * self.cos_anneal_ratio
        )  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_std)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_std)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
        return alpha
    
    def run_forward(self, colors, sdfs, depths, normals, ray_directions, rendering_options, bgcolor=None, real_normals=None):
        # depths: [B N_ray*N_sample 1] 
        # sdfs: [B, N_ray, N_sample 1]
        # import ipdb; ipdb.set_trace() 

        deltas = depths[:, :, 1:] - depths[:, :, :-1]
        colors_mid = (colors[:, :, :-1] + colors[:, :, 1:]) / 2
        sdfs_mid = (sdfs[:, :, :-1] + sdfs[:, :, 1:]) / 2
        depths_mid = (depths[:, :, :-1] + depths[:, :, 1:]) / 2
        normals_mid = (normals[:, :, :-1] + normals[:, :, 1:]) / 2

        # zhaohx add for normal :
        real_normals_mid = (real_normals[:, :, :-1] + real_normals[:, :, 1:]) / 2

        # # using factory mode for better usability
        # densities_mid = self.activation_factory(rendering_options)(densities_mid)

        # density_delta = densities_mid * deltas

        # alpha = 1 - torch.exp(-density_delta)

        # import ipdb; ipdb.set_trace()
        dirs = ray_directions.unsqueeze(2).expand(-1, -1, sdfs_mid.shape[-2], -1)
        B, N_ray, N_sample, _ = sdfs_mid.shape
        alpha = self.get_alpha(sdfs_mid.reshape(-1, 1), normals_mid.reshape(-1, 3), dirs.reshape(-1, 3), deltas.reshape(-1, 1))
        alpha = alpha.reshape(B, N_ray, N_sample, -1)
        
        alpha_shifted = torch.cat([torch.ones_like(alpha[:, :, :1]), 1-alpha + 1e-10], -2)
        weights = alpha * torch.cumprod(alpha_shifted, -2)[:, :, :-1]

        composite_rgb = torch.sum(weights * colors_mid, -2)
        weight_total = weights.sum(2)
        composite_depth = torch.sum(weights * depths_mid, -2) / weight_total

        # clip the composite to min/max range of depths
        composite_depth = torch.nan_to_num(composite_depth, float('inf'))
        composite_depth = torch.clamp(composite_depth, torch.min(depths), torch.max(depths))
        # import pdb; pdb.set_trace()

        # zhaohx add for normal :
        composite_normal = torch.sum(weights * real_normals_mid, -2) / weight_total
        composite_normal = torch.nan_to_num(composite_normal, float('inf'))
        composite_normal = torch.clamp(composite_normal, torch.min(real_normals), torch.max(real_normals))

        if rendering_options.get('white_back', False):
            # composite_rgb = composite_rgb + 1 - weight_total
            # weight_total[weight_total < 0.5] = 0
            # composite_rgb = composite_rgb * weight_total + 1 - weight_total
            # now is this
            if bgcolor is None:
                composite_rgb = composite_rgb + 1 - weight_total
                # composite_rgb = composite_rgb * weight_total + 1 - weight_total
            else:
                # import pdb; pdb.set_trace()
                bgcolor = bgcolor.permute(0, 2, 3, 1).contiguous().view(composite_rgb.shape[0], -1, composite_rgb.shape[-1])
                composite_rgb = composite_rgb + (1 - weight_total) * bgcolor
                # composite_rgb = composite_rgb * weight_total + (1 - weight_total) * bgcolor
            # composite_rgb = composite_rgb
            # print('new white_back')

        # rendered value scale is 0-1, comment out original mipnerf scaling
        # composite_rgb = composite_rgb * 2 - 1 # Scale to (-1, 1)

        return composite_rgb, composite_depth, weights, composite_normal


    def forward(self,  colors, sdfs, depths, normals, ray_directions, rendering_options, bgcolor=None, real_normals=None):
        composite_rgb, composite_depth, weights, composite_normal = self.run_forward(colors, sdfs, depths, normals, ray_directions, rendering_options, bgcolor, real_normals)

        return composite_rgb, composite_depth, weights, composite_normal
