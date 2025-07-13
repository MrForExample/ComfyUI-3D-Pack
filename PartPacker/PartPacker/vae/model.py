"""
-----------------------------------------------------------------------------
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
"""

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs.schema import ModelConfig
from .modules.transformer import AttentionBlock, FlashQueryLayer
from .utils import (
    DiagonalGaussianDistribution,
    DummyLatent,
    calculate_iou,
    calculate_metrics,
    construct_grid_points,
    extract_mesh,
    sync_timer,
)

import sys
import os


class Model(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.precision = torch.bfloat16  # manually handle low-precision training, always use bf16

        # point encoder
        self.proj_input = nn.Linear(3 + config.point_fourier_dim, config.hidden_dim)

        self.perceiver = AttentionBlock(
            config.hidden_dim,
            num_heads=config.num_heads,
            dim_context=config.hidden_dim,
            qknorm=config.qknorm,
            qknorm_type=config.qknorm_type,
        )

        if self.config.salient_attn_mode == "dual":
            self.perceiver_dorases = AttentionBlock(
                config.hidden_dim,
                num_heads=config.num_heads,
                dim_context=config.hidden_dim,
                qknorm=config.qknorm,
                qknorm_type=config.qknorm_type,
            )

        # self-attention encoder
        self.encoder = nn.ModuleList(
            [
                AttentionBlock(
                    config.hidden_dim, config.num_heads, qknorm=config.qknorm, qknorm_type=config.qknorm_type
                )
                for _ in range(config.num_enc_layers)
            ]
        )

        # vae bottleneck
        self.norm_down = nn.LayerNorm(config.hidden_dim)
        self.proj_down_mean = nn.Linear(config.hidden_dim, config.latent_dim)
        if not self.config.use_ae:
            self.proj_down_std = nn.Linear(config.hidden_dim, config.latent_dim)
        self.proj_up = nn.Linear(config.latent_dim, config.dec_hidden_dim)

        # self-attention decoder
        self.decoder = nn.ModuleList(
            [
                AttentionBlock(
                    config.dec_hidden_dim, config.dec_num_heads, qknorm=config.qknorm, qknorm_type=config.qknorm_type
                )
                for _ in range(config.num_dec_layers)
            ]
        )

        # cross-attention query
        self.proj_query = nn.Linear(3 + config.point_fourier_dim, config.query_hidden_dim)
        if self.config.use_flash_query:
            self.norm_query_context = nn.LayerNorm(config.hidden_dim, eps=1e-6, elementwise_affine=False)
            self.attn_query = FlashQueryLayer(
                config.query_hidden_dim,
                num_heads=config.query_num_heads,
                dim_context=config.hidden_dim,
                qknorm=config.qknorm,
                qknorm_type=config.qknorm_type,
            )
        else:
            self.attn_query = AttentionBlock(
                config.query_hidden_dim,
                num_heads=config.query_num_heads,
                dim_context=config.hidden_dim,
                qknorm=config.qknorm,
                qknorm_type=config.qknorm_type,
            )
        self.norm_out = nn.LayerNorm(config.query_hidden_dim)
        self.proj_out = nn.Linear(config.query_hidden_dim, 1)

        # preload from a checkpoint (NOTE: this happens BEFORE checkpointer loading latest checkpoint!)
        if self.config.pretrain_path is not None:
            try:
                ckpt = torch.load(self.config.pretrain_path)  # local path
                self.load_state_dict(ckpt["model"], strict=True)
                del ckpt
                print(f"Loaded VAE from {self.config.pretrain_path}")
            except Exception as e:
                print(
                    f"Failed to load VAE from {self.config.pretrain_path}: {e}, make sure you resumed from a valid checkpoint!"
                )

        # log
        n_params = 0
        for p in self.parameters():
            n_params += p.numel()
        print(f"Number of parameters in VAE: {n_params / 1e6:.2f}M")

    # override to support tolerant loading (only load matched shape)
    def load_state_dict(self, state_dict, strict=True, assign=False):
        local_state_dict = self.state_dict()
        seen_keys = {k: False for k in local_state_dict.keys()}
        for k, v in state_dict.items():
            if k in local_state_dict:
                seen_keys[k] = True
                if local_state_dict[k].shape == v.shape:
                    local_state_dict[k].copy_(v)
                else:
                    print(f"mismatching shape for key {k}: loaded {local_state_dict[k].shape} but model has {v.shape}")
            else:
                print(f"unexpected key {k} in loaded state dict")
        for k in seen_keys:
            if not seen_keys[k]:
                print(f"missing key {k} in loaded state dict")

    def fourier_encoding(self, points: torch.Tensor):
        # points: [B, N, 3], float32 for precision
        # assert points.dtype == torch.float32, "Query points must be float32"

        F = self.config.point_fourier_dim // (2 * points.shape[-1])

        if self.config.fourier_version == "v1":  # default
            exponent = torch.arange(1, F + 1, device=points.device, dtype=torch.float32) / F  # [F], range from 0 to 1
            freq_band = 512**exponent  # [F], min frequency is 1, max frequency is 1/freq
            freq_band *= torch.pi
        elif self.config.fourier_version == "v2":
            exponent = torch.arange(F, device=points.device, dtype=torch.float32) / (F - 1)  # [F], range from 0 to 1
            freq_band = 1024**exponent  # [F]
            freq_band *= torch.pi
        elif self.config.fourier_version == "v3":  # hunyuan3d-2
            freq_band = 2 ** torch.arange(F, device=points.device, dtype=torch.float32)  # [F]

        spectrum = points.unsqueeze(-1) * freq_band  # [B,...,3,F]
        sin, cos = spectrum.sin(), spectrum.cos()  # [B,...,3,F]
        input_enc = torch.stack([sin, cos], dim=-2)  # [B,...,3,2,F]
        input_enc = input_enc.view(*points.shape[:-1], -1)  # [B,...,6F] = [B,...,dim]
        return torch.cat([input_enc, points], dim=-1).to(dtype=self.precision)  # [B,...,dim+input_dim]

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        super().on_train_start(memory_format=memory_format)
        self.to(dtype=self.precision, memory_format=memory_format)  # use bfloat16 for training

    def encode(self, data: dict[str, torch.Tensor]):
        # uniform points
        pointcloud = data["pointcloud"]  # [B, N, 3]

        # fourier embed and project
        pointcloud = self.fourier_encoding(pointcloud)  # [B, N, 3+C]
        pointcloud = self.proj_input(pointcloud)  # [B, N, hidden_dim]

        # salient points
        if self.config.use_salient_point:
            pointcloud_dorases = data["pointcloud_dorases"]  # [B, M, 3]

            # fourier embed and project (shared weights)
            pointcloud_dorases = self.fourier_encoding(pointcloud_dorases)  # [B, M, 3+C]
            pointcloud_dorases = self.proj_input(pointcloud_dorases)  # [B, M, hidden_dim]

        # gather fps point
        fps_indices = data["fps_indices"]  # [B, N']
        pointcloud_query = torch.gather(pointcloud, 1, fps_indices.unsqueeze(-1).expand(-1, -1, pointcloud.shape[-1]))

        if self.config.use_salient_point:
            fps_indices_dorases = data["fps_indices_dorases"]  # [B, M']

            if fps_indices_dorases.shape[1] > 0:
                pointcloud_query_dorases = torch.gather(
                    pointcloud_dorases,
                    1,
                    fps_indices_dorases.unsqueeze(-1).expand(-1, -1, pointcloud_dorases.shape[-1]),
                )

                # combine both fps points as the query
                pointcloud_query = torch.cat(
                    [pointcloud_query, pointcloud_query_dorases], dim=1
                )  # [B, N'+M', hidden_dim]

            # dual cross-attention
            if self.config.salient_attn_mode == "dual_shared":
                hidden_states = self.perceiver(pointcloud_query, pointcloud) + self.perceiver(
                    pointcloud_query, pointcloud_dorases
                )  # [B, N'+M', hidden_dim]
            elif self.config.salient_attn_mode == "dual":
                hidden_states = self.perceiver(pointcloud_query, pointcloud) + self.perceiver_dorases(
                    pointcloud_query, pointcloud_dorases
                )
            else:  # single, hunyuan3d-2 style
                hidden_states = self.perceiver(pointcloud_query, torch.cat([pointcloud, pointcloud_dorases], dim=1))
        else:
            hidden_states = self.perceiver(pointcloud_query, pointcloud)  # [B, N', hidden_dim]

        # encoder
        for block in self.encoder:
            hidden_states = block(hidden_states)

        # bottleneck
        hidden_states = self.norm_down(hidden_states)
        latent_mean = self.proj_down_mean(hidden_states).float()
        if not self.config.use_ae:
            latent_std = self.proj_down_std(hidden_states).float()
            posterior = DiagonalGaussianDistribution(latent_mean, latent_std)
        else:
            posterior = DummyLatent(latent_mean)

        return posterior

    def decode(self, latent: torch.Tensor):
        latent = latent.to(dtype=self.precision)
        hidden_states = self.proj_up(latent)

        for block in self.decoder:
            hidden_states = block(hidden_states)

        return hidden_states

    def query(self, query_points: torch.Tensor, hidden_states: torch.Tensor):
        # query_points: [B, N, 3], float32 to keep the precision

        query_points = self.fourier_encoding(query_points)  # [B, N, 3+C]
        query_points = self.proj_query(query_points)  # [B, N, hidden_dim]

        # cross attention
        query_output = self.attn_query(query_points, hidden_states)  # [B, N, hidden_dim]

        # output linear
        query_output = self.norm_out(query_output)
        pred = self.proj_out(query_output)  # [B, N, 1]

        return pred

    def training_step(
        self,
        data: dict[str, torch.Tensor],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        output = {}

        # cut off fps point during training for progressive flow
        if self.training:
            # randomly choose from a set of cutoff candidates
            cutoff_index = np.random.choice(len(self.config.cutoff_fps_prob), p=self.config.cutoff_fps_prob)
            cutoff_fps_point = self.config.cutoff_fps_point[cutoff_index]
            cutoff_fps_salient_point = self.config.cutoff_fps_salient_point[cutoff_index]
            # prefix of FPS points are still FPS points
            data["fps_indices"] = data["fps_indices"][:, :cutoff_fps_point]
            if self.config.use_salient_point:
                data["fps_indices_dorases"] = data["fps_indices_dorases"][:, :cutoff_fps_salient_point]

        loss = 0

        # encode
        posterior = self.encode(data)
        latent_geom = posterior.sample() if self.training else posterior.mode()

        # decode
        hidden_states = self.decode(latent_geom)

        # cross-attention query
        query_points = data["query_points"]  # [B, N, 3], float32

        # the context norm can be moved out to avoid repeated computation
        if self.config.use_flash_query:
            hidden_states = self.norm_query_context(hidden_states)

        pred = self.query(query_points, hidden_states).squeeze(-1).float()  # [B, N]
        gt = data["query_gt"].float()  # [B, N], in [-1, 1]

        # main loss
        loss_mse = F.mse_loss(pred, gt, reduction="mean")
        loss += loss_mse

        loss_l1 = F.l1_loss(pred, gt, reduction="mean")
        loss += loss_l1

        # kl loss
        loss_kl = posterior.kl().mean()
        loss += self.config.kl_weight * loss_kl

        # metrics
        with torch.no_grad():
            output["scalar"] = {}  # for wandb logging
            output["scalar"]["loss_mse"] = loss_mse.detach()
            output["scalar"]["loss_l1"] = loss_l1.detach()
            output["scalar"]["loss_kl"] = loss_kl.detach()
            output["scalar"]["iou_fg"] = calculate_iou(pred, gt, target_value=1)
            output["scalar"]["iou_bg"] = calculate_iou(pred, gt, target_value=0)
            output["scalar"]["precision"], output["scalar"]["recall"], output["scalar"]["f1"] = calculate_metrics(
                pred, gt, target_value=1
            )

        return output, loss

    @torch.no_grad()
    def validation_step(
        self,
        data: dict[str, torch.Tensor],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        return self.training_step(data, iteration)

    @torch.inference_mode()
    @sync_timer("vae forward")
    def forward(
        self,
        data: dict[str, torch.Tensor],
        mode: Literal["dense", "hierarchical"] = "hierarchical",
        max_samples_per_iter: int =8192,# 512**2 #for low ram <10G
        resolution: int = 512,
        min_resolution: int = 64,  # for hierarchical
    ) -> dict[str, torch.Tensor]:
        output = {}

        # encode
        if "latent" in data:
            latent = data["latent"]
        else:
            posterior = self.encode(data)
            output["posterior"] = posterior
            latent = posterior.mode()

        output["latent"] = latent
        B = latent.shape[0]

        # decode
        hidden_states = self.decode(latent)
        output["hidden_states"] = hidden_states  # [B, N, hidden_dim] for the last cross-attention decoder

        # the context norm can be moved out to avoid repeated computation
        if self.config.use_flash_query:
            hidden_states = self.norm_query_context(hidden_states)

        # query
        def chunked_query(grid_points):
            if grid_points.shape[0] <= max_samples_per_iter:
                return self.query(grid_points.unsqueeze(0), hidden_states).squeeze(-1)  # [B, N]
            all_pred = []
            for i in range(0, grid_points.shape[0], max_samples_per_iter):
                grid_chunk = grid_points[i : i + max_samples_per_iter]
                pred_chunk = self.query(grid_chunk.unsqueeze(0), hidden_states)
                all_pred.append(pred_chunk)
            return torch.cat(all_pred, dim=1).squeeze(-1)  # [B, N]

        if mode == "dense":
            grid_points = construct_grid_points(resolution).to(latent.device)
            grid_points = grid_points.contiguous().view(-1, 3)
            grid_vals = chunked_query(grid_points).float().view(B, resolution + 1, resolution + 1, resolution + 1)

        elif mode == "hierarchical":
            assert resolution >= min_resolution, "Resolution must be greater than or equal to min_resolution"
            assert B == 1, "Only one batch is supported for hierarchical mode"

            resolutions = []
            res = resolution
            while res >= min_resolution:
                resolutions.append(res)
                res = res // 2
            resolutions.reverse()  # e.g., [64, 128, 256, 512]

            # dense-query the coarsest resolution
            res = resolutions[0]
            grid_points = construct_grid_points(res).to(latent.device)
            grid_points = grid_points.contiguous().view(-1, 3)
            grid_vals = chunked_query(grid_points).float().view(res + 1, res + 1, res + 1)

            # sparse-query finer resolutions
            dilate_kernel_3 = torch.ones(1, 1, 3, 3, 3, dtype=torch.float32, device=latent.device)
            dilate_kernel_5 = torch.ones(1, 1, 5, 5, 5, dtype=torch.float32, device=latent.device)
            for i in range(1, len(resolutions)):
                res = resolutions[i]
                # get the boundary grid mask in the coarser grid (where the grid_vals have different signs with at least one of its neighbors)
                grid_signs = grid_vals >= 0
                mask = torch.zeros_like(grid_signs)
                mask[1:, :, :] += grid_signs[1:, :, :] != grid_signs[:-1, :, :]
                mask[:-1, :, :] += grid_signs[:-1, :, :] != grid_signs[1:, :, :]
                mask[:, 1:, :] += grid_signs[:, 1:, :] != grid_signs[:, :-1, :]
                mask[:, :-1, :] += grid_signs[:, :-1, :] != grid_signs[:, 1:, :]
                mask[:, :, 1:] += grid_signs[:, :, 1:] != grid_signs[:, :, :-1]
                mask[:, :, :-1] += grid_signs[:, :, :-1] != grid_signs[:, :, 1:]
                # empirical: also add those with abs(grid_vals) < 0.95
                mask += grid_vals.abs() < 0.95
                mask = (mask > 0).float()
                # empirical: dilate the coarse mask
                if res < 512:
                    mask = mask.unsqueeze(0).unsqueeze(0)
                    mask = F.conv3d(mask, weight=dilate_kernel_3, padding=1)
                    mask = mask.squeeze(0).squeeze(0)
                # get the coarse coordinates
                cidx_x, cidx_y, cidx_z = torch.nonzero(mask, as_tuple=True)
                # fill to the fine indices
                mask_fine = torch.zeros(res + 1, res + 1, res + 1, dtype=torch.float32, device=latent.device)
                mask_fine[cidx_x * 2, cidx_y * 2, cidx_z * 2] = 1
                # empirical: dilate the fine mask
                if res < 512:
                    mask_fine = mask_fine.unsqueeze(0).unsqueeze(0)
                    mask_fine = F.conv3d(mask_fine, weight=dilate_kernel_3, padding=1)
                    mask_fine = mask_fine.squeeze(0).squeeze(0)
                else:
                    mask_fine = mask_fine.unsqueeze(0).unsqueeze(0)
                    mask_fine = F.conv3d(mask_fine, weight=dilate_kernel_5, padding=2)
                    mask_fine = mask_fine.squeeze(0).squeeze(0)
                # get the fine coordinates
                fidx_x, fidx_y, fidx_z = torch.nonzero(mask_fine, as_tuple=True)
                # convert to float query points
                query_points = torch.stack([fidx_x, fidx_y, fidx_z], dim=-1)  # [N, 3]
                query_points = query_points * 2 / res - 1  # [N, 3], in [-1, 1]
                # query
                pred = chunked_query(query_points).float()
                # fill to the fine indices
                grid_vals = torch.full((res + 1, res + 1, res + 1), -100.0, dtype=torch.float32, device=latent.device)
                grid_vals[fidx_x, fidx_y, fidx_z] = pred
                # print(f"[INFO] hierarchical: resolution: {res}, valid coarse points: {len(cidx_x)}, valid fine points: {len(fidx_x)}")

            grid_vals = grid_vals.unsqueeze(0)  # [1, res+1, res+1, res+1]
            grid_vals[grid_vals <= -100.0] = float("nan")  # use nans to ignore invalid regions

        # extract mesh
        meshes = []
        for b in range(B):
            vertices, faces = extract_mesh(grid_vals[b], resolution)
            meshes.append((vertices, faces))
        output["meshes"] = meshes

        return output
