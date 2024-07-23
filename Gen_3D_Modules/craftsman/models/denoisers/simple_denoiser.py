from dataclasses import dataclass

import torch
import torch.nn as nn
from typing import Optional
from diffusers.models.embeddings import Timesteps
import math

import craftsman
from craftsman.models.transformers.attention import ResidualAttentionBlock
from craftsman.models.transformers.utils import init_linear, MLP
from craftsman.utils.base import BaseModule


class UNetDiffusionTransformer(nn.Module):
    def __init__(
            self,
            *,
            n_ctx: int,
            width: int,
            layers: int,
            heads: int,
            init_scale: float = 0.25,
            qkv_bias: bool = False,
            skip_ln: bool = False,
            use_checkpoint: bool = False
    ):
        super().__init__()

        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers

        self.encoder = nn.ModuleList()
        for _ in range(layers):
            resblock = ResidualAttentionBlock(
                n_ctx=n_ctx,
                width=width,
                heads=heads,
                init_scale=init_scale,
                qkv_bias=qkv_bias,
                use_checkpoint=use_checkpoint
            )
            self.encoder.append(resblock)

        self.middle_block = ResidualAttentionBlock(
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_checkpoint=use_checkpoint
        )

        self.decoder = nn.ModuleList()
        for _ in range(layers):
            resblock = ResidualAttentionBlock(
                n_ctx=n_ctx,
                width=width,
                heads=heads,
                init_scale=init_scale,
                qkv_bias=qkv_bias,
                use_checkpoint=use_checkpoint
            )
            linear = nn.Linear(width * 2, width)
            init_linear(linear, init_scale)

            layer_norm = nn.LayerNorm(width) if skip_ln else None

            self.decoder.append(nn.ModuleList([resblock, linear, layer_norm]))

    def forward(self, x: torch.Tensor):

        enc_outputs = []
        for block in self.encoder:
            x = block(x)
            enc_outputs.append(x)

        x = self.middle_block(x)

        for i, (resblock, linear, layer_norm) in enumerate(self.decoder):
            x = torch.cat([enc_outputs.pop(), x], dim=-1)
            x = linear(x)

            if layer_norm is not None:
                x = layer_norm(x)

            x = resblock(x)

        return x


@craftsman.register("simple-denoiser")
class SimpleDenoiser(BaseModule):

    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: Optional[str] = None
        input_channels: int = 32
        output_channels: int = 32
        n_ctx: int = 512
        width: int = 768
        layers: int = 6
        heads: int = 12
        context_dim: int = 1024
        context_ln: bool = True
        skip_ln: bool = False
        init_scale: float = 0.25
        flip_sin_to_cos: bool = False
        use_checkpoint: bool = False
    
    cfg: Config

    def configure(self) -> None:
        super().configure()
    
        init_scale = self.cfg.init_scale * math.sqrt(1.0 / self.cfg.width)

        self.backbone = UNetDiffusionTransformer(
            n_ctx=self.cfg.n_ctx,
            width=self.cfg.width,
            layers=self.cfg.layers,
            heads=self.cfg.heads,
            skip_ln=self.cfg.skip_ln,
            init_scale=init_scale,
            use_checkpoint=self.cfg.use_checkpoint
        )
        self.ln_post = nn.LayerNorm(self.cfg.width)
        self.input_proj = nn.Linear(self.cfg.input_channels, self.cfg.width)
        self.output_proj = nn.Linear(self.cfg.width, self.cfg.output_channels)

        # timestep embedding
        self.time_embed = Timesteps(self.cfg.width, flip_sin_to_cos=self.cfg.flip_sin_to_cos, downscale_freq_shift=0)
        self.time_proj = MLP(width=self.cfg.width, init_scale=init_scale)

        if self.cfg.context_ln:
            self.context_embed = nn.Sequential(
                nn.LayerNorm(self.cfg.context_dim),
                nn.Linear(self.cfg.context_dim, self.cfg.width),
            )
        else:
            self.context_embed = nn.Linear(self.cfg.context_dim, self.cfg.width)
        
        if self.cfg.pretrained_model_name_or_path:
            pretrained_ckpt = torch.load(self.cfg.pretrained_model_name_or_path, map_location="cpu")
            _pretrained_ckpt = {}
            for k, v in pretrained_ckpt.items():
                if k.startswith('denoiser_model.'):
                    _pretrained_ckpt[k.replace('denoiser_model.', '')] = v
            pretrained_ckpt = _pretrained_ckpt
            if 'state_dict' in pretrained_ckpt:
                _pretrained_ckpt = {}
                for k, v in pretrained_ckpt['state_dict'].items():
                    if k.startswith('denoiser_model.'):
                        _pretrained_ckpt[k.replace('denoiser_model.', '')] = v
                pretrained_ckpt = _pretrained_ckpt
            self.load_state_dict(pretrained_ckpt, strict=True)

    def forward(self,
                model_input: torch.FloatTensor,
                timestep: torch.LongTensor,
                context: torch.FloatTensor):

        r"""
        Args:
            model_input (torch.FloatTensor): [bs, n_data, c]
            timestep (torch.LongTensor): [bs,]
            context (torch.FloatTensor): [bs, context_tokens, c]

        Returns:
            sample (torch.FloatTensor): [bs, n_data, c]

        """

        _, n_data, _ = model_input.shape

        # 1. time
        t_emb = self.time_proj(self.time_embed(timestep)).unsqueeze(dim=1)

        # 2. conditions projector
        context = self.context_embed(context)

        # 3. denoiser
        x = self.input_proj(model_input)
        x = torch.cat([t_emb, context, x], dim=1)
        x = self.backbone(x)
        x = self.ln_post(x)
        x = x[:, -n_data:] # B, n_data, width
        sample = self.output_proj(x) # B, n_data, embed_dim

        return sample