from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm, LayerNorm
from diffusers.utils import logging
from diffusers.utils.accelerate_utils import apply_forward_hook
from einops import repeat
# from torch_cluster import fps
from tqdm import tqdm

from ..attention_processor import FusedTripoSGAttnProcessor2_0, TripoSGAttnProcessor2_0, FlashTripoSGAttnProcessor2_0
from ..embeddings import FrequencyPositionalEmbedding
from ..transformers.triposg_transformer import DiTBlock
from .vae import DiagonalGaussianDistribution

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class TripoSGEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 512,
        num_attention_heads: int = 8,
        num_layers: int = 8,
    ):
        super().__init__()

        self.proj_in = nn.Linear(in_channels, dim, bias=True)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    num_attention_heads=num_attention_heads,
                    use_self_attention=False,
                    use_cross_attention=True,
                    cross_attention_dim=dim,
                    cross_attention_norm_type="layer_norm",
                    activation_fn="gelu",
                    norm_type="fp32_layer_norm",
                    norm_eps=1e-5,
                    qk_norm=False,
                    qkv_bias=False,
                )  # cross attention
            ]
            + [
                DiTBlock(
                    dim=dim,
                    num_attention_heads=num_attention_heads,
                    use_self_attention=True,
                    self_attention_norm_type="fp32_layer_norm",
                    use_cross_attention=False,
                    activation_fn="gelu",
                    norm_type="fp32_layer_norm",
                    norm_eps=1e-5,
                    qk_norm=False,
                    qkv_bias=False,
                )
                for _ in range(num_layers)  # self attention
            ]
        )

        self.norm_out = LayerNorm(dim)

    def forward(self, sample_1: torch.Tensor, sample_2: torch.Tensor):
        hidden_states = self.proj_in(sample_1)
        encoder_hidden_states = self.proj_in(sample_2)

        for layer, block in enumerate(self.blocks):
            if layer == 0:
                hidden_states = block(
                    hidden_states, encoder_hidden_states=encoder_hidden_states
                )
            else:
                hidden_states = block(hidden_states)

        hidden_states = self.norm_out(hidden_states)

        return hidden_states


class TripoSGDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        dim: int = 512,
        num_attention_heads: int = 8,
        num_layers: int = 16,
        grad_type: str = "analytical",
        grad_interval: float = 0.001,
    ):
        super().__init__()

        if grad_type not in ["numerical", "analytical"]:
            raise ValueError(f"grad_type must be one of ['numerical', 'analytical']")
        self.grad_type = grad_type
        self.grad_interval = grad_interval

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    num_attention_heads=num_attention_heads,
                    use_self_attention=True,
                    self_attention_norm_type="fp32_layer_norm",
                    use_cross_attention=False,
                    activation_fn="gelu",
                    norm_type="fp32_layer_norm",
                    norm_eps=1e-5,
                    qk_norm=False,
                    qkv_bias=False,
                )
                for _ in range(num_layers)  # self attention
            ]
            + [
                DiTBlock(
                    dim=dim,
                    num_attention_heads=num_attention_heads,
                    use_self_attention=False,
                    use_cross_attention=True,
                    cross_attention_dim=dim,
                    cross_attention_norm_type="layer_norm",
                    activation_fn="gelu",
                    norm_type="fp32_layer_norm",
                    norm_eps=1e-5,
                    qk_norm=False,
                    qkv_bias=False,
                )  # cross attention
            ]
        )

        self.proj_query = nn.Linear(in_channels, dim, bias=True)

        self.norm_out = LayerNorm(dim)
        self.proj_out = nn.Linear(dim, out_channels, bias=True)

    def set_topk(self, topk):
        self.blocks[-1].set_topk(topk)

    def set_flash_processor(self, processor):
        self.blocks[-1].set_flash_processor(processor)

    def query_geometry(
        self,
        model_fn: callable,
        queries: torch.Tensor,
        sample: torch.Tensor,
        grad: bool = False,
    ):
        logits = model_fn(queries, sample)
        if grad:
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                if self.grad_type == "numerical":
                    interval = self.grad_interval
                    grad_value = []
                    for offset in [
                        (interval, 0, 0),
                        (0, interval, 0),
                        (0, 0, interval),
                    ]:
                        offset_tensor = torch.tensor(offset, device=queries.device)[
                            None, :
                        ]
                        res_p = model_fn(queries + offset_tensor, sample)[..., 0]
                        res_n = model_fn(queries - offset_tensor, sample)[..., 0]
                        grad_value.append((res_p - res_n) / (2 * interval))
                    grad_value = torch.stack(grad_value, dim=-1)
                else:
                    queries_d = torch.clone(queries)
                    queries_d.requires_grad = True
                    with torch.enable_grad():
                        res_d = model_fn(queries_d, sample)
                        grad_value = torch.autograd.grad(
                            res_d,
                            [queries_d],
                            grad_outputs=torch.ones_like(res_d),
                            create_graph=self.training,
                        )[0]
        else:
            grad_value = None

        return logits, grad_value

    def forward(
        self,
        sample: torch.Tensor,
        queries: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
    ):
        if kv_cache is None:
            hidden_states = sample
            for _, block in enumerate(self.blocks[:-1]):
                hidden_states = block(hidden_states)
            kv_cache = hidden_states

        # query grid logits by cross attention
        def query_fn(q, kv):
            q = self.proj_query(q)
            l = self.blocks[-1](q, encoder_hidden_states=kv)
            return self.proj_out(self.norm_out(l))

        logits, grad = self.query_geometry(
            query_fn, queries, kv_cache, grad=self.training
        )
        logits = logits * -1 if not isinstance(logits, Tuple) else logits[0] * -1

        return logits, kv_cache


class TripoSGVAEModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,  # NOTE xyz instead of feature dim
        latent_channels: int = 64,
        num_attention_heads: int = 8,
        width_encoder: int = 512,
        width_decoder: int = 1024,
        num_layers_encoder: int = 8,
        num_layers_decoder: int = 16,
        embedding_type: str = "frequency",
        embed_frequency: int = 8,
        embed_include_pi: bool = False,
    ):
        super().__init__()

        self.out_channels = 1

        if embedding_type == "frequency":
            self.embedder = FrequencyPositionalEmbedding(
                num_freqs=embed_frequency,
                logspace=True,
                input_dim=in_channels,
                include_pi=embed_include_pi,
            )
        else:
            raise NotImplementedError(
                f"Embedding type {embedding_type} is not supported."
            )

        self.encoder = TripoSGEncoder(
            in_channels=in_channels + self.embedder.out_dim,
            dim=width_encoder,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers_encoder,
        )
        self.decoder = TripoSGDecoder(
            in_channels=self.embedder.out_dim,
            out_channels=self.out_channels,
            dim=width_decoder,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers_decoder,
        )

        self.quant = nn.Linear(width_encoder, latent_channels * 2, bias=True)
        self.post_quant = nn.Linear(latent_channels, width_decoder, bias=True)

        self.use_slicing = False
        self.slicing_length = 1

    def set_flash_decoder(self):
        self.decoder.set_flash_processor(FlashTripoSGAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedTripoSGAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError(
                    "`fuse_qkv_projections()` is not supported for models having added KV projections."
                )

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedTripoSGAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(TripoSGAttnProcessor2_0())

    def enable_slicing(self, slicing_length: int = 1) -> None:
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True
        self.slicing_length = slicing_length

    def disable_slicing(self) -> None:
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def _sample_features(
        self, x: torch.Tensor, num_tokens: int = 2048, seed: Optional[int] = None
    ):
        """
        Sample points from features of the input point cloud.

        Args:
            x (torch.Tensor): The input point cloud. shape: (B, N, C)
            num_tokens (int, optional): The number of points to sample. Defaults to 2048.
            seed (Optional[int], optional): The random seed. Defaults to None.
        """
        rng = np.random.default_rng(seed)
        indices = rng.choice(
            x.shape[1], num_tokens * 4, replace=num_tokens * 4 > x.shape[1]
        )
        selected_points = x[:, indices]

        batch_size, num_points, num_channels = selected_points.shape
        flattened_points = selected_points.view(batch_size * num_points, num_channels)
        batch_indices = (
            torch.arange(batch_size).to(x.device).repeat_interleave(num_points)
        )

        # fps sampling
        sampling_ratio = 1.0 / 4
        sampled_indices = fps(
            flattened_points[:, :3],
            batch_indices,
            ratio=sampling_ratio,
            random_start=self.training,
        )
        sampled_points = flattened_points[sampled_indices].view(
            batch_size, -1, num_channels
        )

        return sampled_points

    def _encode(
        self, x: torch.Tensor, num_tokens: int = 2048, seed: Optional[int] = None
    ):
        position_channels = self.config.in_channels
        positions, features = x[..., :position_channels], x[..., position_channels:]
        x_kv = torch.cat([self.embedder(positions), features], dim=-1)

        sampled_x = self._sample_features(x, num_tokens, seed)
        positions, features = (
            sampled_x[..., :position_channels],
            sampled_x[..., position_channels:],
        )
        x_q = torch.cat([self.embedder(positions), features], dim=-1)

        x = self.encoder(x_q, x_kv)

        x = self.quant(x)

        return x

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True, **kwargs
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of point features into latents.
        """
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [
                self._encode(x_slice, **kwargs)
                for x_slice in x.split(self.slicing_length)
            ]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x, **kwargs)

        posterior = DiagonalGaussianDistribution(h, feature_dim=-1)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(
        self,
        z: torch.Tensor,
        sampled_points: torch.Tensor,
        num_chunks: int = 50000,
        to_cpu: bool = False,
        return_dict: bool = True,
    ) -> Union[DecoderOutput, torch.Tensor]:
        xyz_samples = sampled_points

        z = self.post_quant(z)

        num_points = xyz_samples.shape[1]
        kv_cache = None
        dec = []

        for i in range(0, num_points, num_chunks):
            queries = xyz_samples[:, i : i + num_chunks, :].to(z.device, dtype=z.dtype)
            queries = self.embedder(queries)

            z_, kv_cache = self.decoder(z, queries, kv_cache)
            dec.append(z_ if not to_cpu else z_.cpu())

        z = torch.cat(dec, dim=1)

        if not return_dict:
            return (z,)

        return DecoderOutput(sample=z)

    @apply_forward_hook
    def decode(
        self,
        z: torch.Tensor,
        sampled_points: torch.Tensor,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[DecoderOutput, torch.Tensor]:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [
                self._decode(z_slice, p_slice, **kwargs).sample
                for z_slice, p_slice in zip(
                    z.split(self.slicing_length),
                    sampled_points.split(self.slicing_length),
                )
            ]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z, sampled_points, **kwargs).sample

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def forward(self, x: torch.Tensor):
        pass
