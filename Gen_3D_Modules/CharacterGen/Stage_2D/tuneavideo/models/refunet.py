import torch
from einops import rearrange
from typing import Any, Dict, Optional
from diffusers.utils.import_utils import is_xformers_available
from .transformer_mv2d import XFormersMVAttnProcessor, MVAttnProcessor
class ReferenceOnlyAttnProc(torch.nn.Module):
    def __init__(
        self,
        chained_proc,
        enabled=False,
        name=None
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.chained_proc = chained_proc
        self.name = name

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
        mode="w", ref_dict: dict = None, is_cfg_guidance = False,num_views=4,
            multiview_attention=True,
            cross_domain_attention=False,
    ) -> Any:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        # print(self.enabled)
        if self.enabled:
            if mode == 'w':
                ref_dict[self.name] = encoder_hidden_states
                res = self.chained_proc(attn, hidden_states, encoder_hidden_states, attention_mask, num_views=1,
                    multiview_attention=False,
                    cross_domain_attention=False,)
            elif mode == 'r':
                encoder_hidden_states = rearrange(encoder_hidden_states, '(b t) d c-> b (t d) c', t=num_views)
                if self.name in ref_dict:
                    encoder_hidden_states = torch.cat([encoder_hidden_states, ref_dict.pop(self.name)], dim=1).unsqueeze(1).repeat(1,num_views,1,1).flatten(0,1)
                res = self.chained_proc(attn, hidden_states, encoder_hidden_states, attention_mask, num_views=num_views,
                    multiview_attention=False,
                    cross_domain_attention=False,)
            elif mode == 'm':
                encoder_hidden_states = torch.cat([encoder_hidden_states, ref_dict[self.name]], dim=1)
            elif mode == 'n':
                encoder_hidden_states = rearrange(encoder_hidden_states, '(b t) d c-> b (t d) c', t=num_views)
                encoder_hidden_states = torch.cat([encoder_hidden_states], dim=1).unsqueeze(1).repeat(1,num_views,1,1).flatten(0,1)
                res = self.chained_proc(attn, hidden_states, encoder_hidden_states, attention_mask, num_views=num_views,
                    multiview_attention=False,
                    cross_domain_attention=False,)
            else:
                assert False, mode
        else:
            res = self.chained_proc(attn, hidden_states, encoder_hidden_states, attention_mask)
        return res
        
class RefOnlyNoisedUNet(torch.nn.Module):
    def __init__(self, unet, train_sched, val_sched) -> None:
        super().__init__()
        self.unet = unet
        self.train_sched = train_sched
        self.val_sched = val_sched

        unet_lora_attn_procs = dict()
        for name, _ in unet.attn_processors.items():
            if is_xformers_available():
                default_attn_proc = XFormersMVAttnProcessor()
            else:
                default_attn_proc = MVAttnProcessor()
            unet_lora_attn_procs[name] = ReferenceOnlyAttnProc(
                default_attn_proc, enabled=name.endswith("attn1.processor"), name=name)
            
        self.unet.set_attn_processor(unet_lora_attn_procs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)

    def forward_cond(self, noisy_cond_lat, timestep, encoder_hidden_states, class_labels, ref_dict, is_cfg_guidance, **kwargs):
        if is_cfg_guidance:
            encoder_hidden_states = encoder_hidden_states[1:]
            class_labels = class_labels[1:]
        self.unet(
            noisy_cond_lat, timestep,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            cross_attention_kwargs=dict(mode="w", ref_dict=ref_dict),
            **kwargs
        )

    def forward(
        self, sample, timestep, encoder_hidden_states, class_labels=None,
        *args, cross_attention_kwargs,
        down_block_res_samples=None, mid_block_res_sample=None,
        **kwargs
    ):
        cond_lat = cross_attention_kwargs['cond_lat']
        is_cfg_guidance = cross_attention_kwargs.get('is_cfg_guidance', False)
        noise = torch.randn_like(cond_lat)
        if self.training:
            noisy_cond_lat = self.train_sched.add_noise(cond_lat, noise, timestep)
            noisy_cond_lat = self.train_sched.scale_model_input(noisy_cond_lat, timestep)
        else:
            noisy_cond_lat = self.val_sched.add_noise(cond_lat, noise, timestep.reshape(-1))
            noisy_cond_lat = self.val_sched.scale_model_input(noisy_cond_lat, timestep.reshape(-1))
        ref_dict = {}
        self.forward_cond(
            noisy_cond_lat, timestep,
            encoder_hidden_states, class_labels,
            ref_dict, is_cfg_guidance, **kwargs
        )
        weight_dtype = self.unet.dtype
        return self.unet(
            sample, timestep,
            encoder_hidden_states, *args,
            class_labels=class_labels,
            cross_attention_kwargs=dict(mode="r", ref_dict=ref_dict, is_cfg_guidance=is_cfg_guidance),
            down_block_additional_residuals=[
                sample.to(dtype=weight_dtype) for sample in down_block_res_samples
            ] if down_block_res_samples is not None else None,
            mid_block_additional_residual=(
                mid_block_res_sample.to(dtype=weight_dtype)
                if mid_block_res_sample is not None else None
            ),
            **kwargs
        )