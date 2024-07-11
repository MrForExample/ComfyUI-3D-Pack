from typing import Any, Dict, Optional
import torch
from diffusers.models.attention_processor import Attention

def construct_pix2pix_attention(hidden_states_dim, norm_type="none"):
    if norm_type == "layernorm":
        norm = torch.nn.LayerNorm(hidden_states_dim)
    else:
        norm = torch.nn.Identity()
    attention = Attention(
        query_dim=hidden_states_dim,
        heads=8,
        dim_head=hidden_states_dim // 8,
        bias=True,
    )
    # NOTE: xformers 0.22 does not support batchsize >= 4096
    attention.xformers_not_supported = True # hacky solution
    return norm, attention

class ExtraAttnProc(torch.nn.Module):
    def __init__(
        self,
        chained_proc,
        enabled=False,
        name=None,
        mode='extract',
        with_proj_in=False,
        proj_in_dim=768,
        target_dim=None,
        pixel_wise_crosspond=False,
        norm_type="none",   # none or layernorm
        crosspond_effect_on="all",  # all or first
        crosspond_chain_pos="parralle",     # before or parralle or after
        simple_3d=False,
        views=4,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.chained_proc = chained_proc
        self.name = name
        self.mode = mode
        self.with_proj_in=with_proj_in
        self.proj_in_dim = proj_in_dim
        self.target_dim = target_dim or proj_in_dim
        self.hidden_states_dim = self.target_dim
        self.pixel_wise_crosspond = pixel_wise_crosspond
        self.crosspond_effect_on = crosspond_effect_on
        self.crosspond_chain_pos = crosspond_chain_pos
        self.views = views
        self.simple_3d = simple_3d
        if self.with_proj_in and self.enabled:
            self.in_linear = torch.nn.Linear(self.proj_in_dim, self.target_dim, bias=False)
            if self.target_dim == self.proj_in_dim:
                self.in_linear.weight.data = torch.eye(proj_in_dim)
        else:
            self.in_linear = None
        if self.pixel_wise_crosspond and self.enabled:
            self.crosspond_norm, self.crosspond_attention = construct_pix2pix_attention(self.hidden_states_dim, norm_type=norm_type)
    
    def do_crosspond_attention(self, hidden_states: torch.FloatTensor, other_states: torch.FloatTensor):
        hidden_states = self.crosspond_norm(hidden_states)
        
        batch, L, D = hidden_states.shape
        assert hidden_states.shape == other_states.shape, f"got {hidden_states.shape} and {other_states.shape}"
        # to -> batch * L, 1, D
        hidden_states = hidden_states.reshape(batch * L, 1, D)
        other_states = other_states.reshape(batch * L, 1, D)
        hidden_states_catted = other_states
        hidden_states = self.crosspond_attention(
            hidden_states,
            encoder_hidden_states=hidden_states_catted,
        )
        return hidden_states.reshape(batch, L, D)
    
    def __call__(
        self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None,
        ref_dict: dict = None, mode=None, **kwargs
    ) -> Any:
        if not self.enabled:
            return self.chained_proc(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        assert ref_dict is not None
        if (mode or self.mode) == 'extract':
            ref_dict[self.name] = hidden_states
            hidden_states1 = self.chained_proc(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)
            if self.pixel_wise_crosspond and self.crosspond_chain_pos == "after":
                ref_dict[self.name] = hidden_states1
            return hidden_states1
        elif (mode or self.mode) == 'inject':
            ref_state = ref_dict.pop(self.name)
            if self.with_proj_in:
                ref_state = self.in_linear(ref_state)
            
            B, L, D = ref_state.shape
            if hidden_states.shape[0] == B:
                modalities = 1
                views = 1
            else:
                modalities = hidden_states.shape[0] // B // self.views
                views = self.views
            if self.pixel_wise_crosspond:
                if self.crosspond_effect_on == "all":
                    ref_state = ref_state[:, None].expand(-1, modalities * views, -1, -1).reshape(-1, *ref_state.shape[-2:])
                    
                    if self.crosspond_chain_pos == "before":
                        hidden_states = hidden_states + self.do_crosspond_attention(hidden_states, ref_state)
                        
                    hidden_states1 = self.chained_proc(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)
                    
                    if self.crosspond_chain_pos == "parralle":
                        hidden_states1 = hidden_states1 + self.do_crosspond_attention(hidden_states, ref_state)
                        
                    if self.crosspond_chain_pos == "after":
                        hidden_states1 = hidden_states1 + self.do_crosspond_attention(hidden_states1, ref_state)
                    return hidden_states1
                else:
                    assert self.crosspond_effect_on == "first"
                    # hidden_states [B * modalities * views, L, D]
                    # ref_state [B, L, D]
                    ref_state = ref_state[:, None].expand(-1, modalities, -1, -1).reshape(-1, ref_state.shape[-2], ref_state.shape[-1])  # [B * modalities, L, D]
                    
                    def do_paritial_crosspond(hidden_states, ref_state):
                        first_view_hidden_states = hidden_states.view(-1, views, hidden_states.shape[1], hidden_states.shape[2])[:, 0]  # [B * modalities, L, D]
                        hidden_states2 = self.do_crosspond_attention(first_view_hidden_states, ref_state) # [B * modalities, L, D]
                        hidden_states2_padded = torch.zeros_like(hidden_states).reshape(-1, views, hidden_states.shape[1], hidden_states.shape[2])
                        hidden_states2_padded[:, 0] = hidden_states2
                        hidden_states2_padded = hidden_states2_padded.reshape(-1, hidden_states.shape[1], hidden_states.shape[2])
                        return hidden_states2_padded
                    
                    if self.crosspond_chain_pos == "before":
                        hidden_states = hidden_states + do_paritial_crosspond(hidden_states, ref_state)
                    
                    hidden_states1 = self.chained_proc(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)    # [B * modalities * views, L, D]
                    if self.crosspond_chain_pos == "parralle":
                        hidden_states1 = hidden_states1 + do_paritial_crosspond(hidden_states, ref_state)
                    if self.crosspond_chain_pos == "after":
                        hidden_states1 = hidden_states1 + do_paritial_crosspond(hidden_states1, ref_state)
                    return hidden_states1
            elif self.simple_3d:
                B, L, C = encoder_hidden_states.shape
                mv = self.views
                encoder_hidden_states = encoder_hidden_states.reshape(B // mv, mv, L, C)
                ref_state = ref_state[:, None]
                encoder_hidden_states = torch.cat([encoder_hidden_states, ref_state], dim=1)
                encoder_hidden_states = encoder_hidden_states.reshape(B // mv, 1, (mv+1) * L, C)
                encoder_hidden_states = encoder_hidden_states.repeat(1, mv, 1, 1).reshape(-1, (mv+1) * L, C)
                return self.chained_proc(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)
            else:
                ref_state = ref_state[:, None].expand(-1, modalities * views, -1, -1).reshape(-1, ref_state.shape[-2], ref_state.shape[-1])
                encoder_hidden_states = torch.cat([encoder_hidden_states, ref_state], dim=1)
                return self.chained_proc(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)
        else:
            raise NotImplementedError("mode or self.mode is required to be 'extract' or 'inject'")

def add_extra_processor(model: torch.nn.Module, enable_filter=lambda x:True, **kwargs):
    return_dict = torch.nn.ModuleDict()
    proj_in_dim = kwargs.get('proj_in_dim', False)
    kwargs.pop('proj_in_dim', None)

    def recursive_add_processors(name: str, module: torch.nn.Module):
        for sub_name, child in module.named_children():
            if "ref_unet" not in (sub_name + name):
                recursive_add_processors(f"{name}.{sub_name}", child)

        if isinstance(module, Attention):
            new_processor = ExtraAttnProc(
                chained_proc=module.get_processor(),
                enabled=enable_filter(f"{name}.processor"),
                name=f"{name}.processor",
                proj_in_dim=proj_in_dim if proj_in_dim else module.cross_attention_dim,
                target_dim=module.cross_attention_dim,
                **kwargs
            )
            module.set_processor(new_processor)
            return_dict[f"{name}.processor".replace(".", "__")] = new_processor

    for name, module in model.named_children():
        recursive_add_processors(name, module)
    return return_dict

def switch_extra_processor(model, enable_filter=lambda x:True):
    def recursive_add_processors(name: str, module: torch.nn.Module):
        for sub_name, child in module.named_children():
            recursive_add_processors(f"{name}.{sub_name}", child)

        if isinstance(module, ExtraAttnProc):
            module.enabled = enable_filter(name)

    for name, module in model.named_children():
        recursive_add_processors(name, module)

class multiviewAttnProc(torch.nn.Module):
    def __init__(
        self,
        chained_proc,
        enabled=False,
        name=None,
        hidden_states_dim=None,
        chain_pos="parralle",     # before or parralle or after
        num_modalities=1,
        views=4,
        base_img_size=64,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.chained_proc = chained_proc
        self.name = name
        self.hidden_states_dim = hidden_states_dim
        self.num_modalities = num_modalities
        self.views = views
        self.base_img_size = base_img_size
        self.chain_pos = chain_pos
        self.diff_joint_attn = True

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> torch.Tensor:
        if not self.enabled:
            return self.chained_proc(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)
        
        B, L, C = hidden_states.shape
        mv = self.views
        hidden_states = hidden_states.reshape(B // mv, mv, L, C).reshape(-1, mv * L, C)
        hidden_states = self.chained_proc(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)
        return hidden_states.reshape(B // mv, mv, L, C).reshape(-1, L, C)

def add_multiview_processor(model: torch.nn.Module, enable_filter=lambda x:True, **kwargs):
    return_dict = torch.nn.ModuleDict()
    def recursive_add_processors(name: str, module: torch.nn.Module):
        for sub_name, child in module.named_children():
            if "ref_unet" not in (sub_name + name):
                recursive_add_processors(f"{name}.{sub_name}", child)

        if isinstance(module, Attention):
            new_processor = multiviewAttnProc(
                chained_proc=module.get_processor(),
                enabled=enable_filter(f"{name}.processor"),
                name=f"{name}.processor",
                hidden_states_dim=module.inner_dim,
                **kwargs
            )
            module.set_processor(new_processor)
            return_dict[f"{name}.processor".replace(".", "__")] = new_processor

    for name, module in model.named_children():
        recursive_add_processors(name, module)

    return return_dict

def switch_multiview_processor(model, enable_filter=lambda x:True):
    def recursive_add_processors(name: str, module: torch.nn.Module):
        for sub_name, child in module.named_children():
            recursive_add_processors(f"{name}.{sub_name}", child)

        if isinstance(module, Attention):
            processor = module.get_processor()
            if isinstance(processor, multiviewAttnProc):
                processor.enabled = enable_filter(f"{name}.processor")

    for name, module in model.named_children():
        recursive_add_processors(name, module)

class NNModuleWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class AttnProcessorSwitch(torch.nn.Module):
    def __init__(
        self,
        proc_dict: dict,
        enabled_proc="default",
        name=None,
        switch_name="default_switch",
    ):
        super().__init__()
        self.proc_dict = torch.nn.ModuleDict({k: (v if isinstance(v, torch.nn.Module) else NNModuleWrapper(v)) for k, v in proc_dict.items()})
        self.enabled_proc = enabled_proc
        self.name = name
        self.switch_name = switch_name
        self.choose_module(enabled_proc)
    
    def choose_module(self, enabled_proc):
        self.enabled_proc = enabled_proc
        assert enabled_proc in self.proc_dict.keys()

    def __call__(
        self,
        *args,
        **kwargs
    ) -> torch.FloatTensor:
        used_proc = self.proc_dict[self.enabled_proc]
        return used_proc(*args, **kwargs)

def add_switch(model: torch.nn.Module, module_filter=lambda x:True, switch_dict_fn=lambda x: {"default": x}, switch_name="default_switch", enabled_proc="default"):
    return_dict = torch.nn.ModuleDict()
    def recursive_add_processors(name: str, module: torch.nn.Module):
        for sub_name, child in module.named_children():
            if "ref_unet" not in (sub_name + name):
                recursive_add_processors(f"{name}.{sub_name}", child)

        if isinstance(module, Attention):
            processor = module.get_processor()
            if module_filter(processor):
                proc_dict = switch_dict_fn(processor)
                new_processor = AttnProcessorSwitch(
                    proc_dict=proc_dict,
                    enabled_proc=enabled_proc,
                    name=f"{name}.processor",
                    switch_name=switch_name,
                )
                module.set_processor(new_processor)
                return_dict[f"{name}.processor".replace(".", "__")] = new_processor

    for name, module in model.named_children():
        recursive_add_processors(name, module)

    return return_dict

def change_switch(model: torch.nn.Module, switch_name="default_switch", enabled_proc="default"):
    def recursive_change_processors(name: str, module: torch.nn.Module):
        for sub_name, child in module.named_children():
            recursive_change_processors(f"{name}.{sub_name}", child)

        if isinstance(module, Attention):
            processor = module.get_processor()
            if isinstance(processor, AttnProcessorSwitch) and processor.switch_name == switch_name:
                processor.choose_module(enabled_proc)

    for name, module in model.named_children():
        recursive_change_processors(name, module)

########## Hack: Attention fix #############
from diffusers.models.attention import Attention

def forward(
    self,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    **cross_attention_kwargs,
) -> torch.Tensor:
    r"""
    The forward method of the `Attention` class.

    Args:
        hidden_states (`torch.Tensor`):
            The hidden states of the query.
        encoder_hidden_states (`torch.Tensor`, *optional*):
            The hidden states of the encoder.
        attention_mask (`torch.Tensor`, *optional*):
            The attention mask to use. If `None`, no mask is applied.
        **cross_attention_kwargs:
            Additional keyword arguments to pass along to the cross attention.

    Returns:
        `torch.Tensor`: The output of the attention layer.
    """
    # The `Attention` class can call different attention processors / attention functions
    # here we simply pass along all tensors to the selected processor class
    # For standard processors that are defined here, `**cross_attention_kwargs` is empty
    return self.processor(
        self,
        hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=attention_mask,
        **cross_attention_kwargs,
    )

Attention.forward = forward