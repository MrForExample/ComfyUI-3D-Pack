from types import FunctionType
from typing import Any, Dict, List
from diffusers import UNet2DConditionModel
import torch
import logging
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel, ImageProjection
from diffusers.models.attention_processor import Attention, AttnProcessor, AttnProcessor2_0, XFormersAttnProcessor
from dataclasses import dataclass, field
from diffusers.loaders import IPAdapterMixin
from Unique3D.custum_3d_diffusion.custum_modules.attention_processors import add_extra_processor, switch_extra_processor, add_multiview_processor, switch_multiview_processor, add_switch, change_switch

@dataclass
class AttnConfig:
    """        
    * CrossAttention: Attention module (inherits knowledge), LoRA module (achieves fine-tuning), IPAdapter module (achieves conceptual control).
    * SelfAttention: Attention module (inherits knowledge), LoRA module (achieves fine-tuning), Reference Attention module (achieves pixel-level control).
    * Multiview Attention module: Multiview Attention module (achieves multi-view consistency).
    * Cross Modality Attention module: Cross Modality Attention module (achieves multi-modality consistency).
    
    For setups:
        train_xxx_lr is implemented in the U-Net architecture.
        enable_xxx_lora is implemented in the U-Net architecture.
        enable_xxx_ip is implemented in the processor and U-Net architecture.
        enable_xxx_ref_proj_in is implemented in the processor.
    """
    latent_size: int = 64
    
    train_lr: float = 0
    # for cross attention
    # 0 learning rate for not training
    train_cross_attn_lr: float = 0
    train_cross_attn_lora_lr: float = 0       
    train_cross_attn_ip_lr: float = 0      # 0 for not trained
    init_cross_attn_lora: bool = False
    enable_cross_attn_lora: bool = False
    init_cross_attn_ip: bool = False
    enable_cross_attn_ip: bool = False
    cross_attn_lora_rank: int = 64        # 0 for not enabled
    cross_attn_lora_only_kv: bool = False
    ipadapter_pretrained_name: str = "h94/IP-Adapter"
    ipadapter_subfolder_name: str = "models"
    ipadapter_weight_name: str = "ip-adapter-plus_sd15.safetensors"
    ipadapter_effect_on: str = "all"    # all, first

    # for self attention
    train_self_attn_lr: float = 0
    train_self_attn_lora_lr: float = 0
    init_self_attn_lora: bool = False
    enable_self_attn_lora: bool = False
    self_attn_lora_rank: int = 64
    self_attn_lora_only_kv: bool = False

    train_self_attn_ref_lr: float = 0
    train_ref_unet_lr: float = 0
    init_self_attn_ref: bool = False
    enable_self_attn_ref: bool = False      
    self_attn_ref_other_model_name: str = ""
    self_attn_ref_position: str = "attn1"
    self_attn_ref_pixel_wise_crosspond: bool = False    # enable pixel_wise_crosspond in refattn
    self_attn_ref_chain_pos: str = "parralle"           # before or parralle or after
    self_attn_ref_effect_on: str = "all"                # all or first, for _crosspond attn
    self_attn_ref_zero_init: bool = True
    use_simple3d_attn: bool = False

    # for multiview attention
    init_multiview_attn: bool = False
    enable_multiview_attn: bool = False
    multiview_attn_position: str = "attn1"
    multiview_chain_pose: str = "parralle"             # before or parralle or after
    num_modalities: int = 1
    use_mv_joint_attn: bool = False
    
    # for unet
    init_unet_path: str = "runwayml/stable-diffusion-v1-5"
    init_num_cls_label: int = 0                         # for initialize
    cls_labels: List[int] = field(default_factory=lambda: [])
    cls_label_type: str = "embedding"
    cat_condition: bool = False                         # cat condition to input

class Configurable:
    attn_config: AttnConfig

    def set_config(self, attn_config: AttnConfig):
        raise NotImplementedError()
    
    def update_config(self, attn_config: AttnConfig):
        self.attn_config = attn_config
    
    def do_set_config(self, attn_config: AttnConfig):
        self.set_config(attn_config)
        for name, module in self.named_modules():
            if isinstance(module, Configurable):
                if hasattr(module, "do_set_config"):
                    module.do_set_config(attn_config)
                else:
                    print(f"Warning: {name} has no attribute do_set_config, but is an instance of Configurable")
                    module.attn_config = attn_config

    def do_update_config(self, attn_config: AttnConfig):
        self.update_config(attn_config)
        for name, module in self.named_modules():
            if isinstance(module, Configurable):
                if hasattr(module, "do_update_config"):
                    module.do_update_config(attn_config)
                else:
                    print(f"Warning: {name} has no attribute do_update_config, but is an instance of Configurable")
                    module.attn_config = attn_config

from diffusers import ModelMixin  # Must import ModelMixin for CompiledUNet
class UnifieldWrappedUNet(UNet2DConditionModel):
    forward_hook: FunctionType

    def forward(self, *args, **kwargs):
        if hasattr(self, 'forward_hook'):
            return self.forward_hook(super().forward, *args, **kwargs)
        return super().forward(*args, **kwargs)


class ConfigurableUNet2DConditionModel(Configurable, IPAdapterMixin):
    unet: UNet2DConditionModel

    cls_embedding_param_dict = {}
    cross_attn_lora_param_dict = {}
    self_attn_lora_param_dict = {}
    cross_attn_param_dict = {}
    self_attn_param_dict = {}
    ipadapter_param_dict = {}
    ref_attn_param_dict = {}
    ref_unet_param_dict = {}
    multiview_attn_param_dict = {}
    other_param_dict = {}
    
    rev_param_name_mapping = {}

    class_labels = []
    def set_class_labels(self, class_labels: torch.Tensor):
        if self.attn_config.init_num_cls_label != 0:
            self.class_labels = class_labels.to(self.unet.device).long()

    def __init__(self, init_config: AttnConfig, weight_dtype) -> None:
        super().__init__()
        self.weight_dtype = weight_dtype
        self.set_config(init_config)

    def enable_xformers_memory_efficient_attention(self):
        self.unet.enable_xformers_memory_efficient_attention
        def recursive_add_processors(name: str, module: torch.nn.Module):
            for sub_name, child in module.named_children():
                recursive_add_processors(f"{name}.{sub_name}", child)

            if isinstance(module, Attention):
                if hasattr(module, 'xformers_not_supported'):
                    return
                old_processor = module.get_processor()
                if isinstance(old_processor, (AttnProcessor, AttnProcessor2_0)):
                    module.set_use_memory_efficient_attention_xformers(True)

        for name, module in self.unet.named_children():
            recursive_add_processors(name, module)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)
    
    # --- for IPAdapterMixin
    
    def register_modules(self, **kwargs):
        for name, module in kwargs.items():
            # set models
            setattr(self, name, module)

    def register_to_config(self, **kwargs):
        pass

    def unload_ip_adapter(self):
        raise NotImplementedError()

    # --- for Configurable

    def get_refunet(self):
        if self.attn_config.self_attn_ref_other_model_name == "self":
            return self.unet
        else:
            return self.unet.ref_unet

    def set_config(self, attn_config: AttnConfig):
        self.attn_config = attn_config

        unet_type = UnifieldWrappedUNet
        # class_embed_type = "projection" for 'camera'
        # class_embed_type = None for 'embedding'
        unet_kwargs = {}
        if attn_config.init_num_cls_label > 0:
            if attn_config.cls_label_type == "embedding":
                unet_kwargs = {
                    "num_class_embeds": attn_config.init_num_cls_label, 
                    "device_map": None, 
                    "low_cpu_mem_usage": False,
                    "class_embed_type": None,
                }
            else:
                raise ValueError(f"cls_label_type {attn_config.cls_label_type} is not supported")
        
        self.unet: UnifieldWrappedUNet = unet_type.from_pretrained(
            attn_config.init_unet_path, subfolder="unet", torch_dtype=self.weight_dtype, 
            **unet_kwargs
        )
        assert isinstance(self.unet, UnifieldWrappedUNet)
        self.unet.forward_hook = self.unet_forward_hook

        if self.attn_config.cat_condition:
            # double in_channels
            if self.unet.config.in_channels != 8:
                self.unet.register_to_config(in_channels=self.unet.config.in_channels * 2)
                # repeate unet.conv_in weight twice
                doubled_conv_in = torch.nn.Conv2d(self.unet.conv_in.in_channels * 2, self.unet.conv_in.out_channels, self.unet.conv_in.kernel_size, self.unet.conv_in.stride, self.unet.conv_in.padding)
                doubled_conv_in.weight.data = torch.cat([self.unet.conv_in.weight.data, torch.zeros_like(self.unet.conv_in.weight.data)], dim=1)
                doubled_conv_in.bias.data = self.unet.conv_in.bias.data
                self.unet.conv_in = doubled_conv_in
        
        used_param_ids = set()
        
        if attn_config.init_cross_attn_lora:
            # setup lora
            from peft import LoraConfig
            from peft.utils import get_peft_model_state_dict
            if attn_config.cross_attn_lora_only_kv:
                target_modules=["attn2.to_k", "attn2.to_v"]
            else:
                target_modules=["attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0"]
            lora_config: LoraConfig = LoraConfig(
                r=attn_config.cross_attn_lora_rank,
                lora_alpha=attn_config.cross_attn_lora_rank,
                init_lora_weights="gaussian",
                target_modules=target_modules,
            )
            adapter_name="cross_attn_lora"
            self.unet.add_adapter(lora_config, adapter_name=adapter_name)
            # update cross_attn_lora_param_dict
            self.cross_attn_lora_param_dict = {id(param): param for name, param in self.unet.named_parameters() if adapter_name in name and id(param) not in used_param_ids}
            used_param_ids.update(self.cross_attn_lora_param_dict.keys())

        if attn_config.init_self_attn_lora:
            # setup lora
            from peft import LoraConfig
            if attn_config.self_attn_lora_only_kv:
                target_modules=["attn1.to_k", "attn1.to_v"]
            else:
                target_modules=["attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0"]
            lora_config: LoraConfig = LoraConfig(
                r=attn_config.self_attn_lora_rank,
                lora_alpha=attn_config.self_attn_lora_rank,
                init_lora_weights="gaussian",
                target_modules=target_modules,
            )
            adapter_name="self_attn_lora"
            self.unet.add_adapter(lora_config, adapter_name=adapter_name)
            # update cross_self_lora_param_dict
            self.self_attn_lora_param_dict = {id(param): param for name, param in self.unet.named_parameters() if adapter_name in name and id(param) not in used_param_ids}
            used_param_ids.update(self.self_attn_lora_param_dict.keys())

        if attn_config.init_num_cls_label != 0:
            self.cls_embedding_param_dict = {id(param): param for param in self.unet.class_embedding.parameters()}
            used_param_ids.update(self.cls_embedding_param_dict.keys())
            self.set_class_labels(torch.tensor(attn_config.cls_labels).long())
        
        if attn_config.init_cross_attn_ip:
            self.image_encoder = None
            # setup ipadapter
            self.load_ip_adapter(
                attn_config.ipadapter_pretrained_name,
                subfolder=attn_config.ipadapter_subfolder_name,
                weight_name=attn_config.ipadapter_weight_name
            )
            # warp ip_adapter_attn_proc with switch
            from diffusers.models.attention_processor import IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0
            add_switch(self.unet, module_filter=lambda x: isinstance(x, (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0)), switch_dict_fn=lambda x: {"ipadapter": x, "default": XFormersAttnProcessor()}, switch_name="ipadapter_switch", enabled_proc="ipadapter")
            # update ipadapter_param_dict
            # weights are in attention processors and unet.encoder_hid_proj
            self.ipadapter_param_dict = {id(param): param for param in self.unet.encoder_hid_proj.parameters() if id(param) not in used_param_ids}
            used_param_ids.update(self.ipadapter_param_dict.keys())
            print("DEBUG: ipadapter_param_dict len in encoder_hid_proj", len(self.ipadapter_param_dict))
            for name, processor in self.unet.attn_processors.items():
                if hasattr(processor, "to_k_ip"):
                    self.ipadapter_param_dict.update({id(param): param for param in processor.parameters()})
            print(f"DEBUG: ipadapter_param_dict len in all", len(self.ipadapter_param_dict))

        ref_unet = None
        if attn_config.init_self_attn_ref:
            # setup reference attention processor
            if attn_config.self_attn_ref_other_model_name == "self":
                raise NotImplementedError("self reference is not fully implemented")
            else:
                ref_unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
                    attn_config.self_attn_ref_other_model_name, subfolder="unet", torch_dtype=self.unet.dtype
                )
                ref_unet.to(self.unet.device)
                if self.attn_config.train_ref_unet_lr == 0:
                    ref_unet.eval()
                    ref_unet.requires_grad_(False)
                else:
                    ref_unet.train()

                add_extra_processor(
                    model=ref_unet, 
                    enable_filter=lambda name: name.endswith(f"{attn_config.self_attn_ref_position}.processor"), 
                    mode='extract',
                    with_proj_in=False,
                    pixel_wise_crosspond=False,
                )
                # NOTE: Here require cross_attention_dim in two unet's self attention should be the same
                processor_dict = add_extra_processor(
                    model=self.unet,
                    enable_filter=lambda name: name.endswith(f"{attn_config.self_attn_ref_position}.processor"),
                    mode='inject',
                    with_proj_in=False,
                    pixel_wise_crosspond=attn_config.self_attn_ref_pixel_wise_crosspond,
                    crosspond_effect_on=attn_config.self_attn_ref_effect_on,
                    crosspond_chain_pos=attn_config.self_attn_ref_chain_pos,
                    simple_3d=attn_config.use_simple3d_attn,
                )
                self.ref_unet_param_dict = {id(param): param for name, param in ref_unet.named_parameters() if id(param) not in used_param_ids and (attn_config.self_attn_ref_position in name)}
                if attn_config.self_attn_ref_chain_pos != "after":
                    # pop untrainable paramters
                    for name, param in ref_unet.named_parameters():
                        if id(param) in self.ref_unet_param_dict and ('up_blocks.3.attentions.2.transformer_blocks.0.' in name):
                            self.ref_unet_param_dict.pop(id(param))
                used_param_ids.update(self.ref_unet_param_dict.keys())
            # update ref_attn_param_dict
            self.ref_attn_param_dict = {id(param): param for name, param in processor_dict.named_parameters() if id(param) not in used_param_ids}
            used_param_ids.update(self.ref_attn_param_dict.keys())

        if attn_config.init_multiview_attn:
            processor_dict = add_multiview_processor(
                model = self.unet, 
                enable_filter = lambda name: name.endswith(f"{attn_config.multiview_attn_position}.processor"),
                num_modalities = attn_config.num_modalities,    
                base_img_size = attn_config.latent_size,      
                chain_pos = attn_config.multiview_chain_pose,
            )
            # update multiview_attn_param_dict
            self.multiview_attn_param_dict = {id(param): param for name, param in processor_dict.named_parameters() if id(param) not in used_param_ids}
            used_param_ids.update(self.multiview_attn_param_dict.keys())
        
        # initialize cross_attn_param_dict parameters
        self.cross_attn_param_dict = {id(param): param for name, param in self.unet.named_parameters() if "attn2" in name and id(param) not in used_param_ids}
        used_param_ids.update(self.cross_attn_param_dict.keys())
        
        # initialize self_attn_param_dict parameters
        self.self_attn_param_dict = {id(param): param for name, param in self.unet.named_parameters() if "attn1" in name and id(param) not in used_param_ids}
        used_param_ids.update(self.self_attn_param_dict.keys())
        
        # initialize other_param_dict parameters
        self.other_param_dict = {id(param): param for name, param in self.unet.named_parameters() if id(param) not in used_param_ids}
        
        if ref_unet is not None:
            self.unet.ref_unet = ref_unet
            
        self.rev_param_name_mapping = {id(param): name for name, param in self.unet.named_parameters()}
        
        self.update_config(attn_config, force_update=True)
        return self.unet
    
    _attn_keys_to_update = ["enable_cross_attn_lora", "enable_cross_attn_ip", "enable_self_attn_lora", "enable_self_attn_ref", "enable_multiview_attn", "cls_labels"]
    
    def update_config(self, attn_config: AttnConfig, force_update=False):
        assert isinstance(self.unet, UNet2DConditionModel), "unet must be an instance of UNet2DConditionModel"

        need_to_update = False
        # update cls_labels
        for key in self._attn_keys_to_update:
            if getattr(self.attn_config, key) != getattr(attn_config, key):
                need_to_update = True
                break
        if not force_update and not need_to_update:
            return

        self.set_class_labels(torch.tensor(attn_config.cls_labels).long())
        
        # setup loras
        if self.attn_config.init_cross_attn_lora or self.attn_config.init_self_attn_lora:
            if attn_config.enable_cross_attn_lora or attn_config.enable_self_attn_lora:
                cross_attn_lora_weight = 1. if attn_config.enable_cross_attn_lora > 0 else 0
                self_attn_lora_weight = 1. if attn_config.enable_self_attn_lora > 0 else 0
                self.unet.set_adapters(["cross_attn_lora", "self_attn_lora"], weights=[cross_attn_lora_weight, self_attn_lora_weight])
            else:
                self.unet.disable_adapters()

        # setup ipadapter
        if self.attn_config.init_cross_attn_ip:
            if attn_config.enable_cross_attn_ip:
                change_switch(self.unet, "ipadapter_switch", "ipadapter")
            else:
                change_switch(self.unet, "ipadapter_switch", "default")
            
        # setup reference attention processor
        if self.attn_config.init_self_attn_ref:
            if attn_config.enable_self_attn_ref:
                switch_extra_processor(self.unet, enable_filter=lambda name: name.endswith(f"{attn_config.self_attn_ref_position}.processor"))
            else:
                switch_extra_processor(self.unet, enable_filter=lambda name: False)
        
        # setup multiview attention processor
        if self.attn_config.init_multiview_attn:
            if attn_config.enable_multiview_attn:
                switch_multiview_processor(self.unet, enable_filter=lambda name: name.endswith(f"{attn_config.multiview_attn_position}.processor"))
            else:
                switch_multiview_processor(self.unet, enable_filter=lambda name: False)
        
        # update cls_labels
        for key in self._attn_keys_to_update:
            setattr(self.attn_config, key, getattr(attn_config, key))

    def unet_forward_hook(self, raw_forward, sample: torch.FloatTensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor, *args, cross_attention_kwargs=None, condition_latents=None, class_labels=None, noisy_condition_input=False, cond_pixels_clip=None, **kwargs):
        if class_labels is None and len(self.class_labels) > 0:
            class_labels = self.class_labels.repeat(sample.shape[0] // self.class_labels.shape[0]).to(sample.device)
        elif self.attn_config.init_num_cls_label != 0:
            assert class_labels is not None, "class_labels should be passed if self.class_labels is empty and self.attn_config.init_num_cls_label is not 0"
        if class_labels is not None:
            if self.attn_config.cls_label_type == "embedding":
                pass
            else:
                raise ValueError(f"cls_label_type {self.attn_config.cls_label_type} is not supported")
        if self.attn_config.init_self_attn_ref and self.attn_config.enable_self_attn_ref:
            # NOTE: extra step, extract condition
            ref_dict = {}
            ref_unet = self.get_refunet().to(sample.device)
            assert condition_latents is not None
            if self.attn_config.self_attn_ref_other_model_name == "self":
                raise NotImplementedError()
            else:
                with torch.no_grad():
                    cond_encoder_hidden_states = encoder_hidden_states.reshape(condition_latents.shape[0], -1, *encoder_hidden_states.shape[1:])[:, 0]
                    if timestep.dim() == 0:
                        cond_timestep = timestep
                    else:
                        cond_timestep = timestep.reshape(condition_latents.shape[0], -1)[:, 0]
                ref_unet(condition_latents, cond_timestep, cond_encoder_hidden_states,  cross_attention_kwargs=dict(ref_dict=ref_dict))
            # NOTE: extra step, inject condition
            # Predict the noise residual and compute loss
            if cross_attention_kwargs is None:
                cross_attention_kwargs = {}
            cross_attention_kwargs.update(ref_dict=ref_dict, mode='inject')
        elif condition_latents is not None:
            if not hasattr(self, 'condition_latents_raised'):
                logging.warn("Warning! condition_latents is not None, but self_attn_ref is not enabled! This warning will only be raised once.")
                self.condition_latents_raised = True
        
        if self.attn_config.init_cross_attn_ip:
            raise NotImplementedError()
        
        if self.attn_config.cat_condition:
            assert condition_latents is not None
            B = condition_latents.shape[0]
            cat_latents = condition_latents.reshape(B, 1, *condition_latents.shape[1:]).repeat(1, sample.shape[0] // B, 1, 1, 1).reshape(*sample.shape)
            sample = torch.cat([sample, cat_latents], dim=1)
            
        return raw_forward(sample, timestep, encoder_hidden_states, *args, cross_attention_kwargs=cross_attention_kwargs, class_labels=class_labels, **kwargs)
