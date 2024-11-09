# Open Source Model Licensed under the Apache License Version 2.0 and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved. 
# The below software and/or models in this distribution may have been 
# modified by THL A29 Limited ("Tencent Modifications"). 
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT 
# except for the third-party components listed below. 
# Hunyuan 3D does not impose any additional limitations beyond what is outlined 
# in the repsective licenses of these third-party components. 
# Users must comply with all terms and conditions of original licenses of these third-party 
# components and must ensure that the usage of the third party components adheres to 
# all relevant laws and regulations. 

# For avoidance of doubts, Hunyuan 3D means the large language models and 
# their software and algorithms, including trained model weights, parameters (including 
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code, 
# fine-tuning enabling code and other elements of the foregoing made publicly available 
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import torch
from .utils import seed_everything, timing_decorator, auto_amp_inference
from .utils import get_parameter_number, set_parameter_grad_false
from diffusers import HunyuanDiTPipeline, AutoPipelineForText2Image

class Text2Image():
    def __init__(self, pretrain="./weights/hunyuanDiT", device="cuda:0", save_memory=False):
        '''
            save_memory: if GPU memory is low, can set it
        '''
        self.save_memory = save_memory
        self.device = device
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            pretrain, 
            torch_dtype = torch.float16, 
            enable_pag = True, 
            pag_applied_layers = ["blocks.(16|17|18|19)"]
        )
        set_parameter_grad_false(self.pipe.transformer)
        print('text2image transformer model', get_parameter_number(self.pipe.transformer))
        if not save_memory: 
            self.pipe = self.pipe.to(device)
        self.neg_txt = "文本,特写,裁剪,出框,最差质量,低质量,JPEG伪影,PGLY,重复,病态,残缺,多余的手指,变异的手," \
                       "画得不好的手,画得不好的脸,变异,畸形,模糊,脱水,糟糕的解剖学,糟糕的比例,多余的肢体,克隆的脸," \
                       "毁容,恶心的比例,畸形的肢体,缺失的手臂,缺失的腿,额外的手臂,额外的腿,融合的手指,手指太多,长脖子"

    @torch.no_grad()
    @timing_decorator('text to image')
    @auto_amp_inference
    def __call__(self, *args, **kwargs):
        if self.save_memory:
            self.pipe = self.pipe.to(self.device)
            torch.cuda.empty_cache()
            res = self.call(*args, **kwargs)
            self.pipe = self.pipe.to("cpu")
        else:
            res = self.call(*args, **kwargs)
        torch.cuda.empty_cache()
        return res

    def call(self, prompt, seed=0, steps=25):
        '''
            inputs:
                prompr: str
                seed: int
                steps: int
            return:
                rgb: PIL.Image
        '''
        prompt = prompt + ",白色背景,3D风格,最佳质量"
        seed_everything(seed)
        generator = torch.Generator(device=self.device)
        if seed is not None: generator = generator.manual_seed(int(seed))
        rgb = self.pipe(prompt=prompt, negative_prompt=self.neg_txt, num_inference_steps=steps, 
            pag_scale=1.3, width=1024, height=1024, generator=generator, return_dict=False)[0][0]
        torch.cuda.empty_cache()
        return rgb
    