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

import os
import time
import torch
import random
import numpy as np
from PIL import Image
from einops import rearrange
from PIL import Image, ImageSequence

from .utils import seed_everything, timing_decorator, auto_amp_inference
from .utils import get_parameter_number, set_parameter_grad_false
from Hunyuan3D_V1.svrm.predictor import MV23DPredictor


class Views2Mesh():
    def __init__(self, mv23d_cfg_path, mv23d_ckt_path, device="cuda:0", use_lite=False):
        '''
            mv23d_cfg_path: config yaml file 
            mv23d_ckt_path: path to ckpt
            use_lite: 
        '''
        self.mv23d_predictor = MV23DPredictor(mv23d_ckt_path, mv23d_cfg_path, device=device)  
        self.mv23d_predictor.model.eval()
        self.order = [0, 1, 2, 3, 4, 5] if use_lite else [0, 2, 4, 5, 3, 1]
        set_parameter_grad_false(self.mv23d_predictor.model)
        print('view2mesh model', get_parameter_number(self.mv23d_predictor.model))

    @torch.no_grad()
    @timing_decorator("views to mesh")
    @auto_amp_inference
    def __call__(
        self,
        views_pil=None, 
        cond_pil=None, 
        gif_pil=None, 
        seed=0, 
        target_face_count = 90000,
        do_texture_mapping = True,
        save_folder='./outputs/test'
    ):
        '''
            can set views_pil, cond_pil simutaously or set gif_pil only
            seed: int
            target_face_count: int 
            save_folder: path to save mesh files
        '''
        save_dir = save_folder
        os.makedirs(save_dir, exist_ok=True)

        if views_pil is not None and cond_pil is not None:
            show_image = rearrange(np.asarray(views_pil, dtype=np.uint8), 
                                   '(n h) (m w) c -> (n m) h w c', n=3, m=2)
            views = [Image.fromarray(show_image[idx]) for idx in self.order] 
            image_list = [cond_pil]+ views
            image_list = [img.convert('RGB') for img in image_list]
        elif gif_pil is not None:
            image_list = [img.convert('RGB') for img in ImageSequence.Iterator(gif_pil)]
        
        image_input = image_list[0]
        image_list = image_list[1:] + image_list[:1]
        
        seed_everything(seed)
        vtx_refine, faces_refine, vtx_colors = self.mv23d_predictor.predict(
            image_list, 
            save_dir = save_dir, 
            image_input = image_input,
            target_face_count = target_face_count,
            do_texture_mapping = do_texture_mapping
        )
        torch.cuda.empty_cache()
        return vtx_refine, faces_refine, vtx_colors
        