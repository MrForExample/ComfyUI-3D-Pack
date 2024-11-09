# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from .dinov2.hub.backbones import dinov2_vitb14

class FrozenDinoV2ImageEmbedder(nn.Module):
    """
        Uses the dinov2 image encoder with camera modulation.
        Not actually frozen... If you want that set cond_stage_trainable=False in cfg
        """
    def __init__(
            self,
            version='dinov2_vitb14',
            ckpt_path=None,
            lrm_mode='plain_lrm',
        ):
        super().__init__()
        self.lrm_mode = lrm_mode
        assert version in ['dinov2_vitb14', 'dinov2_vits14', 'dinov2_vitl14', 'dinov2_vitg14']

    
        self.model = dinov2_vitb14(pretrained=False)

        if ckpt_path is not None:
            self.load_pretrained(ckpt_path)
        else:
            print('None pretrained model for dinov2 encoder ...')


    def load_pretrained(self, ckpt_path):
        print('Loading dinov2 encoder ...')
        orig_state_dict = torch.load(ckpt_path, map_location='cpu')
        try:
            ret = self.model.load_state_dict(orig_state_dict, strict=False)
            print(ret)
            print('Successfully loaded orig state dict')
        except:
            new_state_dict = OrderedDict()
            for k, v in orig_state_dict['state_dict'].items():
                if 'img_encoder' in k:
                    new_state_dict[k.replace('img_encoder.model.', '')] = v
            ret = self.model.load_state_dict(new_state_dict, strict=False)
            print(ret)
            print('Successfully loaded new state dict')
            

    def forward(self, x, *args, **kwargs):
        ret = self.model.forward_features_with_camera(x, *args, **kwargs)
        output = torch.cat([ret['x_norm_clstoken'].unsqueeze(1), ret['x_norm_patchtokens']], dim=1)
        return output
