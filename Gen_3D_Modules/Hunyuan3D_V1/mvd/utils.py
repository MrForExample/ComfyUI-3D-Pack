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

import numpy as np
from PIL import Image

def to_rgb_image(maybe_rgba: Image.Image):
    '''
        convert a PIL.Image to rgb mode with white background
        maybe_rgba: PIL.Image
        return: PIL.Image
    '''
    if maybe_rgba.mode == 'RGB':
        return maybe_rgba
    elif maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        img = np.random.randint(255, 256, size=[rgba.size[1], rgba.size[0], 3], dtype=np.uint8)
        img = Image.fromarray(img, 'RGB')
        img.paste(rgba, mask=rgba.getchannel('A'))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)
        
def white_out_background(pil_img, is_gray_fg=True):
    data = pil_img.getdata()
    new_data = []
    #  convert fore-ground white to gray
    for r, g, b, a in data:
        if a < 16: 
            new_data.append((255, 255, 255, 0))  # back-ground to be black
        else:
            is_white = is_gray_fg and (r>235) and (g>235) and (b>235)
            new_r = 235 if is_white else r
            new_g = 235 if is_white else g
            new_b = 235 if is_white else b
            new_data.append((new_r, new_g, new_b, a))
    pil_img.putdata(new_data)
    return pil_img
    
def recenter_img(img, size=512, color=(255,255,255)):
    img = white_out_background(img)
    mask = np.array(img)[..., 3]
    image = np.array(img)[..., :3]
    
    H, W, C = image.shape
    coords = np.nonzero(mask)
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    h = x_max - x_min
    w = y_max - y_min
    if h == 0 or w == 0: raise ValueError
    roi = image[x_min:x_max, y_min:y_max]

    border_ratio = 0.15 # 0.2
    pad_h = int(h * border_ratio)
    pad_w = int(w * border_ratio)

    result_tmp = np.full((h + pad_h, w + pad_w, C), color, dtype=np.uint8)
    result_tmp[pad_h // 2: pad_h // 2 + h, pad_w // 2: pad_w // 2 + w] = roi

    cur_h, cur_w = result_tmp.shape[:2]
    side = max(cur_h, cur_w)
    result = np.full((side, side, C), color, dtype=np.uint8)
    result[(side-cur_h)//2:(side-cur_h)//2+cur_h, (side-cur_w)//2:(side - cur_w)//2+cur_w,:] = result_tmp
    result = Image.fromarray(result)
    return result.resize((size, size), Image.LANCZOS) if size else result
