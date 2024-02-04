import torch
import torch.nn.functional as F
import numpy as np

def prepare_torch_img(img, size_H, size_W, device="cuda"):
    # [H, W, C] -> [1, C, H, W]
    img_new = img.permute(2, 0, 1).unsqueeze(0).to(device)
    img_new = F.interpolate(img_new, (size_H, size_W), mode="bilinear", align_corners=False).contiguous()
    return img_new