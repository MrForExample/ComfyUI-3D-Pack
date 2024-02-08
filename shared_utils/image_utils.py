import torch
import torch.nn.functional as F
import numpy as np

def prepare_torch_img(img, size_H, size_W, device="cuda"):
    # [N, H, W, C] -> [N, C, H, W]
    img_new = img.permute(0, 3, 1, 2).to(device)
    img_new = F.interpolate(img_new, (size_H, size_W), mode="bilinear", align_corners=False).contiguous()
    return img_new