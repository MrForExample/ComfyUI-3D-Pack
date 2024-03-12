import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

def prepare_torch_img(img, size_H, size_W, device="cuda", keep_shape=False):
    # [N, H, W, C] -> [N, C, H, W]
    img_new = img.permute(0, 3, 1, 2).to(device)
    img_new = F.interpolate(img_new, (size_H, size_W), mode="bilinear", align_corners=False).contiguous()
    if keep_shape:
        img_new = img_new.permute(0, 2, 3, 1)
    return img_new

def torch_img_to_pil_rgba(img, mask):
    """
        img (torch): [1, H, W, C] or [H, W, C]
        mask (torch): [1, H, W] or [H, W]
    """
    if len(img.shape) == 4:
        img = img.squeeze(0)
    if len(mask.shape) == 3:
        mask = mask.squeeze(0).unsqueeze(2)

    single_image = torch.cat((img, mask), dim=2).detach().cpu().numpy()
    single_image = Image.fromarray((single_image * 255).astype(np.uint8), mode="RGBA")
    
    return single_image