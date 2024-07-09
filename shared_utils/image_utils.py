import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

def prepare_torch_img(img, size_H, size_W, device="cuda", keep_shape=False):
    # [N, H, W, C] -> [N, C, H, W]
    img_new = img.permute(0, 3, 1, 2).to(device)
    img_new = F.interpolate(img_new, (size_H, size_W), mode="bilinear", align_corners=False).contiguous()
    if keep_shape:
        img_new = img_new.permute(0, 2, 3, 1)
    return img_new

def torch_imgs_to_pils(images, masks=None, alpha_min=0.1):
    """
        images (torch): [N, H, W, C] or [H, W, C]
        masks (torch): [N, H, W] or [H, W]
    """
    if len(images.shape) == 3:
        images = images.unsqueeze(0)

    if masks is not None:
        if len(masks.shape) == 2:
            masks = masks.unsqueeze(0)

        inv_mask_index = masks < alpha_min
        images[inv_mask_index] = 0.
        
        masks = masks.unsqueeze(3)
        images = torch.cat((images, masks), dim=3)
        mode="RGBA"
    else:
        mode="RGB"

    pil_image_list = [Image.fromarray((images[i].detach().cpu().numpy() * 255).astype(np.uint8), mode=mode) for i in range(images.shape[0])]

    return pil_image_list

def troch_image_dilate(img):
    """
    Remove thin seams on generated texture
        img (torch): [H, W, C]
    """
    import cv2
    img = np.asarray(img.cpu().numpy(), dtype=np.float32)
    img = img * 255
    img = img.clip(0, 255)
    mask = np.sum(img.astype(np.float32), axis=-1, keepdims=True)
    mask = (mask <= 3.0).astype(np.float32)
    kernel = np.ones((3, 3), 'uint8')
    dilate_img = cv2.dilate(img, kernel, iterations=1)
    img = img * (1 - mask) + dilate_img * mask
    img = (img.clip(0, 255) / 255).astype(np.float32)
    return torch.from_numpy(img)

def pils_to_torch_imgs(pils: Union[Image.Image, List[Image.Image]], device="cuda"):
    if isinstance(pils, Image.Image):
        pils = [pils]
    
    images = []
    for pil in pils:
        if pil.mode == "RGBA":
            pil = pil.convert('RGB')

        images.append(TF.to_tensor(pil).permute(1, 2, 0))

    images = torch.stack(images, dim=0)

    return images

def pils_rgba_to_rgb(pils: Union[Image.Image, List[Image.Image]], bkgd="WHITE"):
    if isinstance(pils, Image.Image):
        pils = [pils]
    
    rgbs = []
    for pil in pils:
        if pil.mode == 'RGBA':
            new_image = Image.new("RGBA", pil.size, bkgd)
            new_image.paste(pil, (0, 0), pil)
            rgbs.append(new_image.convert('RGB'))
        else:
            rgbs.append(pil)

    return rgbs

def pil_split_image(image, rows=None, cols=None):
    """
        inverse function of make_image_grid
    """
    # image is in square
    if rows is None and cols is None:
        # image.size [W, H]
        rows = 1
        cols = image.size[0] // image.size[1]
        assert cols * image.size[1] == image.size[0]
        subimg_size = image.size[1]
    elif rows is None:
        subimg_size = image.size[0] // cols
        rows = image.size[1] // subimg_size
        assert rows * subimg_size == image.size[1]
    elif cols is None:
        subimg_size = image.size[1] // rows
        cols = image.size[0] // subimg_size
        assert cols * subimg_size == image.size[0]
    else:
        subimg_size = image.size[1] // rows
        assert cols * subimg_size == image.size[0]
    subimgs = []
    for i in range(rows):
        for j in range(cols):
            subimg = image.crop((j*subimg_size, i*subimg_size, (j+1)*subimg_size, (i+1)*subimg_size))
            subimgs.append(subimg)
    return subimgs

def pil_make_image_grid(images, rows=None, cols=None):
    if rows is None and cols is None:
        rows = 1
        cols = len(images)
    if rows is None:
        rows = len(images) // cols
        if len(images) % cols != 0:
            rows += 1
    if cols is None:
        cols = len(images) // rows
        if len(images) % rows != 0:
            cols += 1
    total_imgs = rows * cols
    if total_imgs > len(images):
        images += [Image.new(images[0].mode, images[0].size) for _ in range(total_imgs - len(images))]

    w, h = images[0].size
    grid = Image.new(images[0].mode, size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def pils_erode_masks(mask_list):
    out_mask_list = []
    for idx, mask in enumerate(mask_list):
        arr = np.array(mask)
        alpha = (arr[:, :, 3] > 127).astype(np.uint8)
        # erode 1px
        import cv2
        alpha = cv2.erode(alpha, np.ones((3, 3), np.uint8), iterations=1)
        alpha = (alpha * 255).astype(np.uint8)
        out_mask_list.append(Image.fromarray(alpha[:, :, None]))

    return out_mask_list