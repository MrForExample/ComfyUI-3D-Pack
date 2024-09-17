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
        masks = masks.to(dtype=images.dtype, device=images.device)
        
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

def pils_to_torch_imgs(pils: Union[Image.Image, List[Image.Image]], dtype=torch.float16, device="cuda", force_rgb=True):
    if isinstance(pils, Image.Image):
        pils = [pils]
    
    images = []
    for pil in pils:
        if pil.mode == "RGBA" and force_rgb:
            pil = pil.convert('RGB')

        images.append(TF.to_tensor(pil).permute(1, 2, 0))

    images = torch.stack(images, dim=0).to(dtype=dtype, device=device)

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

def pils_resize_foreground(
    pils: Union[Image.Image, List[Image.Image]],
    ratio: float,
) -> List[Image.Image]:
    if isinstance(pils, Image.Image):
        pils = [pils]
        
    new_pils = []
    for image in pils:
        image = np.array(image)
        assert image.shape[-1] == 4
        alpha = np.where(image[..., 3] > 0)
        y1, y2, x1, x2 = (
            alpha[0].min(),
            alpha[0].max(),
            alpha[1].min(),
            alpha[1].max(),
        )
        # crop the foreground
        fg = image[y1:y2, x1:x2]
        # pad to square
        size = max(fg.shape[0], fg.shape[1])
        ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
        ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
        new_image = np.pad(
            fg,
            ((ph0, ph1), (pw0, pw1), (0, 0)),
            mode="constant",
            constant_values=((0, 0), (0, 0), (0, 0)),
        )

        # compute padding according to the ratio
        new_size = int(new_image.shape[0] / ratio)
        # pad to size, double side
        ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
        ph1, pw1 = new_size - size - ph0, new_size - size - pw0
        new_image = np.pad(
            new_image,
            ((ph0, ph1), (pw0, pw1), (0, 0)),
            mode="constant",
            constant_values=((0, 0), (0, 0), (0, 0)),
        )
        new_image = Image.fromarray(new_image, mode="RGBA")
        new_pils.append(new_image)
    
    return new_pils