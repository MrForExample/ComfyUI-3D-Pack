# -*- coding: utf-8 -*-
import os
from skimage.morphology import remove_small_objects
from skimage.measure import label
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def find_bounding_box(gray_image):
    _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    return x, y, w, h

def load_image(img_path, bg_color=None, rmbg_net=None, padding_ratio=0.1, device='cuda'):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return f"invalid image path {img_path}"

    def is_valid_alpha(alpha, min_ratio = 0.01):
        bins = 20
        if isinstance(alpha, np.ndarray):
            hist = cv2.calcHist([alpha], [0], None, [bins], [0, 256])
        else:
            hist = torch.histc(alpha, bins=bins, min=0, max=1) 
        min_hist_val = alpha.shape[0] * alpha.shape[1] * min_ratio
        return hist[0] >= min_hist_val and hist[-1] >= min_hist_val
    
    def rmbg(image: torch.Tensor) -> torch.Tensor:
        image = TF.normalize(image, [0.5,0.5,0.5], [1.0,1.0,1.0]).unsqueeze(0)
        result=rmbg_net(image)
        return result[0][0]

    if len(img.shape) == 2:
        num_channels = 1
    else:
        num_channels = img.shape[2]

    # check if too large
    height, width = img.shape[:2]
    if height > width:
        scale = 2000 / height
    else:
        scale = 2000 / width
    if scale < 1:
        new_size = (int(width * scale), int(height * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    if img.dtype != 'uint8':
        img = (img * (255. / np.iinfo(img.dtype).max)).astype(np.uint8)

    rgb_image = None
    alpha = None

    if num_channels == 1:  
        rgb_image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif num_channels == 3:  
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif num_channels == 4:  
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        b, g, r, alpha = cv2.split(img)
        if not is_valid_alpha(alpha):
            alpha = None
        else:
            alpha_gpu = torch.from_numpy(alpha).unsqueeze(0).to(device).float() / 255.
    else:
        return f"invalid image: channels {num_channels}"
    
    rgb_image_gpu = torch.from_numpy(rgb_image).to(device).float().permute(2, 0, 1) / 255.
    if alpha is None:
        resize_transform = transforms.Resize((384, 384), antialias=True)
        rgb_image_resized = resize_transform(rgb_image_gpu)
        normalize_image = rgb_image_resized * 2 - 1

        mean_color = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        resize_transform = transforms.Resize((1024, 1024), antialias=True)
        rgb_image_resized = resize_transform(rgb_image_gpu)
        max_value = rgb_image_resized.flatten().max()
        if max_value < 1e-3:
            return "invalid image: pure black image"
        normalize_image = rgb_image_resized / max_value - mean_color
        normalize_image = normalize_image.unsqueeze(0)
        resize_transform = transforms.Resize((rgb_image_gpu.shape[1], rgb_image_gpu.shape[2]), antialias=True)

        # seg from rmbg
        alpha_gpu_rmbg = rmbg(rgb_image_resized)
        alpha_gpu_rmbg = alpha_gpu_rmbg.squeeze(0)
        alpha_gpu_rmbg = resize_transform(alpha_gpu_rmbg)
        ma, mi = alpha_gpu_rmbg.max(), alpha_gpu_rmbg.min()
        alpha_gpu_rmbg = (alpha_gpu_rmbg - mi) / (ma - mi)

        alpha_gpu = alpha_gpu_rmbg
        
        alpha_gpu_tmp = alpha_gpu * 255
        alpha = alpha_gpu_tmp.to(torch.uint8).squeeze().cpu().numpy()

        _, alpha = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        labeled_alpha = label(alpha)
        cleaned_alpha = remove_small_objects(labeled_alpha, min_size=200)
        cleaned_alpha = (cleaned_alpha > 0).astype(np.uint8)
        alpha = cleaned_alpha * 255
        alpha_gpu = torch.from_numpy(cleaned_alpha).to(device).float().unsqueeze(0)
        x, y, w, h = find_bounding_box(alpha)

    # If alpha is provided, the bounds of all foreground are used
    else: 
        rows, cols = np.where(alpha > 0)
        if rows.size > 0 and cols.size > 0:
            x_min = np.min(cols)
            y_min = np.min(rows)
            x_max = np.max(cols)
            y_max = np.max(rows)

            width = x_max - x_min + 1
            height = y_max - y_min + 1
        x, y, w, h = x_min, y_min, width, height

    if np.all(alpha==0):
        raise ValueError(f"input image too small")
    
    bg_gray = bg_color[0]
    bg_color = torch.from_numpy(bg_color).float().to(device).repeat(alpha_gpu.shape[1], alpha_gpu.shape[2], 1).permute(2, 0, 1)
    rgb_image_gpu = rgb_image_gpu * alpha_gpu + bg_color * (1 - alpha_gpu)
    padding_size = [0] * 6
    if w > h:
        padding_size[0] = int(w * padding_ratio)
        padding_size[2] = int(padding_size[0] + (w - h) / 2)
    else:
        padding_size[2] = int(h * padding_ratio)
        padding_size[0] = int(padding_size[2] + (h - w) / 2)
    padding_size[1] = padding_size[0]
    padding_size[3] = padding_size[2]
    padded_tensor = F.pad(rgb_image_gpu[:, y:(y+h), x:(x+w)], pad=tuple(padding_size), mode='constant', value=bg_gray)

    return padded_tensor

def prepare_image(image_path, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=None, padding_ratio=0.1, device='cuda'):
    if os.path.isfile(image_path):
        img_tensor = load_image(image_path, bg_color=bg_color, rmbg_net=rmbg_net, padding_ratio=padding_ratio, device=device)
        img_np = img_tensor.permute(1,2,0).cpu().numpy()
        img_pil = Image.fromarray((img_np*255).astype(np.uint8))
        
        return img_pil
    else:
        raise ValueError(f"Invalid image path: {image_path}")