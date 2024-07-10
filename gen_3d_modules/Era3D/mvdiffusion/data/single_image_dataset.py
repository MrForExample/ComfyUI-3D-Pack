import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Literal, Tuple, Optional, Any
import cv2

import json
import os

import PIL.Image

import cv2
import numpy as np

def add_margin(pil_img, color=0, size=256):
    width, height = pil_img.size
    result = Image.new(pil_img.mode, (size, size), color)
    result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
    return result

def scale_and_place_object(image, scale_factor):
    assert np.shape(image)[-1]==4  # RGBA

    # Extract the alpha channel (transparency) and the object (RGB channels)
    alpha_channel = image[:, :, 3]

    # Find the bounding box coordinates of the object
    coords = cv2.findNonZero(alpha_channel)
    x, y, width, height = cv2.boundingRect(coords)

    # Calculate the scale factor for resizing
    original_height, original_width = image.shape[:2]

    if width > height:
        size = width
        original_size = original_width
    else:
        size = height
        original_size = original_height

    scale_factor = min(scale_factor, size / (original_size+0.0))

    new_size = scale_factor * original_size
    scale_factor = new_size / size

    # Calculate the new size based on the scale factor
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    center_x = original_width // 2
    center_y = original_height // 2

    paste_x = center_x - (new_width // 2)
    paste_y = center_y - (new_height // 2)

    # Resize the object (RGB channels) to the new size
    rescaled_object = cv2.resize(image[y:y+height, x:x+width], (new_width, new_height))

    # Create a new RGBA image with the resized image
    new_image = np.zeros((original_height, original_width, 4), dtype=np.uint8)

    new_image[paste_y:paste_y + new_height, paste_x:paste_x + new_width] = rescaled_object

    return new_image

class SingleImageDataset(Dataset):
    def __init__(self,
        num_views: int,
        img_wh: Tuple[int, int],
        bg_color: str,
        crop_size: int = 224,
        single_image: Optional[PIL.Image.Image] = None,
        images_dir: Optional[str] = None,
        num_validation_samples: Optional[int] = None,
        filepaths: Optional[list] = None,
        cond_type: Optional[str] = None,
        prompt_embeds_path: Optional[str] = None,
        gt_path: Optional[str] = None,
        dtype: Optional[torch.dtype] = torch.float16,
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.images_dir = images_dir
        self.num_views = num_views
        self.img_wh = img_wh
        self.crop_size = crop_size
        self.bg_color = bg_color
        self.cond_type = cond_type
        self.gt_path = gt_path
        self.dtype = dtype

        
        if single_image is None:
            if filepaths is None:
                # Get a list of all files in the directory
                file_list = os.listdir(self.images_dir)
            else:
                file_list = filepaths

            # Filter the files that end with .png or .jpg
            self.file_list = [file for file in file_list if file.endswith(('.png', '.jpg', '.webp'))]
        else:
            self.file_list = None

        # load all images
        self.all_images = []
        self.all_alphas = []
        bg_color = self.get_bg_color()

        if single_image is not None:
            image, alpha = self.load_image(None, bg_color, return_type='pt', Imagefile=single_image)
            self.all_images.append(image)
            self.all_alphas.append(alpha)
        else:
            for file in self.file_list:
                print(os.path.join(self.images_dir, file))
                image, alpha = self.load_image(os.path.join(self.images_dir, file), bg_color, return_type='pt')
                self.all_images.append(image)
                self.all_alphas.append(alpha)
                
            
        self.all_images = self.all_images[:num_validation_samples]
        self.all_alphas = self.all_alphas[:num_validation_samples]
        
        try:
            self.normal_text_embeds = torch.load(f'{prompt_embeds_path}/normal_embeds.pt')
            self.color_text_embeds = torch.load(f'{prompt_embeds_path}/clr_embeds.pt') # 4view
        except:
            self.color_text_embeds = torch.load(f'{prompt_embeds_path}/embeds.pt')
            self.normal_text_embeds = None

    def __len__(self):
        return len(self.all_images)

    def get_bg_color(self):
        if self.bg_color == 'white':
            bg_color = np.array([1., 1., 1.], dtype=np.float32)
        elif self.bg_color == 'black':
            bg_color = np.array([0., 0., 0.], dtype=np.float32)
        elif self.bg_color == 'gray':
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif self.bg_color == 'random':
            bg_color = np.random.rand(3)
        elif isinstance(self.bg_color, float):
            bg_color = np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color
    
    
    def load_image(self, img_path, bg_color, return_type='np', Imagefile=None):
        # pil always returns uint8
        if Imagefile is None:
            image_input = Image.open(img_path)
        else:
            image_input = Imagefile
        image_size = self.img_wh[0]

        if self.crop_size!=-1:
            alpha_np = np.asarray(image_input)[:, :, 3]
            coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
            min_x, min_y = np.min(coords, 0)
            max_x, max_y = np.max(coords, 0)
            ref_img_ = image_input.crop((min_x, min_y, max_x, max_y))
            h, w = ref_img_.height, ref_img_.width
            scale = self.crop_size / max(h, w)
            h_, w_ = int(scale * h), int(scale * w)
            ref_img_ = ref_img_.resize((w_, h_))
            image_input = add_margin(ref_img_, size=image_size)
        else:
            image_input = add_margin(image_input, size=max(image_input.height, image_input.width))
            image_input = image_input.resize((image_size, image_size))

        # img = scale_and_place_object(img, self.scale_ratio)
        img = np.array(image_input)
        img = img.astype(np.float32) / 255. # [0, 1]
        assert img.shape[-1] == 4 # RGBA

        alpha = img[...,3:4]
        img = img[...,:3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
            alpha = torch.from_numpy(alpha)
        else:
            raise NotImplementedError
        
        return img, alpha
    

    def __getitem__(self, index):
        image = self.all_images[index%len(self.all_images)]
        alpha = self.all_alphas[index%len(self.all_images)]
        if self.file_list is not None:
            filename = self.file_list[index%len(self.all_images)].replace(".png", "")
        else:
            filename = 'null'
        img_tensors_in = [
            image.permute(2, 0, 1)
        ] * self.num_views

        alpha_tensors_in = [
            alpha.permute(2, 0, 1)
        ] * self.num_views

        img_tensors_in = torch.stack(img_tensors_in, dim=0).to(self.dtype) # (Nv, 3, H, W)
        alpha_tensors_in = torch.stack(alpha_tensors_in, dim=0).to(self.dtype) # (Nv, 3, H, W)
        
        if self.gt_path is not None:
            gt_image = self.gt_images[index%len(self.all_images)]
            gt_alpha = self.gt_alpha[index%len(self.all_images)]
            gt_img_tensors_in = [gt_image.permute(2, 0, 1) ] * self.num_views
            gt_alpha_tensors_in = [gt_alpha.permute(2, 0, 1) ] * self.num_views
            gt_img_tensors_in = torch.stack(gt_img_tensors_in, dim=0).to(self.dtype)
            gt_alpha_tensors_in = torch.stack(gt_alpha_tensors_in, dim=0).to(self.dtype)
                
        normal_prompt_embeddings = self.normal_text_embeds if hasattr(self, 'normal_text_embeds') else None
        color_prompt_embeddings = self.color_text_embeds if hasattr(self, 'color_text_embeds') else None
        
        out =  {
            'imgs_in': img_tensors_in,
            'alphas': alpha_tensors_in,
            'normal_prompt_embeddings': normal_prompt_embeddings,
            'color_prompt_embeddings': color_prompt_embeddings,
            'filename': filename,
            }
            
        return out

        

