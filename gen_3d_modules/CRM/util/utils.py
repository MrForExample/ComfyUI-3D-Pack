import numpy as np
import torch
import random


# Reworked so this matches gluPerspective / glm::perspective, using fovy
def perspective(fovx=0.7854, aspect=1.0, n=0.1, f=1000.0, device=None):
    # y = np.tan(fovy / 2)
    x = np.tan(fovx / 2)
    return torch.tensor([[1/x,         0,            0,              0],
                         [  0, -aspect/x,            0,              0],
                         [  0,         0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                         [  0,         0,           -1,              0]], dtype=torch.float32, device=device)


def translate(x, y, z, device=None):
    return torch.tensor([[1, 0, 0, x],
                         [0, 1, 0, y],
                         [0, 0, 1, z],
                         [0, 0, 0, 1]], dtype=torch.float32, device=device)


def rotate_x(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[1, 0,  0, 0],
                         [0, c, -s, 0],
                         [0, s,  c, 0],
                         [0, 0,  0, 1]], dtype=torch.float32, device=device)


def rotate_y(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[ c, 0, s, 0],
                         [ 0, 1, 0, 0],
                         [-s, 0, c, 0],
                         [ 0, 0, 0, 1]], dtype=torch.float32, device=device)


def rotate_z(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[c, -s, 0, 0],
                         [s,  c, 0, 0],
                         [0,  0, 1, 0],
                         [0,  0, 0, 1]], dtype=torch.float32, device=device)

@torch.no_grad()
def batch_random_rotation_translation(b, t, device=None):
    m = np.random.normal(size=[b, 3, 3])
    m[:, 1] = np.cross(m[:, 0], m[:, 2])
    m[:, 2] = np.cross(m[:, 0], m[:, 1])
    m = m / np.linalg.norm(m, axis=2, keepdims=True)
    m = np.pad(m, [[0, 0], [0, 1], [0, 1]], mode='constant')
    m[:, 3, 3] = 1.0
    m[:, :3, 3] = np.random.uniform(-t, t, size=[b, 3])
    return torch.tensor(m, dtype=torch.float32, device=device)

@torch.no_grad()
def random_rotation_translation(t, device=None):
    m = np.random.normal(size=[3, 3])
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.random.uniform(-t, t, size=[3])
    return torch.tensor(m, dtype=torch.float32, device=device)


@torch.no_grad()
def random_rotation(device=None):
    m = np.random.normal(size=[3, 3])
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.array([0,0,0]).astype(np.float32)
    return torch.tensor(m, dtype=torch.float32, device=device)


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)


def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN


def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)


def lr_schedule(iter, warmup_iter, scheduler_decay):
    if iter < warmup_iter:
        return iter / warmup_iter
    return max(0.0, 10 ** (
            -(iter - warmup_iter) * scheduler_decay)) 


def trans_depth(depth):
    depth = depth[0].detach().cpu().numpy()
    valid = depth > 0
    depth[valid] -= depth[valid].min()
    depth[valid] = ((depth[valid] / depth[valid].max()) * 255)
    return depth.astype('uint8')


def nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None):
    assert isinstance(input, torch.Tensor)
    if posinf is None:
        posinf = torch.finfo(input.dtype).max
    if neginf is None:
        neginf = torch.finfo(input.dtype).min
    assert nan == 0
    return torch.clamp(input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out)


def load_item(filepath):
    with open(filepath, 'r') as f:
        items = [name.strip() for name in f.readlines()]
    return set(items)

def load_prompt(filepath):
    uuid2prompt = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            list_line = line.split(',')
            uuid2prompt[list_line[0]] = ','.join(list_line[1:]).strip()
    return uuid2prompt

def resize_and_center_image(image_tensor, scale=0.95, c = 0, shift = 0, rgb=False, aug_shift = 0):
    if scale == 1:
        return image_tensor
    B, C, H, W = image_tensor.shape
    new_H, new_W = int(H * scale), int(W * scale)
    resized_image = torch.nn.functional.interpolate(image_tensor, size=(new_H, new_W), mode='bilinear', align_corners=False).squeeze(0)
    background = torch.zeros_like(image_tensor) + c
    start_y, start_x = (H - new_H) // 2, (W - new_W) // 2
    if shift == 0:
        background[:, :, start_y:start_y + new_H, start_x:start_x + new_W] = resized_image
    else:
        for i in range(B):
            randx = random.randint(-shift, shift)
            randy = random.randint(-shift, shift)   
            if rgb == True:
                if i == 0 or i==2 or i==4:
                    randx = 0
                    randy = 0 
            background[i, :, start_y+randy:start_y + new_H+randy, start_x+randx:start_x + new_W+randx] = resized_image[i]
    if aug_shift == 0:
        return background  
    for i in range(B):
        for j in range(C):
            background[i, j, :, :] += (random.random() - 0.5)*2 * aug_shift / 255
    return background 
                               
def get_tri(triview_color, dim = 1, blender=True, c = 0, scale=0.95, shift = 0, fix = False, rgb=False, aug_shift = 0):
    # triview_color: [6,C,H,W]
    # rgb is useful when shift is not 0
    triview_color = resize_and_center_image(triview_color, scale=scale, c = c, shift=shift,rgb=rgb, aug_shift = aug_shift)
    if blender is False:
        triview_color0 = torch.rot90(triview_color[0],k=2,dims=[1,2])
        triview_color1 = torch.rot90(triview_color[4],k=1,dims=[1,2]).flip(2).flip(1)
        triview_color2 = torch.rot90(triview_color[5],k=1,dims=[1,2]).flip(2)
        triview_color3 = torch.rot90(triview_color[3],k=2,dims=[1,2]).flip(2)
        triview_color4 = torch.rot90(triview_color[1],k=3,dims=[1,2]).flip(1)
        triview_color5 = torch.rot90(triview_color[2],k=3,dims=[1,2]).flip(1).flip(2)
    else:
        triview_color0 = torch.rot90(triview_color[2],k=2,dims=[1,2])
        triview_color1 = torch.rot90(triview_color[4],k=0,dims=[1,2]).flip(2).flip(1)
        triview_color2 = torch.rot90(torch.rot90(triview_color[0],k=3,dims=[1,2]).flip(2), k=2,dims=[1,2])
        triview_color3 = torch.rot90(torch.rot90(triview_color[5],k=2,dims=[1,2]).flip(2), k=2,dims=[1,2])
        triview_color4 = torch.rot90(triview_color[1],k=2,dims=[1,2]).flip(1).flip(1).flip(2)
        triview_color5 = torch.rot90(triview_color[3],k=1,dims=[1,2]).flip(1).flip(2)
        if fix == True:
            triview_color0[1] = triview_color0[1] * 0
            triview_color0[2] = triview_color0[2] * 0
            triview_color3[1] = triview_color3[1] * 0
            triview_color3[2] = triview_color3[2] * 0

            triview_color1[0] = triview_color1[0] * 0
            triview_color1[1] = triview_color1[1] * 0
            triview_color4[0] = triview_color4[0] * 0
            triview_color4[1] = triview_color4[1] * 0

            triview_color2[0] = triview_color2[0] * 0
            triview_color2[2] = triview_color2[2] * 0
            triview_color5[0] = triview_color5[0] * 0
            triview_color5[2] = triview_color5[2] * 0
    color_tensor1_gt = torch.cat((triview_color0, triview_color1, triview_color2), dim=2)
    color_tensor2_gt = torch.cat((triview_color3, triview_color4, triview_color5), dim=2)
    color_tensor_gt = torch.cat((color_tensor1_gt, color_tensor2_gt), dim = dim)
    return color_tensor_gt

