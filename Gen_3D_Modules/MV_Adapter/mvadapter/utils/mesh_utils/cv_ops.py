from typing import Optional


import torch
import torch.nn.functional as F

try:
    import cvcuda
    torch_to_cvc = lambda x, layout: cvcuda.as_tensor(x, layout)

    cvc_to_torch = lambda x, device: torch.tensor(x.cuda(), device=device)
except ImportError:
    cvcuda = None
    torch_to_cvc = None
    cvc_to_torch = None


def inpaint_cvc(
    image: torch.Tensor,
    mask: torch.Tensor,
    padding_size: int,
    return_dtype: Optional[torch.dtype] = None,
):
    input_dtype = image.dtype
    input_device = image.device

    image = image.detach()
    mask = mask.detach()

    if image.dtype != torch.uint8:
        image = (image * 255).to(torch.uint8)
    if mask.dtype != torch.uint8:
        mask = (mask * 255).to(torch.uint8)

    image_cvc = torch_to_cvc(image, "HWC")
    mask_cvc = torch_to_cvc(mask, "HW")
    output_cvc = cvcuda.inpaint(image_cvc, mask_cvc, padding_size)
    output = cvc_to_torch(output_cvc, device=input_device)

    if return_dtype == torch.uint8 or input_dtype == torch.uint8:
        return output
    return output.to(dtype=input_dtype) / 255.0


def inpaint_torch(
    image: torch.Tensor,
    mask: torch.Tensor,
    padding_size: int,
    return_dtype: Optional[torch.dtype] = None,
):
    """
    Inpaint masked areas in an image using PyTorch operations.

    Args:
        image: Input image tensor [H, W, C]
        mask: Binary mask tensor [H, W] where non-zero values indicate pixels to inpaint
        padding_size: Size of neighborhood to consider for inpainting
        return_dtype: Optional dtype for the output tensor

    Returns:
        Inpainted image tensor
    """
    input_dtype = image.dtype
    input_device = image.device

    image = image.detach()
    mask = mask.detach()

    if image.dtype != torch.uint8:
        image = (image * 255).to(torch.uint8)
    if mask.dtype != torch.uint8:
        mask = (mask * 255).to(torch.uint8)

    # Convert to float for processing
    image_float = image.float() / 255.0
    mask_float = (mask > 0).float()  # 1 for areas to inpaint, 0 for known areas

    # Initialize output with original image
    output = image_float.clone()

    # Create distance-based weights for neighborhood pixels
    kernel_size = min(2 * padding_size + 1, 31)  # Limit kernel size to avoid memory issues
    y, x = torch.meshgrid(
        torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=input_device),
        torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=input_device),
        indexing="ij"
    )
    dist = torch.sqrt(x.float() ** 2 + y.float() ** 2)
    weights = 1.0 / (dist + 1e-6)
    weights = weights / weights.sum()
    weights_kernel = weights.unsqueeze(0).unsqueeze(0)

    # For each channel, compute weighted average of non-masked pixels
    inpainted = []

    for c in range(image_float.shape[-1]):
        channel = image_float[..., c]

        # Compute weighted sum of valid pixels
        valid_values = channel * (1 - mask_float)
        weighted_sum = F.conv2d(
            valid_values.unsqueeze(0).unsqueeze(0),
            weights_kernel,
            padding=kernel_size // 2
        ).squeeze(0).squeeze(0)

        # Compute sum of weights for valid pixels
        valid_weights = F.conv2d(
            (1 - mask_float).unsqueeze(0).unsqueeze(0),
            weights_kernel,
            padding=kernel_size // 2
        ).squeeze(0).squeeze(0)

        # Avoid division by zero
        valid_weights = torch.clamp(valid_weights, min=1e-6)

        # Compute weighted average
        inpainted_channel = weighted_sum / valid_weights

        # Only update masked pixels
        inpainted_channel = torch.where(mask_float > 0, inpainted_channel, channel)
        inpainted.append(inpainted_channel)

    # Combine channels
    output = torch.stack(inpainted, dim=-1)

    # Convert back to appropriate dtype
    if return_dtype == torch.uint8 or input_dtype == torch.uint8:
        output = (output * 255).to(torch.uint8)
    else:
        output = output.to(dtype=input_dtype)

    return output


def batch_inpaint_cvc(
    images: torch.Tensor,
    masks: torch.Tensor,
    padding_size: int,
    return_dtype: Optional[torch.dtype] = None,
):
    output = torch.stack(
        [
            inpaint_cvc(image, mask, padding_size, return_dtype)
            for (image, mask) in zip(images, masks)
        ],
        axis=0,
    )
    return output


def batch_erode(
    masks: torch.Tensor, kernel_size: int, return_dtype: Optional[torch.dtype] = None
):
    input_dtype = masks.dtype
    input_device = masks.device
    masks = masks.detach()
    if masks.dtype != torch.uint8:
        masks = (masks.float() * 255).to(torch.uint8)
    masks_cvc = torch_to_cvc(masks[..., None], "NHWC")
    masks_erode_cvc = cvcuda.morphology(
        masks_cvc,
        cvcuda.MorphologyType.ERODE,
        maskSize=(kernel_size, kernel_size),
        anchor=(-1, -1),
    )
    masks_erode = cvc_to_torch(masks_erode_cvc, device=input_device)[..., 0]
    if return_dtype == torch.uint8 or input_dtype == torch.uint8:
        return masks_erode
    return (masks_erode > 0).to(dtype=input_dtype)


def batch_dilate(
    masks: torch.Tensor, kernel_size: int, return_dtype: Optional[torch.dtype] = None
):
    input_dtype = masks.dtype
    input_device = masks.device
    masks = masks.detach()
    if masks.dtype != torch.uint8:
        masks = (masks.float() * 255).to(torch.uint8)
    masks_cvc = torch_to_cvc(masks[..., None], "NHWC")
    masks_dilate_cvc = cvcuda.morphology(
        masks_cvc,
        cvcuda.MorphologyType.DILATE,
        maskSize=(kernel_size, kernel_size),
        anchor=(-1, -1),
    )
    masks_dilate = cvc_to_torch(masks_dilate_cvc, device=input_device)[..., 0]
    if return_dtype == torch.uint8 or input_dtype == torch.uint8:
        return masks_dilate
    return (masks_dilate > 0).to(dtype=input_dtype)
