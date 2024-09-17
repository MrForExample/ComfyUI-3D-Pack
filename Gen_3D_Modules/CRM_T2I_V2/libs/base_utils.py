import numpy as np
import cv2
import torch
import numpy as np
from PIL import Image

        
def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    import importlib
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

        
def tensor_detail(t):
    assert type(t) == torch.Tensor
    print(f"shape: {t.shape} mean: {t.mean():.2f}, std: {t.std():.2f}, min: {t.min():.2f}, max: {t.max():.2f}")



def drawRoundRec(draw, color, x, y, w, h, r):
    drawObject = draw

    '''Rounds'''
    drawObject.ellipse((x, y, x + r, y + r), fill=color)
    drawObject.ellipse((x + w - r, y, x + w, y + r), fill=color)
    drawObject.ellipse((x, y + h - r, x + r, y + h), fill=color)
    drawObject.ellipse((x + w - r, y + h - r, x + w, y + h), fill=color)

    '''rec.s'''
    drawObject.rectangle((x + r / 2, y, x + w - (r / 2), y + h), fill=color)
    drawObject.rectangle((x, y + r / 2, x + w, y + h - (r / 2)), fill=color)


def do_resize_content(original_image: Image, scale_rate):
    # resize image content wile retain the original image size
    if scale_rate != 1:
        # Calculate the new size after rescaling
        new_size = tuple(int(dim * scale_rate) for dim in original_image.size)
        # Resize the image while maintaining the aspect ratio
        resized_image = original_image.resize(new_size)
        # Create a new image with the original size and black background
        padded_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
        paste_position = ((original_image.width - resized_image.width) // 2, (original_image.height - resized_image.height) // 2)
        padded_image.paste(resized_image, paste_position)
        return padded_image
    else:
        return original_image

def add_stroke(img, color=(255, 255, 255), stroke_radius=3):
    # color in R, G, B format
    if isinstance(img, Image.Image):
        assert img.mode == "RGBA"
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
    else:
        assert img.shape[2] == 4
    gray = img[:,:, 3]
    ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
    res = cv2.drawContours(img, contours,-1, tuple(color)[::-1] + (255,), stroke_radius)
    return Image.fromarray(cv2.cvtColor(res,cv2.COLOR_BGRA2RGBA))

def make_blob(image_size=(512, 512), sigma=0.2):
    """
    make 2D blob image with:
    I(x, y)=1-\exp \left(-\frac{(x-H / 2)^2+(y-W / 2)^2}{2 \sigma^2 HS}\right)
    """
    import numpy as np
    H, W = image_size
    x = np.arange(0, W, 1, float)
    y = np.arange(0, H, 1, float)
    x, y = np.meshgrid(x, y)
    x0 = W // 2
    y0 = H // 2
    img = 1 - np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2 * H * W))
    return (img * 255).astype(np.uint8)