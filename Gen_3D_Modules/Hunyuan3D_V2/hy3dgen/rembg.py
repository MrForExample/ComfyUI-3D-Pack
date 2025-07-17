# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from PIL import Image
from rembg import remove, new_session
import numpy as np


class BackgroundRemover():
    def __init__(self, model_name='u2net'):
        self.session = new_session(model_name)

    def __call__(self, image: Image.Image):
        if image.mode in ('RGBA', 'LA') or image.mode != 'RGB':
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode in ('RGBA', 'LA'):
                background.paste(image, mask=image.split()[-1])
            else:
                background.paste(image)
            image = background
        
        output = remove(image, session=self.session)
        
        if output.mode != 'RGBA':
            output = output.convert('RGBA')
            
        return output
