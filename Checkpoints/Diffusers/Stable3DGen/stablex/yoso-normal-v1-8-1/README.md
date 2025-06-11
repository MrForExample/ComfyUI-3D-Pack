---
library_name: diffusers
pipeline_tag: image-to-image
license: apache-2.0
---
# Model Card for StableNormal

This repository contains the weights of StableNormal: Reducing Diffusion Variance for Stable and Sharp Normal

## Usage

See the Github repository: https://github.com/Stable-X/StableNormal regarding installation instructions.

The model can then be used as follows:

```python
import torch
from PIL import Image
# Load an image
input_image = Image.open("path/to/your/image.jpg")
# Create predictor instance
predictor = torch.hub.load("hugoycj/StableNormal", "StableNormal_turbo", trust_repo=True, yoso_version='yoso-normal-v1-8-1')
# Generate normal map using alpha channel for masking
normal_map = predictor(rgba_image, data_type="object")  # Will mask out background, if alpha channel is avalible, else use birefnet
# Apply the model to the image
normal_image = predictor(input_image)
# Save or display the result
normal_image.save("output/normal_map.png")
```