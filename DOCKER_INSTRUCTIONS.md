## Docker Instructions

This document contains instructions on how to build and run the application using Docker.

### Disclaimer

These instructions are based on the assumption that you have Docker installed on your machine. If you don't have Docker installed, you can download it from the [official website](https://www.docker.com/get-started).

All tests were performed using Docker version 26.1.3 and Docker Compose version 2.27.0 under Ubuntu 22.04.

These instructions are focused on running the Unique3D workflows, you can use the same steps to run the other workflows, but you need to check the model requirements for each workflow at the [README.md](./README.md) file.

### Requirements

You need the `nvidia-container-runtime` installed and configured to run the application. You can check the installation instructions at the [official documentation](https://developer.nvidia.com/container-runtime).

#### Models

To use Unique3D you will need to download the following models and place them at the models folder in the root directory:

```bash
wget -c https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors -P models/
wget -c https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors -P models/
wget -c https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P models/
wget -c https://huggingface.co/spaces/Wuvin/Unique3D/resolve/main/ckpt/controlnet-tile/diffusion_pytorch_model.safetensors -P models/
wget -c https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt -P models/
```

You can check more details at the Unique3D section of the [README.md](./README.md) file.

### Build Docker Image

You can build the docker image using the following command:

```bash
DOCKER_BUILDKIT=0 docker build -t comfy3d --file Dockerfile .
```

You can also build the image using the docker-compose file:

```bash
DOCKER_BUILDKIT=0 docker compose build
```

> OBS: The `DOCKER_BUILDKIT=0` is used to disable the buildkit feature, which is not supported by the docker-compose file.

### Run Docker Container

You can run the docker container using the following command:

```bash
docker run -d \
  --name comfy3d \
  --platform linux/amd64 \
  --runtime=nvidia \
  -p 8188:8188 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e COMFYUI_PATH=/app \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/models/diffusion_pytorch_model.safetensors:/app/models/controlnet/control_unique3d_sd15_tile.safetensors \
  -v $(pwd)/models/ip-adapter_sd15.safetensors:/app/models/ipadapter/ip-adapter_sd15.safetensors \
  -v $(pwd)/models/RealESRGAN_x4plus.pth:/app/models/upscale_models/RealESRGAN_x4plus.pth \
  -v $(pwd)/models/v1-5-pruned-emaonly.ckpt:/app/models/checkpoints/v1-5-pruned-emaonly.ckpt \
  -v $(pwd)/models/model.safetensors:/app/models/clip_vision/OpenCLIP-ViT-H-14.safetensors \
  comfy3d
```

You can also run the container using the docker-compose file:

```bash
docker compose up
```

### Access the Application

After running the container, you can access the application by opening the following URL in your browser:

```
http://localhost:8188
```

To test, you can load the following workflow:

- `./example_workflows/Unique3D/Unique3D_All_Stages.json`

