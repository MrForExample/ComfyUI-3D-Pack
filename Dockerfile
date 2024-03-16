ARG CUDA_VERSION=12.1.0-devel

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-ubuntu22.04

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        build-essential \
        curl \
        ffmpeg \
        git \
        libgl1 libglib2.0-0 \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
        libegl1-mesa-dev \
        ninja-build \
        python3.11 \
        python3.11-dev \
        wget \
        && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.11 /usr/bin/python & \
    ln -s /usr/bin/pip3.11 /usr/bin/pip

RUN adduser --uid 1001 -q user && \
    mkdir /app && chown user /app
USER user

RUN pip install --no-cache --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.2.0 torchvision==0.17.0 xformers==0.0.24

WORKDIR /app
RUN git clone "https://github.com/comfyanonymous/ComfyUI.git" ./ && \
    git reset --hard 05cd00695a84cebd5603a31f665eb7301fba2beb
RUN pip install --no-cache -r requirements.txt

WORKDIR /app/custom_nodes/ComfyUI-3D-Pack/
COPY --chown=user:user ./ ./

RUN pip install --no-cache -r requirements.txt \
    # post requirements installation require gpu, setup
    # `nvidia-container-runtime`, for docker, see
    # https://stackoverflow.com/a/61737404  
    -r requirements_post.txt \
    # those seem to be missed
    ninja scikit-learn rembg[gpu] open_clip_torch

WORKDIR /app
ENTRYPOINT [ "python", "main.py", "--listen", "0.0.0.0" ]
