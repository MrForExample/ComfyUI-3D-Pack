# ComfyUI-3D-Pack
**Make ComfyUI generates 3D assets as good & convenient as it generates image/video!**
<br>
This is an extensive node suite that enables ComfyUI to process 3D inputs (Mesh & UV Texture, etc.) using cutting edge algorithms (3DGS, NeRF, etc.) and models (InstantMesh, CRM, TripoSR, etc.)

<span style="font-size:1.5em;">
<a href=#currently-support>Features</a> &mdash;
<a href=#roadmap>Roadmap</a> &mdash;
<a href=#install>Install</a> &mdash;
<a href=#run>Run</a> &mdash;
<a href=#tips>Tips</a> &mdash;
<a href=#supporters>Supporters</a>
</span>

## Currently support:
- For use case please check [Example Workflows](./_Example_Workflows/). [**Last update: 07/06/2024**]
  - **Note:** you need to put [Example Inputs Files & Folders](_Example_Workflows/_Example_Inputs_Files/) under ComfyUI Root Directory\ComfyUI\input folder before you can run the example workflow
  - [tripoSR-layered-diffusion workflow](https://github.com/C0nsumption/Consume-ComfyUI-Workflows/tree/main/assets/tripo_sr/00) by [@Consumption](https://twitter.com/c0nsumption_)

- **Unique3D**: [AiuniAI/Unique3D](https://github.com/AiuniAI/Unique3D)
  - Four stages pipeline: 
    1. Single image to 4 multi-view images with resulution: 256X256
    2. Consistent Multi-view images Upscale to 512X512, super resolution to 2048X2048
    3. Multi-view images to Normal maps with resulution: 512X512, super resolution to 2048X2048
    4. Multi-view images & Normal maps to 3D mesh with texture
  - To use the [pure Unique3D workflow](./_Example_Workflows/Unique3D/Unique3D_All_Stages.json), Download Models:
    - [img2mvimg](https://huggingface.co/spaces/Wuvin/Unique3D/tree/main/ckpt/img2mvimg) and put it into [./checkpoints/Wuvin/Unique3D/image2mvimage](./checkpoints/Wuvin/Unique3D/image2mvimage)
    - [image2normal](https://huggingface.co/spaces/Wuvin/Unique3D/tree/main/ckpt/image2normal) and put it into [./checkpoints/Wuvin/Unique3D/image2normal](./checkpoints/Wuvin/Unique3D/image2normal)
    - [fine-tuned controlnet-tile](https://huggingface.co/spaces/Wuvin/Unique3D/tree/main/ckpt/controlnet-tile) and put it into `Your ComfyUI root directory/ComfyUI/models/controlnet`
    - [ip-adapter_sd15](https://huggingface.co/h94/IP-Adapter/blob/main/models/ip-adapter_sd15.safetensors) and put it into `Your ComfyUI root directory/ComfyUI/models/ipadapter`
    - [RealESRGAN_x4plus](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) and put it into `Your ComfyUI root directory/ComfyUI/models/upscale_models`

  <video controls autoplay loop src="https://github.com/MrForExample/ComfyUI-3D-Pack/assets/62230687/45dd6bfc-4f2b-4b1f-baed-13a1b0722896"></video>

- **Era3D Diffusion Model**: [pengHTYX/Era3D](https://github.com/pengHTYX/Era3D)
  - Single image to 6 multi-view images & normal maps with resulution: 512X512
  - *Note: you need at least 16GB vram to run this model*

  <video controls autoplay loop src="https://github.com/MrForExample/ComfyUI-3D-Pack/assets/62230687/fc210cac-6c7d-4a55-926c-adb5fb7b0c57"></video>

- **InstantMesh Reconstruction Model**: [TencentARC/InstantMesh](https://github.com/TencentARC/InstantMesh)
  - Sparse multi-view images with white background to 3D Mesh with RGB texture
  - Works with arbitrary MVDiffusion models (Probably works best with Zero123++, but also works with CRM MVDiffusion model)

  <video controls autoplay loop src="https://github.com/MrForExample/ComfyUI-3D-Pack/assets/62230687/a0648a44-f8cb-4f78-9704-a907f9174936"></video>
  <video controls autoplay loop src="https://github.com/MrForExample/ComfyUI-3D-Pack/assets/62230687/33aecedb-f595-4c12-90dd-89d5f718598e"></video>

- **Zero123++**: [SUDO-AI-3D/zero123plus](https://github.com/SUDO-AI-3D/zero123plus)
  - Single image to 6 view images with resulution: 320X320

- **CRM**: [thu-ml/CRM](https://github.com/thu-ml/CRM)
  - Three stages pipeline: 
    1. Single image to 6 view images (Front, Back, Left, Right, Top & Down)
    2. Single image & 6 view images to 6 same views CCMs (Canonical Coordinate Maps)
    3. 6 view images & CCMs to 3D mesh
  - *Note: For low vram pc, if you can't fit all three models for each stages into your GPU memory, then you can divide those three stages into different comfy workflow and run them separately*

    <video controls autoplay loop src="https://github.com/MrForExample/ComfyUI-3D-Pack/assets/62230687/cf68bb83-9244-44df-9db8-f80eb3fdc29e"></video>

- **TripoSR**: [VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR) | [ComfyUI-Flowty-TripoSR](https://github.com/flowtyone/ComfyUI-Flowty-TripoSR)
  - Generate NeRF representation and using marching cube to turn it into 3D mesh
 
    <video controls autoplay loop src="https://github.com/MrForExample/ComfyUI-3D-Pack/assets/62230687/ec4f8df5-5907-4bbf-ba19-c0565fe95a97"></video>

- **Wonder3D**: [xxlong0/Wonder3D](https://github.com/xxlong0/Wonder3D)
  - Generate spatial consistent 6 views images & normal maps from a single image
  ![Wonder3D_FatCat_MVs](_Example_Workflows/_Example_Outputs/Wonder3D_FatCat_MVs.jpg)

- **Large Multiview Gaussian Model**: [3DTopia/LGM](https://github.com/3DTopia/LGM)
  - Enable single image to 3D Gaussian in less than 30 seconds on a RTX3080 GPU, later you can also convert 3D Gaussian to mesh

    <video controls autoplay loop src="https://github.com/MrForExample/ComfyUI-3D-Pack/assets/62230687/e221d7f8-49ac-4ed4-809b-d4c790b6270e"></video>

- **Triplane Gaussian Transformers**: [VAST-AI-Research/TriplaneGaussian](https://github.com/VAST-AI-Research/TriplaneGaussian)
  - Enable single image to 3D Gaussian in less than 10 seconds on a RTX3080 GPU, later you can also convert 3D Gaussian to mesh
 
    <video controls autoplay loop src="https://github.com/MrForExample/ComfyUI-3D-Pack/assets/62230687/90e7f298-bdbd-4c15-9378-1ca46cbb4871"></video>

- **Preview 3DGS and 3D Mesh**: 3D Visualization inside ComfyUI:
  - Using [gsplat.js](https://github.com/huggingface/gsplat.js/tree/main) and [three.js](https://github.com/mrdoob/three.js/tree/dev) for 3DGS & 3D Mesh visualization respectively
  - Custumizable background base on JS library: [mdbassit/Coloris](https://github.com/mdbassit/Coloris)

    <video controls autoplay loop src="https://github.com/MrForExample/ComfyUI-3D-Pack/assets/62230687/9f3c56b1-afb3-4bf1-8845-ab1025a87463"></video>

- **Stack Orbit Camera Poses**: Automatically generate all range of camera pose combinations
  - You can use it to conditioning the [StableZero123 (You need to Download the checkpoint first)](https://comfyanonymous.github.io/ComfyUI_examples/3d/), with full range of camera poses in one prompt pass
  - You can use it to generate the orbit camera poses and directly input to other 3D process node (e.g. GaussianSplatting and BakeTextureToMesh)
  - Example usage:

    <img src="_Example_Workflows/_Example_Outputs/Cammy_Cam_Rotate_Clockwise_Camposes.png" width="256"/> <img src="_Example_Workflows/_Example_Outputs/Cammy_Cam_Rotate_Counter_Clockwise_Camposes.png" width="256"/>
    <br>
    <img src="_Example_Workflows/_Example_Outputs/Cammy_Cam_Rotate_Clockwise.gif" width="256"/> <img src="_Example_Workflows/_Example_Outputs/Cammy_Cam_Rotate_Counter_Clockwise.gif" width="256"/> 
  - Coordinate system:
    - Azimuth: In top view, from angle 0 rotate 360 degree with step -90 you get (0, -90, -180/180, 90, 0), in this case camera rotates clock-wise, vice versa.
    - Elevation: 0 when camera points horizontally forward, pointing down to the ground is negitive angle, vice versa.

- **FlexiCubes**: [nv-tlabs/FlexiCubes](https://github.com/nv-tlabs/FlexiCubes)
  - Multi-View depth & mask (optional normal maps) as inputs
  - Export to 3D Mesh
  - Usage guide: 
    - *voxel_grids_resolution*: determine mesh resolution/quality
    - *depth_min_distance* *depth_max_distance* : distance from object to camera, object parts in the render that is closer(futher) to camera than depth_min_distance(depth_max_distance) will be rendered with pure white(black) RGB value 1, 1, 1(0, 0, 0)
    - *mask_loss_weight*: Control the silhouette of reconstrocted 3D mesh
    - *depth_loss_weight*: Control the shape of reconstrocted 3D mesh, this loss will also affect the mesh deform detail on the surface, so results depends on quality of the depth map
    - *normal_loss_weight*: Optional. Use to refine the mesh deform detail on the surface
    - *sdf_regularizer_weight*: Helps to remove floaters in areas of the shape that are not supervised by the application objective, such as internal faces when using image supervision only
    - *remove_floaters_weight*: This can be increased if you observe artifacts in flat areas
    - *cube_stabilizer_weight*: This does not have a significant impact during the optimization of a single shape, however it helps to stabilizing training in somecases

    <video controls autoplay loop src="https://github.com/MrForExample/ComfyUI-3D-Pack/assets/62230687/166bbc1f-04b7-42c8-87bb-302e3f5aabb2"></video>


- **Instant NGP**: [nerfacc](https://github.com/nerfstudio-project/nerfacc)
  - Multi-View images as inputs
  - Export to 3D Mesh using marching cubes

- **3D Gaussian Splatting**
  - [Improved Differential Gaussian Rasterization](https://github.com/ashawkey/diff-gaussian-rasterization)
  - Better Compactness-based Densification method from [Gsgen](https://gsgen3d.github.io/), 
  - Support initialize gaussians from given 3D mesh (Optional)
  - Support mini-batch optimazation
  - Multi-View images as inputs
  - Export to standard 3DGS .ply format supported
  
- **Gaussian Splatting Orbit Renderer**
  - Render 3DGS to images sequences or video, given a 3DGS file and camera poses generated by **Stack Orbit Camera Poses** node
  
- **Mesh Orbit Renderer**
  - Render 3D mesh to images sequences or video, given a mesh file and camera poses generated by **Stack Orbit Camera Poses** node

- **Fitting_Mesh_With_Multiview_Images**
  - Bake Multi-View images into UVTexture of given 3D mesh using [Nvdiffrast](https://github.com/NVlabs/nvdiffrast), supports:
  - Export to .obj, .ply, .glb

- **NeuS**
  - Fit a coarse mesh from sparse multi-view images & normal maps, as little as 4 to 6 views, pretty good at reconstruct the shape from reference images but texture lacking details.

- **Deep Marching Tetrahedrons**
  - Allow convert 3DGS .ply file to 3D mesh <br>
  *Note: I didn't spent time to turn the hyperprameters yet, the result will be improved in the future!*

- **Save & Load 3D file**
  - .obj, .ply, .glb for 3D Mesh
  - .ply for 3DGS

- **Switch Axis for 3DGS & 3D Mesh**
  - Since different algorithms likely use different coordinate system, so the ability to re-mapping the axis of coordinate is crucial for passing generated result between differnt nodes.

- **[Customizable system config file](configs/system.conf)**
  - Custom clients IP address 

## Roadmap:
- [x] Add DMTet algorithm to allow conversion from points cloud(Gaussian/.ply) to mesh (.obj, .ply, .glb)

- [x] Integrate [Triplane Meets Gaussian Splatting: Fast and Generalizable Single-View 3D Reconstruction with Transformers](https://zouzx.github.io/TriplaneGaussian/)

- [x] Add interactive 3D UI inside ComfuUI to visulaize training and generated results for 3D representations

- [x] Add a new node to generate renderer image sequence given a 3D gaussians and orbit camera poses (So we can later feed it to the differentiable renderer to bake it onto a given mesh)

- [x] Integrate [LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation](https://me.kiui.moe/lgm/)

- [ ] Add camera pose estimation from raw multi-views images

- [ ] Add & Improve a few best MVS algorithms (e.g instant-ngp, NeuS2, GaussianPro, etc.)

- [ ] Improve 3DGS/Nerf to Mesh conversion algorithms:
  -  Support to training DMTet with images(RGB, Alpha, Normal Map)
  -  Find better methods to converts 3DGS or Points Cloud to Mesh (Normal maps reconstruction maybe?)

- Add a general SDS/ISM Optimization algorithm to allow training 3D representations with diffusion model
  - Need to do some in-depth research on Interval Score Matching (ISM), since math behind it makes perfect sense and also there are so many ways we could improve upon the result obtained from [LucidDreamer](https://github.com/EnVision-Research/LucidDreamer)
  - On Hold since runtime cost to generate an is too big (3+hours for an average RTX GPU like 3080)

## Install:

**[IMPORTANT!!!]** <br> Currently this package is only been tested in following setups:
- Windows 10/11 (Tested on my laptop)
- Ubuntu 23.10 [(Tested by @watsieboi)](https://github.com/MrForExample/ComfyUI-3D-Pack/issues/16)
- ComfyUI python_embed/Miniconda/Conda Python 3.11.x
- Torch version >= 2.1.2+cu121

<br>

Assume you have already downloaded [ComfyUI](https://github.com/comfyanonymous/ComfyUI) & Configed your [CUDA](https://developer.nvidia.com/cuda-12-1-0-download-archive) environment.

### Install Method 0: Directly inside ComfyUI Windows Python Embeded Environment 
***Currently support: (python3.10/3.11/3.12 cuda12.1)***

First install [Visual Studio Build Tools 2022/2019](https://visualstudio.microsoft.com/downloads/?q=build+tools) with Workloads: Desktop development with C++ (There are a few JIT torch cpp extension that builds in runtime)
- Alternatively, according to [@doctorpangloss](https://github.com/MrForExample/ComfyUI-3D-Pack/issues/5), you can setup the c++/cuda build environments in windows by using [chocolatey](https://chocolatey.org/)

Go to the Comfy3D root directory: *ComfyUI Root Directory\ComfyUI\custom_nodes\ComfyUI-3D-Pack* and run:

```bash
# Run .bat with python version corresponding to the version of your ComfyUI python environment

# install_windows_portable_win_py310_cu121.bat
install_windows_portable_win_py311_cu121.bat
# install_windows_portable_win_py312_cu121.bat
```

### Install Method 1: Using Miniconda(Works on Windows & Linux & Mac)
***Note: [In some edge cases Miniconda fails but Anaconda could fix the issue](https://github.com/MrForExample/ComfyUI-3D-Pack/issues/49)***

#### Setup with Miniconda:
First download [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) (*One of the best way to manage a clean and separated python envirments*)

Then running following commands to setup the Miniconda environment for ComfyUI:

```bash
# Go to your Your ComfyUI root directory, for my example:
cd C:\Users\reall\Softwares\ComfyUI_windows_portable 

conda create -p ./python_miniconda_env/ComfyUI python=3.11

# conda will tell what command to use to activate the env
conda activate C:\Users\reall\Softwares\ComfyUI_windows_portable\python_miniconda_env\ComfyUI

# update pip
python -m pip install --upgrade pip

# You can using following command to installing CUDA only in the miniconda environment you just created if you don't want to donwload and install it manually & globally:
# conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit

# Install the main packahes
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r ./ComfyUI/requirements.txt

# Then go to ComfyUI-3D-Pack directory under the ComfyUI Root Directory\ComfyUI\custom_nodes for my example is:
cd C:\Users\reall\Softwares\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-3D-Pack
```

   - Alternatively you can check this tutorial: [Installing ComfyUI with Miniconda On Windows and Mac](https://www.comflowy.com/preparation-for-study/install#step-two-download-comfyui)
  
#### Install with Miniconda:
Go to the Comfy3D root directory: *ComfyUI Root Directory\ComfyUI\custom_nodes\ComfyUI-3D-Pack* and run:

```bash
install_miniconda.bat
```
Just in case `install_miniconda.bat` may not working in your OS, you could also run the following commands under the same directory: (Works with Linux & macOS)

```bash
pip install -r requirements.txt

pip install -r requirements_post.txt
```
<br>

**Plus:**<br>
- For those who want to run it inside Google Colab, you can check the [install instruction from @lovisdotio](https://github.com/MrForExample/ComfyUI-3D-Pack/issues/13)
- You can find some of the pre-build wheels for Linux here: [remsky/ComfyUI3D-Assorted-Wheels](https://github.com/remsky/ComfyUI3D-Assorted-Wheels)

#### Install and run with docker:

Gpu support during Docker build time is required to install all requirenents. 
On Linux host you could setup `nvidia-container-runtime`. On Windows
it is quite different and not checked at moment.

##### Linux setup:

1. Install nvidia-container-runtime:
    ```bash
    sudo apt-get install nvidia-container-runtime
    ```

1. Edit/create the /etc/docker/daemon.json with content:
    ```json
    {
        "runtimes": {
            "nvidia": {
                "path": "/usr/bin/nvidia-container-runtime",
                "runtimeArgs": []
            } 
        },
        "default-runtime": "nvidia" 
    }
    ```
    
1. Restart docker daemon:
    ```bash
    sudo systemctl restart docker
    ```

Finally build and run docker container with:
```bash
docker build -t comfy3d . && docker run --rm -it -p 8188:8188 --gpus all comfy3d
```

## Run:
Copy the files inside folder [__New_ComfyUI_Bats](./_New_ComfyUI_Bats/) to your ComfyUI root directory, and double click run_nvidia_gpu_miniconda.bat to start ComfyUI!
- Alternatively you can just activate the Conda env: `python_miniconda_env\ComfyUI`, and go to your ComfyUI root directory then run command `python ./ComfyUI/main.py`

## Tips
* OpenGL world & camera coordinate system:
```
    World            Camera        
  
     +y              up  target                                              
     |               |  /                                            
     |               | /                                                
     |______+x       |/______right                                      
    /                /         
   /                /          
  /                /           
 +z               forward           

elevation: in (-90, 90), from +y to -y is (-90, 90)
azimuth: in (-180, 180), from +z to +x is (0, 90)
```

* Wonder3D world & camera coordinate system:

![wonder3d_coordinate](_Example_Workflows/_Example_Outputs/wonder3d_coordinate.png)

* Three.js coordinate system: (z-axis is pointing towards you and is coming out of the screen)

![right_hand_coordinate_system](_Example_Workflows/_Example_Outputs/right_hand_coordinate_system.png)

* If you encounter OpenGL errors (e.g., `[F glutil.cpp:338] eglInitialize() failed`), then set `force_cuda_rasterize` to true on corresponding node
* If after the installation, your ComfyUI get stucked at starting or running, you could following the instruction in following link to solve the problem: [Code Hangs Indefinitely When Evaluating Neuron Models on GPU](https://github.com/lava-nc/lava-dl/discussions/211)


## Supporters
- [MrNeRF](https://twitter.com/janusch_patas)
