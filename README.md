# ComfyUI-3D-Pack
 An extensive node suite that enables ComfyUI to process 3D inputs (Mesh & UV Texture, etc) using cutting edge algorithms (3DGS, NeRF, Differentiable Rendering, SDS/VSD Optimization, etc.)

### Note: this project is still a WIP and not been released into ComFyUI package database yet

ps: I'll show some generated result soon

---
<br>

**[IMPORTANT!!!]** <br> Currently this package is only been tested in following setups:
- Windows 10/11
- Miniconda/Conda Python 3.11.7 
  - I tried install this package with ComfyUI embed python env first, but I can't find a way to build CUDA related libraries, e.g. diff-gaussian-rasterization, nvdiffrast, simple-knn.
- Torch version: 2.1.2+cu121/V.2.1.2+cu118

### Install:

Assume you have already downloaded [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

First download [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) (*One of the best way to manage a clean and separated python envirments*)
- Alternatively you can check this tutorial: [Installing ComfyUI with Miniconda On Windows and Mac](https://www.comflowy.com/preparation-for-study/install#step-two-download-comfyui)

```bash
# Go to your Your ComfyUI root directory, for my example:
cd C:\Users\reall\Softwares\ComfyUI_windows_portable 

conda create -p ./python_miniconda_env/ComfyUI python=3.11

# conda will tell what command to use to activate the env
conda activate C:\Users\reall\Softwares\ComfyUI_windows_portable\python_miniconda_env\ComfyUI

# This package also works with cu118
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r ./ComfyUI/requirements.txt

# Then go to ComfyUI-3D-Pack directory under the .\ComfyUI\custom_nodes for my example is:
cd C:\Users\reall\Softwares\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-3D-Pack
# Finally you can double click following .bat script or run it in CLI:
install.bat
```

Just in case `install.bat` may not working in your PC, you could also run the following commands under this package's root directory:
```bash
# First make sure the Conda env: python_miniconda_env\ComfyUI is activated, then go to Go to ComfyUI Root Directory\ComfyUI\custom_nodes\ComfyUI-3D-Pack and:
pip install -r requirements.txt

git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

pip install ./simple-knn

git clone --recursive https://github.com/NVlabs/nvdiffrast/`
pip install ./nvdiffrast
```

### Run:
Copy the files inside folder [__New_ComfyUI_Bats](./_New_ComfyUI_Bats/) to your ComfyUI root directory, and double click run_nvidia_gpu_miniconda.bat to start ComfyUI!
- Alternatively you can just activate the Conda env: python_miniconda_env\ComfyUI, and go to your ComfyUI root directory then run command `python ./ComfyUI/main.py`

---

### Currently support:
- For use case please check [example workflows](./_Example_Workflows/)
  - **Note:** you need to put [Images_enhanced](_Example_Workflows/_Example_Inputs_Files/Images_enhanced) and [Meshes](./_Example_Workflows/_Example_Inputs_Files/Meshes) under ComfyUI Root Directory\ComfyUI\input folder before you can run the example workflow
- Load 3D file (.obj, .ply, .glb)
- 3D Gaussian Splatting, with:
  - [Improved Differential Gaussian Rasterization](https://github.com/ashawkey/diff-gaussian-rasterization)
  - Better Compactness-based Densification method from [Gsgen](https://gsgen3d.github.io/), 
  - Support initialize gaussians from given 3D mesh (Optional)
  - Support mini-batch optimazation
  - Multi-View images as inputs
  - Export to .ply support

- Bake Multi-View images into UVTexture of given 3D mesh using [Nvdiffrast](https://github.com/NVlabs/nvdiffrast), supports:
  - Export to .obj, .ply, .glb

### To-Do Next:
- Add interactive 3D UI inside ComfuUI to visulaize training and generated results for 3D representations
- Add DMTet algorithm to allow convertion from points cloud(Gaussian/.ply) to mesh (.obj, .ply, .glb)
- Add a general SDS/VSD Optimization algorithm to allow training 3D representations with diffusion model, *The real fun begins here* ;) 
- Add a few best Nerf algorithms (No idea yet, [instant-ngp](https://github.com/NVlabs/instant-ngp) maybe?)