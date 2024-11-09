## Automatic Build System
- **[_Build_Scripts/auto_build_all.py](_Pre_Builds/_Build_Scripts/auto_build_all.py)**: The build pipeline for a single given build  environment (e.g. py312, cu124)
- **[_Build_Scripts/build_config.yaml](_Pre_Builds/_Build_Scripts/build_config.yaml)**: Core config for build pipeline

## Build Required Packages Semi-Automatically
#### Build for Windows:
The following steps are the one that I took to build all required packages for install Comfy3D in Windows 11:
1. Install [Visual Studio Build Tools 2022/2019](https://visualstudio.microsoft.com/downloads/?q=build+tools) with Workloads: Desktop development with C++ (There are a few JIT torch cpp extension that builds in runtime)
   - Alternatively, according to [@doctorpangloss](https://github.com/MrForExample/ComfyUI-3D-Pack/issues/5), you can setup the c++/cuda build environments in windows by using [chocolatey](https://chocolatey.org/)
2. Download [CUDA 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network), [CUDA 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network) and [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network) environment.
3. Download [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) (*Note: [In a few edge cases Miniconda fails but Anaconda could fix the issue](https://github.com/MrForExample/ComfyUI-3D-Pack/issues/49)*)
4. Create conda environments:
   ```
    conda create --name Comy3D_py310_cu124 python=3.10
    conda create --name Comy3D_py311_cu124 python=3.11
    conda create --name Comy3D_py312_cu124 python=3.12

    conda create --name Comy3D_py310_cu121 python=3.10
    conda create --name Comy3D_py311_cu121 python=3.11
    conda create --name Comy3D_py312_cu121 python=3.12
	
    conda create --name Comy3D_py310_cu118 python=3.10
    conda create --name Comy3D_py311_cu118 python=3.11
    conda create --name Comy3D_py312_cu118 python=3.12
    ```
5. Download Comfy3D: `git clone https://github.com/MrForExample/ComfyUI-3D-Pack.git`
6. Swap CUDA Toolkit Versions to 12.4 ([Follow this 3 simple steps guide](https://github.com/bycloudai/SwapCudaVersionWindows))
7. Build Wheels with cuda 12.4:
    ```
    # Example of using the path to the miniconda envs on my PC, you may need to change the python path
    C:\Users\reall\Softwares\Miniconda\envs\Comy3D_py310_cu124\python ./ComfyUI-3D-Pack/_Pre_Builds/_Build_Scripts/auto_build_all.py
    C:\Users\reall\Softwares\Miniconda\envs\Comy3D_py311_cu124\python ./ComfyUI-3D-Pack/_Pre_Builds/_Build_Scripts/auto_build_all.py
    C:\Users\reall\Softwares\Miniconda\envs\Comy3D_py312_cu124\python ./ComfyUI-3D-Pack/_Pre_Builds/_Build_Scripts/auto_build_all.py
    ```
7. Swap CUDA Toolkit Versions to 12.1 ([Follow this 3 simple steps guide](https://github.com/bycloudai/SwapCudaVersionWindows))
8. Build Wheels with cuda 12.1:
    ```
    # Example of using the path to the miniconda envs on my PC, you may need to change the python path
    C:\Users\reall\Softwares\Miniconda\envs\Comy3D_py310_cu121\python ./ComfyUI-3D-Pack/_Pre_Builds/_Build_Scripts/auto_build_all.py
    C:\Users\reall\Softwares\Miniconda\envs\Comy3D_py311_cu121\python ./ComfyUI-3D-Pack/_Pre_Builds/_Build_Scripts/auto_build_all.py
    C:\Users\reall\Softwares\Miniconda\envs\Comy3D_py312_cu121\python ./ComfyUI-3D-Pack/_Pre_Builds/_Build_Scripts/auto_build_all.py
    ```
9. Swap CUDA Toolkit Versions to 11.8 ([Follow this 3 simple steps guide](https://github.com/bycloudai/SwapCudaVersionWindows))
10. Build Wheels with cuda 11.8:
    ```
    # Example of using the path to the miniconda envs on my PC, you may need to change the python path
    C:\Users\reall\Softwares\Miniconda\envs\Comy3D_py310_cu118\python ./ComfyUI-3D-Pack/_Pre_Builds/_Build_Scripts/auto_build_all.py
    C:\Users\reall\Softwares\Miniconda\envs\Comy3D_py311_cu118\python ./ComfyUI-3D-Pack/_Pre_Builds/_Build_Scripts/auto_build_all.py
    C:\Users\reall\Softwares\Miniconda\envs\Comy3D_py312_cu118\python ./ComfyUI-3D-Pack/_Pre_Builds/_Build_Scripts/auto_build_all.py
    ```
**Additional Information:**
- [Fix for Ninja doesn't support filename longer than 260 characters issue](https://github.com/ninja-build/ninja/issues/1900#issuecomment-1817532728)
- pytorch3d pre-build wheels from [MiroPsota/torch_packages_builder](https://github.com/MiroPsota/torch_packages_builder/releases)
- [Fix for CUDA Toolkits unsupported Microsoft Visual Studio version error](https://forums.developer.nvidia.com/t/problems-with-latest-vs2022-update/294150)
#### Build for Linux:

Then following commands are the one that I used to build all required packages for install Comfy3D in Ubuntu 22.04:

```bash
# Install build tools:
sudo apt update
sudo apt install gcc g++

# Install CUDA 12.1: 
# https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# Install CUDA 11.8:
# https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add CUDA environment variables:
export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
export PATH="/usr/local/cuda-12.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH"

# Managing Multiple CUDA:
# https://medium.com/@yushantripleseven/managing-multiple-cuda-cudnn-installations-ba9cdc5e2654
sudo update-alternatives --install /usr/local/cuda cuda /usr/local/cuda-11.8 10
sudo update-alternatives --install /usr/local/cuda cuda /usr/local/cuda-12.1 10

# Install Miniconda: 
# https://www.rosehosting.com/blog/how-to-install-miniconda-on-ubuntu-22-04/
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda-installer.sh
bash miniconda-installer.sh
# Open Miniconda prompt
source ~/.bashrc

# Create conda environments
conda create --name Comy3D_py310_cu121 python=3.10
conda create --name Comy3D_py311_cu121 python=3.11
conda create --name Comy3D_py312_cu121 python=3.12
conda create --name Comy3D_py310_cu118 python=3.10
conda create --name Comy3D_py311_cu118 python=3.11
conda create --name Comy3D_py312_cu118 python=3.12

# Download Comfy3D
git clone https://github.com/MrForExample/ComfyUI-3D-Pack.git

# Build Wheels with cuda 12.1:
sudo update-alternatives --config cuda
2
export CUDA_HOME="/usr/local/cuda-12.1/"
~/miniconda3/envs/Comy3D_py310_cu121/bin/python ./ComfyUI-3D-Pack/_Pre_Builds/_Build_Scripts/auto_build_all.py
~/miniconda3/envs/Comy3D_py311_cu121/bin/python ./ComfyUI-3D-Pack/_Pre_Builds/_Build_Scripts/auto_build_all.py
~/miniconda3/envs/Comy3D_py312_cu121/bin/python ./ComfyUI-3D-Pack/_Pre_Builds/_Build_Scripts/auto_build_all.py

# Build Wheels with cuda 11.8:
sudo update-alternatives --config cuda
1
export CUDA_HOME="/usr/local/cuda-11.8/"
~/miniconda3/envs/Comy3D_py310_cu118/bin/python ./ComfyUI-3D-Pack/_Pre_Builds/_Build_Scripts/auto_build_all.py
~/miniconda3/envs/Comy3D_py311_cu118/bin/python ./ComfyUI-3D-Pack/_Pre_Builds/_Build_Scripts/auto_build_all.py
~/miniconda3/envs/Comy3D_py312_cu118/bin/python ./ComfyUI-3D-Pack/_Pre_Builds/_Build_Scripts/auto_build_all.py
```

#### Use your Built Wheels:
In order to use your built wheels, you need to copy the folder `_Wheels_win_py{python_version}_cu{cuda_version}` to `{Your ComfyUI Root Folder Path}\ComfyUI\custom_nodes\ComfyUI-3D-Pack\_Pre_Builds\_Build_Wheels` and then reinstall the Comfy3D
<br>For example, `_Wheels_win_py311_cu121` to `C:\Users\reall\Softwares\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-3D-Pack\_Pre_Builds\_Build_Wheels`
<br>

### Install and run with docker:
Gpu support during Docker build time is required to install all requirenents. 
On Linux host you could setup `nvidia-container-runtime`. On Windows
it is quite different and not checked at moment.

#### Linux setup:
1. Install nvidia-container-runtime:
    ```bash
    sudo apt-get install nvidia-container-runtime
    ```
2. Edit/create the /etc/docker/daemon.json with content:
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
    
3. Restart docker daemon:
    ```bash
    sudo systemctl restart docker
    ```

4. Finally build and run docker container with:
    ```bash
    docker build -t comfy3d . && docker run --rm -it -p 8188:8188 --gpus all comfy3d
    ```