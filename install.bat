@echo off

set "requirements_txt=%~dp0\requirements.txt"
set "python_miniconda_exec=..\..\..\python_miniconda_env\ComfyUI\python.exe"

echo Installing ComfyUI-3D-Pack ...


if exist "%python_miniconda_exec%" (
    echo Installing with Miniconda Environment
    "%python_miniconda_exec%" -s -m pip install -r "%requirements_txt%"

    git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
    "%python_miniconda_exec%" -s -m pip install ./diff-gaussian-rasterization
    "%python_miniconda_exec%" -s -m pip install ./simple-knn
    git clone --recursive https://github.com/NVlabs/nvdiffrast
    "%python_miniconda_exec%" -s -m pip install ./nvdiffrast
    "%python_miniconda_exec%" -s -m pip install git+https://github.com/facebookresearch/pytorch3d.git@stable
)
else (
    echo ERROR: Cannot find Miniconda Environment "%python_miniconda_exec%"
)


echo Install Finished!

pause