@echo off

set "requirements_txt=%~dp0\requirements.txt"
set "python_miniconda_exec=..\..\..\python_miniconda_env\ComfyUI\python.exe"
set "python_exec=..\..\..\python_embeded\python.exe"

echo Installing ComfyUI-3D-Pack ...


if exist "%python_miniconda_exec%" (
    echo Installing with Miniconda Environment
    "%python_miniconda_exec%" -s -m pip install -r "%requirements_txt%"

    git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
    "%python_miniconda_exec%" -s -m pip install ./diff-gaussian-rasterization
    "%python_miniconda_exec%" -s -m pip install ./simple-knn
    "%python_miniconda_exec%" -s -m pip install git+https://github.com/NVlabs/nvdiffrast/
    "%python_miniconda_exec%" -s -m pip install git+https://github.com/ashawkey/kiuikit
)
else if exist "%python_exec%" (
    echo Installing with ComfyUI Portable
    "%python_exec%" -s -m pip install -r "%requirements_txt%"
) else (
    echo Installing with system Python
    pip install -r "%requirements_txt%"
)


echo Install Finished!

pause