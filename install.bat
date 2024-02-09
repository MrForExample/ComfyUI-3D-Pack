@echo off

set "requirements_txt=%~dp0\requirements.txt"
set "python_miniconda_exec=..\..\..\python_miniconda_env\ComfyUI\python.exe"

echo Installing ComfyUI-3D-Pack ...


if exist "%python_miniconda_exec%" (
    echo Installing with Miniconda Environment
    "%python_miniconda_exec%" -s -m pip install -r "%requirements_txt%"

    "%python_miniconda_exec%" -s -m pip install -U xformers --index-url https://download.pytorch.org/whl/cu121

    set "python_miniconda_exec=..\..\..\..\..\..\..\python_miniconda_env\ComfyUI\python.exe"

    cd tgs/models/snowflake/pointnet2_ops_lib && "%python_miniconda_exec%" -s setup.py install && cd ../../../../
) else (
    echo ERROR: Cannot find Miniconda Environment "%python_miniconda_exec%"
)


echo Install Finished!

pause