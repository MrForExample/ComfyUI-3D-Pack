@echo off

set "requirements_txt=%~dp0\requirements.txt"
set "requirements_post_txt=%~dp0\requirements_post.txt"
set "python_miniconda_exec=..\..\..\python_miniconda_env\ComfyUI\python.exe"

echo Starting to install ComfyUI-3D-Pack...

if exist "%python_miniconda_exec%" (
    echo Installing with Miniconda Environment

    "%python_miniconda_exec%" -s -m pip install -r "%requirements_txt%"
    "%python_miniconda_exec%" -s -m pip install -r "%requirements_post_txt%"

    "%python_miniconda_exec%" -s -m pip install --force-reinstall xformers --index-url https://download.pytorch.org/whl/cu121
    
) else (
    echo ERROR: Cannot find Miniconda Environment "%python_miniconda_exec%"
)

echo Install Finished. Press any key to continue...

pause