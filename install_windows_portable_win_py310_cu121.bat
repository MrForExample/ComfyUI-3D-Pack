@echo off

set "requirements_txt=%~dp0\requirements.txt"
set "requirements_post_txt=%~dp0\requirements_post_win_py310_cu121.txt"
set "python_exec=..\..\..\python_embeded\python.exe"

set "python_target_include=..\..\..\python_embeded\include"
set "python_target_libs=..\..\..\python_embeded\libs"
set "python_include=%~dp0\_Pre_Builds\_Python310_cpp\include"
set "python_libs=%~dp0\_Pre_Builds\_Python310_cpp\libs"

echo Starting to install ComfyUI-3D-Pack...

if exist "%python_exec%" (
    echo Installing with ComfyUI Windows Portable Python Embeded Environment

    xcopy /s /i "%python_include%" "%python_target_include%"
    xcopy /s /i "%python_libs%" "%python_target_libs%"

    "%python_exec%" -s -m pip install -r "%requirements_txt%"
    "%python_exec%" -s -m pip install -r "%requirements_post_txt%"

    "%python_exec%" -s -m pip install --force-reinstall xformers --index-url https://download.pytorch.org/whl/cu121
    
) else (
    echo ERROR: Cannot find ComfyUI Windows Portable Python Embeded Environment "%python_exec%"
)

echo Install Finished. Press any key to continue...

pause