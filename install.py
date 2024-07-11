# This script is called by ComfyUI-Manager & Comfy-CLI after requirements.txt is installed: 
# https://github.com/ltdrdata/ComfyUI-Manager/tree/386af67a4c34db3525aa89af47a6f78c819926f2?tab=readme-ov-file#custom-node-support-guide

import sys
import os
from os.path import dirname
import glob
import subprocess
import traceback

if sys.argv[0] == 'install.py':
    sys.path.append('.')   # for portable version

COMFY3D_ROOT_ABS_PATH = dirname(__file__)
BUILD_SCRIPT_ROOT_ABS_PATH = os.path.join(COMFY3D_ROOT_ABS_PATH, "_Pre_Builds/_Build_Scripts")
sys.path.append(BUILD_SCRIPT_ROOT_ABS_PATH)

try:
    from build_utils import (
        get_platform_config_name,
        git_folder_parallel,
        install_remote_packages,
        build_config,
        PYTHON_PATH,
        WHEELS_ROOT_ABS_PATH,
        PYTHON_VERSION
    )
    from shared_utils.log_utils import cstr
    
    def try_auto_build_all(builds_dir):
        cstr(f"Try building all required packages...").msg.print()
        result = subprocess.run(
            [PYTHON_PATH, "auto_build_all.py", "--output_root_dir", builds_dir], 
            cwd=BUILD_SCRIPT_ROOT_ABS_PATH, text=True, capture_output=True
        )
        build_succeed = result.returncode == 0
        
        cstr(f"[Comfy3D BUILD LOG]\n{result.stdout}").msg.print()
        if not build_succeed:
            cstr(f"[Comfy3D BUILD ERROR LOG]\n{result.stderr}").error.print()
            
        return build_succeed
    
    def install_local_wheels(builds_dir):
        for wheel_path in glob.glob(os.path.join(builds_dir, "**/*.whl"), recursive=True):
            subprocess.run([PYTHON_PATH, "-s", "-m", "pip", "install", wheel_path])
            cstr(f"pip install {wheel_path} to {PYTHON_PATH}").msg.print()
    
    # Install packages that needs specify remote url
    install_remote_packages(build_config.remote_packages.keys())
    
    # Get the target remote pre-built wheels directory name and path
    platform_config_name = get_platform_config_name()
    remote_builds_dir_name = f"{build_config.wheels_dir_name}/{platform_config_name}"
    # Get the directory path which wheels will be downloaded or build into it
    builds_dir = os.path.join(WHEELS_ROOT_ABS_PATH, platform_config_name)
    
    build_succeed = False
    # Download pre-build wheels if exist
    if git_folder_parallel(build_config.repo_id, remote_builds_dir_name, recursive=True, root_outdir=builds_dir):
        build_succeed = True
    # Build the wheels if couldn't find pre-build wheels
    elif try_auto_build_all(builds_dir):
        build_succeed = True
            
    if build_succeed:
        install_local_wheels(builds_dir)
        cstr("Successfully installed all required wheels").msg.print()
    else:
        raise RuntimeError("Comfy3D build failed")
    
    # Download python cpp source files for current python environment
    remote_pycpp_dir_name = f"_Python_Source_cpp/{PYTHON_VERSION}"
    python_root_dir = dirname(PYTHON_PATH)
    if git_folder_parallel(build_config.repo_id, remote_pycpp_dir_name, recursive=True, root_outdir=python_root_dir):
        cstr("Successfully downloaded required python cpp source files").msg.print()
    else:
        cstr(f"[WARNING] Couldn't download directory {remote_pycpp_dir_name} in remote repository {build_config.repo_id} to {python_root_dir}, some nodes may not work properly!").warning.print()
    
    cstr("Successfully installed Comfy3D! Let's Accelerate!").msg.print()
    
except Exception as e:
    cstr("Comfy3D install failed: Dependency installation has failed. Please install manually: https://github.com/MrForExample/ComfyUI-3D-Pack/tree/main/_Pre_Builds/README.md.").error.print()
    traceback.print_exc()


