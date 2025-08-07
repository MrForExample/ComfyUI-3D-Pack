# This script is called by ComfyUI-Manager & Comfy-CLI after requirements.txt is installed: 
# https://github.com/ltdrdata/ComfyUI-Manager/tree/386af67a4c34db3525aa89af47a6f78c819926f2?tab=readme-ov-file#custom-node-support-guide

import sys
import os
from os.path import dirname
import glob
import subprocess
import traceback
import platform

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
        install_platform_packages,
        install_isolated_packages,
        wheels_dir_exists_and_not_empty,
        build_config,
        PYTHON_PATH,
        WHEELS_ROOT_ABS_PATH,
        PYTHON_VERSION
    )
    from shared_utils.log_utils import cstr
    
    # Ensure PyGithub is installed for downloading wheels
    try:
        import github
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "PyGithub"])

    def install_local_wheels(builds_dir):
        """Install all wheels from local directory"""
        wheel_files = glob.glob(os.path.join(builds_dir, "**/*.whl"), recursive=True)
        if not wheel_files:
            cstr("No wheel files found in directory").warning.print()
            return False
            
        success_count = 0
        for wheel_path in wheel_files:
            result = subprocess.run([PYTHON_PATH, "-s", "-m", "pip", "install", "--no-deps", "--force-reinstall", wheel_path], 
                                  text=True, capture_output=True)
            if result.returncode == 0:
                cstr(f"Successfully installed wheel: {os.path.basename(wheel_path)}").msg.print()
                success_count += 1
            else:
                cstr(f"Failed to install wheel {os.path.basename(wheel_path)}: {result.stderr}").error.print()
        
        if success_count == len(wheel_files):
            cstr(f"Successfully installed all {len(wheel_files)} wheels").msg.print()
            return True
        elif success_count > 0:
            cstr(f"Partially successful: {success_count}/{len(wheel_files)} wheels installed").warning.print()
            return False
        else:
            cstr("Failed to install any wheels").error.print()
            return False

    def try_wheels_first_approach():
        """Try wheels-first approach for all platforms"""
        platform_config_name = get_platform_config_name()
        builds_dir = os.path.join(WHEELS_ROOT_ABS_PATH, platform_config_name)
        
        cstr("Trying wheels-first approach...").msg.print()
        wheels_installed = False
        
        # Check existing wheels
        if wheels_dir_exists_and_not_empty(builds_dir):
            cstr(f"Found existing wheels in {builds_dir}").msg.print()
            if install_local_wheels(builds_dir):
                wheels_installed = True
                cstr("Installed wheels from local directory").msg.print()
        
        # Try downloading wheels from repository if not found locally
        if not wheels_installed:
            remote_builds_dir_name = f"{build_config.wheels_dir_name}/{platform_config_name}"
            if git_folder_parallel(build_config.repo_id, remote_builds_dir_name, recursive=True, root_outdir=builds_dir):
                cstr("Downloaded wheels from repository").msg.print()
                if install_local_wheels(builds_dir):
                    wheels_installed = True
                    cstr("Installed wheels from repository").msg.print()
            else:
                cstr("Could not download wheels from repository").warning.print()
        
        return wheels_installed

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
    
    # Install packages that needs specify remote url
    install_remote_packages(build_config.build_base_packages)
    install_platform_packages()
    
    # Install packages requiring special flags (like --no-build-isolation)
    if hasattr(build_config, 'isolated_packages'):
        install_isolated_packages(build_config.isolated_packages)
    
    # Check and install build tools if needed
    cstr("Checking build tools...").msg.print()
    build_tools = ["ninja", "cmake", "setuptools", "wheel"]
    for tool in build_tools:
        try:
            __import__(tool)
            cstr(f"{tool} is already installed").msg.print()
        except ImportError:
            cstr(f"Installing {tool}...").msg.print()
            result = subprocess.run(
                [PYTHON_PATH, "-m", "pip", "install", "--upgrade", tool],
                text=True, capture_output=True
            )
            if result.returncode != 0:
                cstr(f"[{tool} INSTALL ERROR]\n{result.stderr}").error.print()
                raise RuntimeError(f"Failed to install {tool}")
    
    # Main installation logic
    wheels_success = False
    
    # Get the target remote pre-built wheels directory name and path
    platform_config_name = get_platform_config_name()
    builds_dir = os.path.join(WHEELS_ROOT_ABS_PATH, platform_config_name)
    
    # Unified wheels-first approach for all platforms
    cstr("Starting unified installation process...").msg.print()
    
    # Step 1: Try wheels first
    wheels_success = try_wheels_first_approach()
    
    # Step 2: If wheels failed, try building
    if not wheels_success:
        cstr("Wheels installation failed, trying to build from source...").warning.print()
        if try_auto_build_all(builds_dir):
            install_local_wheels(builds_dir)
            wheels_success = True
            cstr("Successfully built and installed wheels").msg.print()
        else:
            cstr("Building wheels also failed").error.print()
    
    # Download python cpp source files for current python environment
    remote_pycpp_dir_name = f"_Python_Source_cpp/{PYTHON_VERSION}"
    python_root_dir = dirname(PYTHON_PATH)
    if git_folder_parallel(build_config.repo_id, remote_pycpp_dir_name, recursive=True, root_outdir=python_root_dir):
        cstr("Successfully downloaded required python cpp source files").msg.print()
    else:
        cstr(f"[WARNING] Couldn't download directory {remote_pycpp_dir_name} in remote repository {build_config.repo_id} to {python_root_dir}, some nodes may not work properly!").warning.print()
    
    cstr("Successfully installed Comfy3D! Let's Accelerate!").msg.print()
    
except Exception as e:
    traceback.print_exc()
    cstr("Comfy3D install failed: Dependency installation has failed. Please install manually: https://github.com/MrForExample/ComfyUI-3D-Pack/tree/main/_Pre_Builds/README.md.").error.print()


