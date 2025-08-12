import sys
import os
from os.path import dirname
import subprocess
import re
import time
import platform

BUILD_SCRIPT_ROOT_ABS_PATH = dirname(__file__)
sys.path.append(BUILD_SCRIPT_ROOT_ABS_PATH)

from build_utils import (
    get_platform_config_name,
    calculate_runtime,
    git_folder_parallel,
    install_remote_packages,
    install_platform_packages,
    build_config,
    PYTHON_PATH,
    DEPENDENCIES_FILE_ABS_PATH,
    BUILD_REQUIREMENTS_FILE_ABS_PATH,
    DEPENDENCIES_ROOT_ABS_PATH,
    WHEELS_ROOT_ABS_PATH,
    LIBS_ROOT_ABS_PATH
)

def read_dependencies(file_path):
    # Download required source libraries from remote
    if not os.path.exists(LIBS_ROOT_ABS_PATH) or not os.listdir(LIBS_ROOT_ABS_PATH):
        if not git_folder_parallel(build_config.repo_id, build_config.libs_dir_name, recursive=True, root_outdir=LIBS_ROOT_ABS_PATH):
            raise RuntimeError(f"Comfy3D install failed, couldn't download directory {build_config.libs_dir_name} in remote repository {build_config.repo_id}")
    dependencies = [ f.path for f in os.scandir(LIBS_ROOT_ABS_PATH) if f.is_dir() ]
    #dependencies = []  #ignore libraries for debug
    with open(file_path, "r") as f:
        dependencies += [line.strip() for line in f]
        
    return dependencies
    
def get_dependency_dir(dependency):
    is_url = dependency[:6] == "https:"
    if is_url:
        dependency_name = re.split(r'[/\.]+', dependency)[-2]
        dependency_dir = os.path.join(DEPENDENCIES_ROOT_ABS_PATH, dependency_name)
    else:
        dependency_dir = os.path.join(LIBS_ROOT_ABS_PATH, dependency)
    return dependency_dir, is_url
    
def setup_build_env():
    # Set CMake argumens fo build packages based on CUDA
    os.environ["CMAKE_ARGS"] = "-DBUILD_opencv_world=ON -DWITH_CUDA=ON -DCUDA_FAST_MATH=ON -DWITH_CUBLAS=ON -DCUDA_ARCH_PTX=9.0 -DWITH_NVCUVID=ON"
    subprocess.run([PYTHON_PATH, "-s", "-m", "pip", "install", "-r", BUILD_REQUIREMENTS_FILE_ABS_PATH])
    
    install_remote_packages(build_config.build_base_packages)
    install_platform_packages()

def clone_or_update_read_dependency(dependency_url, dependency_dir):
    if os.path.exists(dependency_dir):
        # Dependency already exists, update it
        subprocess.run(["git", "pull"], cwd=dependency_dir)
    else:
        # Clone the dependency
        subprocess.run(["git", "clone", "--recursive", dependency_url, dependency_dir])

def build_python_wheel(dependency_dir, output_dir):
    """Build wheel using available build method"""
    print(f"Checking {dependency_dir}")
    
    setup_py_path = os.path.join(dependency_dir, "setup.py")
    pyproject_toml_path = os.path.join(dependency_dir, "pyproject.toml")
    
    has_setup_py = os.path.exists(setup_py_path)
    has_pyproject = os.path.exists(pyproject_toml_path)
    
    if not has_setup_py and not has_pyproject:
        print(f"No setup.py or pyproject.toml found in {dependency_dir}, skipping")
        return None  # Skipped
    
    # Try different build methods
    if has_setup_py:
        print(f"Building wheel for {dependency_dir} using setup.py")
        result = subprocess.run(
            [PYTHON_PATH, "setup.py", "bdist_wheel", "--dist-dir", output_dir], 
            cwd=dependency_dir, text=True, capture_output=True
        )
    else:
        print(f"Building wheel for {dependency_dir} using pip wheel")
        result = subprocess.run(
            [PYTHON_PATH, "-m", "pip", "wheel", ".", "--no-deps", "--wheel-dir", output_dir], 
            cwd=dependency_dir, text=True, capture_output=True
        )
    
    build_failed = result.returncode != 0
    
    if build_failed:
        print(f"[Wheel BUILD LOG]\n{result.stdout}")
        print(f"[Wheel BUILD ERROR LOG]\n{result.stderr}")
        
        # If first method failed and we have both files, try the other method
        if has_setup_py and has_pyproject:
            print(f"Retrying with pip wheel method...")
            result = subprocess.run(
                [PYTHON_PATH, "-m", "pip", "wheel", ".", "--no-deps", "--wheel-dir", output_dir], 
                cwd=dependency_dir, text=True, capture_output=True
            )
            build_failed = result.returncode != 0
            
            if build_failed:
                print(f"[Retry BUILD LOG]\n{result.stdout}")
                print(f"[Retry BUILD ERROR LOG]\n{result.stderr}")
    
    # If wheel building failed, try direct installation as fallback
    if build_failed:
        print(f"Wheel building failed, trying direct installation...")
        result = subprocess.run(
            [PYTHON_PATH, "-m", "pip", "install", "."], 
            cwd=dependency_dir, text=True, capture_output=True
        )
        
        if result.returncode == 0:
            print(f"Direct install {dependency_dir} Succeed (no wheel created)")
            return "installed"  # Installed directly, no wheel
        else:
            print(f"[Direct INSTALL LOG]\n{result.stdout}")
            print(f"[Direct INSTALL ERROR LOG]\n{result.stderr}")
            print(f"Build and install {dependency_dir} Failed")
            return False  # Failed
    else:
        print(f"Build {dependency_dir} Succeed")
        return True  # Success

def main(args):
    start_time = time.time()
    
    # Create the output folder if it doesn't exist
    output_root_path = args.output_root_dir
    if output_root_path is None:
        output_root_path = os.path.join(WHEELS_ROOT_ABS_PATH, get_platform_config_name())
    elif not os.path.isabs(output_root_path):
        output_root_path = os.path.join(WHEELS_ROOT_ABS_PATH, output_root_path)
        
    os.makedirs(output_root_path, exist_ok=True)
    
    # Setup environment variables and fundamental packages for build
    setup_build_env()
    
    # Read dependencies names from the text file
    dependencies = read_dependencies(DEPENDENCIES_FILE_ABS_PATH)
    
    # Build all dependencies and move them to output_root_path
    failed_build = []
    skipped_build = []
    successful_build = []
    direct_installed = []
    for dependency in dependencies:
        dependency_dir, is_url = get_dependency_dir(dependency)
        if is_url:
            clone_or_update_read_dependency(dependency, dependency_dir)
        
        # Check if directory exists
        if not os.path.exists(dependency_dir):
            print(f"Directory {dependency_dir} does not exist, skipping")
            skipped_build.append(dependency)
            continue
            
        result = build_python_wheel(dependency_dir, output_root_path)
        if result is None:
            skipped_build.append(dependency)
        elif result == "installed":
            direct_installed.append(dependency)
        elif result is False:
            failed_build.append(dependency)
        else:
            successful_build.append(dependency)
            
    hours, minutes, seconds = calculate_runtime(start_time)
    print(f"Build all dependencies finished in {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}")
    print(f"Results: {len(successful_build)} wheels built, {len(direct_installed)} direct installs, {len(skipped_build)} skipped, {len(failed_build)} failed")
    
    if len(skipped_build) > 0:
        print(f"Skipped dependencies (no setup.py/pyproject.toml): {skipped_build}")
    
    if len(direct_installed) > 0:
        print(f"Direct installed dependencies (no wheel): {direct_installed}")
            
    if len(failed_build) == 0:
        print(f"[Comfy3D BUILD SUCCEED]")
    else:
        raise RuntimeError(f"[Comfy3D BUILD FAILED]: Following dependencies failed to build: {failed_build}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_root_dir",
        type=str,
        default=None,
        help="Path to the target build directory",
    )
    args = parser.parse_args()
    
    main(args)