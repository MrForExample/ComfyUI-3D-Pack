import sys
import os
from os.path import dirname
import glob
import platform
import subprocess
import re

SCRIPT_ROOT_ABS_PATH = dirname(os.path.abspath(__file__))
DEPENDENCIES_FILE_ABS_PATH = os.path.join(SCRIPT_ROOT_ABS_PATH, "dependencies.txt")
BUILD_REQUIREMENTS_FILE_ABS_PATH = os.path.join(SCRIPT_ROOT_ABS_PATH, "build_requirements.txt")
BUILD_ROOT_ABS_PATH = dirname(SCRIPT_ROOT_ABS_PATH)
DEPENDENCIES_ROOT_ABS_PATH = os.path.join(BUILD_ROOT_ABS_PATH, "_Build_Dependencies")
LIBS_ROOT_ABS_PATH = os.path.join(BUILD_ROOT_ABS_PATH, "_Libs")

def get_parent_dirpath_n_level_up(abs_path, n=1):
    for i in range(n):
        abs_path = dirname(abs_path)
    return abs_path

COMFY_PYTHON_ABS_PATH = os.path.join(get_parent_dirpath_n_level_up(BUILD_ROOT_ABS_PATH, 4), "python_embeded", "python")

def get_platform_config_name(cuda_version):
    platform_config_name = "_Wheels"
    
    # Add OS Type
    if platform.system() == 'Windows':
        platform_config_name += "_win"
    elif platform.system() == "Linux":
        platform_config_name += "_linux"
    else:
        raise NotImplementedError(f"Platform {platform.system()} not supported!")
    
    # Add Python Version, only first two version numbers, e.g. 3.12.4 -> 312
    platform_config_name += "_py" + "".join(platform.python_version().split('.')[:-1])
    
    # Add CUDA Version
    platform_config_name += "_" + cuda_version
    
    return platform_config_name

def read_dependencies(file_path):
    dependencies = [ f.path for f in os.scandir(LIBS_ROOT_ABS_PATH) if f.is_dir() ]
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
    
def setup_build_env(python_path, args):
    # Set CMake argumens fo build packages based on CUDA
    os.environ["CMAKE_ARGS"]="-DBUILD_opencv_world=ON -DWITH_CUDA=ON -DCUDA_FAST_MATH=ON -DWITH_CUBLAS=ON -DCUDA_ARCH_PTX=9.0 -DWITH_NVCUVID=ON"
    subprocess.run([python_path, "-s", "-m", "pip", "install", "-r", BUILD_REQUIREMENTS_FILE_ABS_PATH])
    subprocess.run([
        python_path, "-s", "-m", "pip", "install", 
        f"torch=={args.torch_version}", f"torchvision=={args.torchvision_version}", 
        "--index-url", f"https://download.pytorch.org/whl/{args.cuda_version}"
    ])

def clone_or_update_read_dependency(dependency_url, dependency_dir):
    if os.path.exists(dependency_dir):
        # Dependency already exists, update it
        subprocess.run(["git", "pull"], cwd=dependency_dir)
    else:
        # Clone the dependency
        subprocess.run(["git", "clone", "--recursive", dependency_url, dependency_dir])

def build_python_wheel(python_path, dependency_dir, output_dir):
    # Build wheel and move the wheel file we just built to the output directory
    subprocess.run([python_path, "setup.py", "bdist_wheel", "--dist-dir", output_dir], cwd=dependency_dir)

def main(args):
    # Get python executable from path
    python_path = sys.executable
    
    # Create the output folder if it doesn't exist
    output_root_path = args.output_root_dir
    if output_root_path is None:
        output_root_path = os.path.join(BUILD_ROOT_ABS_PATH, get_platform_config_name(args.cuda_version))
    elif not os.path.isabs(output_root_path):
        output_root_path = os.path.join(BUILD_ROOT_ABS_PATH, output_root_path)
        
    os.makedirs(output_root_path, exist_ok=True)

    # Read dependencies names from the text file
    dependencies = read_dependencies(DEPENDENCIES_FILE_ABS_PATH)
    print(dependencies)
    
    # Setup environment variables and fundamental packages for build
    setup_build_env(python_path, args)
    
    # Build all dependencies and move them to output_root_path
    for dependency in dependencies:
        dependency_dir, is_url = get_dependency_dir(dependency)
        if is_url:
            clone_or_update_read_dependency(dependency, dependency_dir)
            
        build_python_wheel(python_path, dependency_dir, output_root_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_root_dir",
        type=str,
        default=None,
        help="Path to the target build directory",
    )
    parser.add_argument(
        "--cuda_version",
        type=str,
        default="cu121",
        choices="[cu121, cu118]",
        help="CUDA Toolkit version",
    )
    parser.add_argument(
        "--torch_version",
        type=str,
        default="2.3.0",
        help="Pytorch version",
    )
    parser.add_argument(
        "--torchvision_version",
        type=str,
        default="0.18.0",
        help="Pytorch vision library version",
    )
    args = parser.parse_args()
    
    main(args)
    
    #C:\Users\reall\Softwares\Miniconda\envs\Comfy3D_Builds_py310_cu118\python _Pre_Builds\_Build_Scripts\auto_build_all.py --cuda_version cu118