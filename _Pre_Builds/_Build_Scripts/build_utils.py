import sys
import os
from os.path import dirname
import platform
import subprocess
import time
import glob

PYTHON_PATH = sys.executable

try:
    from omegaconf import OmegaConf
except ImportError as e:
    subprocess.run([PYTHON_PATH, "-s", "-m", "pip", "install", "OmegaConf"])
    from omegaconf import OmegaConf

BUILD_SCRIPT_ROOT_ABS_PATH = dirname(os.path.abspath(__file__))
build_config = OmegaConf.load(os.path.join(BUILD_SCRIPT_ROOT_ABS_PATH, "build_config.yaml"))

DEPENDENCIES_FILE_ABS_PATH = os.path.join(BUILD_SCRIPT_ROOT_ABS_PATH, build_config.dependencies)
BUILD_REQUIREMENTS_FILE_ABS_PATH = os.path.join(BUILD_SCRIPT_ROOT_ABS_PATH, build_config.build_requirements)
BUILD_ROOT_ABS_PATH = dirname(BUILD_SCRIPT_ROOT_ABS_PATH)
DEPENDENCIES_ROOT_ABS_PATH = os.path.join(BUILD_ROOT_ABS_PATH, build_config.dependencies_dir_name)
WHEELS_ROOT_ABS_PATH = os.path.join(BUILD_ROOT_ABS_PATH, build_config.wheels_dir_name)
LIBS_ROOT_ABS_PATH = os.path.join(BUILD_ROOT_ABS_PATH, build_config.libs_dir_name)

def get_os_type():
    if platform.system() == "Windows":
        return "win"
    elif platform.system() == "Linux":
        return "linux"
    else:
        raise NotImplementedError(f"Platform {platform.system()} not supported!")

def get_python_version():
    # Output: only first two version numbers, e.g. 3.12.4 -> py312
    return "py" + "".join(platform.python_version().split('.')[:-1])

def get_pytorch_version():
    return "torch" + build_config.remote_packages["torch"].version

def get_cuda_version():
    # Output: e.g. "cu121" or cu118
    try:
        result = subprocess.run(["nvcc", "--version"], text=True, capture_output=True)
        if result.returncode == 0:
            for cuda_version in build_config.supported_cuda_versions:
                if "cuda_" + cuda_version in result.stdout:
                    return "cu" + cuda_version.replace(".", "")
        
        # If nvcc command succeeded but no supported CUDA version found, use default
        print(f"Warning: No supported CUDA version detected, using default version: {build_config.cuda_version}")
        return "cu" + build_config.cuda_version.replace(".", "")
        
    except Exception as e:
        # If nvcc command failed or any other error occurred, CUDA is not installed
        print("CUDA toolkit not found. Please install CUDA 12.8:")
        print("https://developer.nvidia.com/cuda-12-8-0-download-archive")
        sys.exit(1)

OS_TYPE = get_os_type()
PYTHON_VERSION = get_python_version()
PYTORCH_VERSION = get_pytorch_version()
CUDA_VERSION = get_cuda_version()
build_config.cuda_version = CUDA_VERSION

def get_platform_config_name():
    platform_config_name = "_Wheels"
    
    # Add OS Type
    platform_config_name += "_" + OS_TYPE
    
    # Add Python Version
    platform_config_name += "_" + PYTHON_VERSION
    
    # Add Pytorch Version
    platform_config_name += "_" + PYTORCH_VERSION
    
    # Add CUDA Version
    platform_config_name += "_" + CUDA_VERSION
    
    return platform_config_name

def calculate_runtime(start_time):
    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return hours, minutes, seconds

def git_file(data):
    try: 
        import requests
        
        c, out, folder = data

        r = requests.get(c.download_url)
        output_path = c.path[len(folder):]
        output_abs_path = out + output_path
        os.makedirs(dirname(output_abs_path), exist_ok=True)
        with open(output_abs_path, 'wb') as f:
            print(f"Downloading {output_path} to {output_abs_path}")
            f.write(r.content)
            return False, c
        
    except Exception as e: 
        print(f"Exception in download_url(): {c.download_url}", e)
        return True, c

def git_contents_in_folder(repo, folder: str, recursive: bool=True):
    # Modified from https://github.com/Nordgaren/Github-Folder-Downloader
    contents = repo.get_contents(folder)
    file_contents = []
    for c in contents:
        if c.download_url is None:
            if recursive:
                file_contents += git_contents_in_folder(repo, c.path, recursive)
            continue
        
        file_contents.append(c)

    return file_contents
        
def git_folder_parallel(repo_id: str, folder: str, recursive: bool=True, root_outdir: str=""):
    try:
        from github import Github
        from multiprocessing import cpu_count 
        from concurrent.futures.thread import ThreadPoolExecutor
        
        start_time = time.time()
        
        # Get all the file contents from repo (i.e. urls, relative path)
        repo = Github().get_repo(repo_id)
        file_contents = git_contents_in_folder(repo, folder, recursive)
        
        inputs = zip(file_contents, [root_outdir] * len(file_contents), [folder] * len(file_contents))
        with ThreadPoolExecutor(max_workers=cpu_count() - 1) as executor:
            results = executor.map(git_file, inputs)
            #results = ThreadPool(cpus - 1).imap_unordered(git_file, inputs)
            for git_failed, c in results:
                if git_failed:
                    raise RuntimeError(f"Could not download {c.path} from {c.download_url}, please check your internet connection.")
                
        hours, minutes, seconds = calculate_runtime(start_time)
        print(f"Git folder finished in {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}")
        return True
    
    except Exception as e: 
        print(f"Couldn't download folder {folder} from repo {repo_id}", e)
        return False
    
def is_package_installed(package_name, required_version=None):
    """Check if a package is installed with the required version"""
    try:
        import importlib.metadata
        installed_version = importlib.metadata.version(package_name)
        
        if required_version is None:
            return True
        
        # Handle versions with suffixes like "2.7.1+cu128"
        installed_base_version = installed_version.split('+')[0]
        required_base_version = required_version.split('+')[0]
        
        # For PyTorch-related packages, we're more flexible with patch versions
        if package_name in ['torch', 'torchvision', 'xformers']:
            # Parse version components
            try:
                installed_parts = [int(x) for x in installed_base_version.split('.')]
                required_parts = [int(x) for x in required_base_version.split('.')]
                print(f"[DEBUG] {package_name}: version parts - installed={installed_parts}, required={required_parts}")
                
                # Pad with zeros if needed (e.g., "2.7" vs "2.7.0")
                max_len = max(len(installed_parts), len(required_parts))
                installed_parts.extend([0] * (max_len - len(installed_parts)))
                required_parts.extend([0] * (max_len - len(required_parts)))
                print(f"[DEBUG] {package_name}: padded parts - installed={installed_parts}, required={required_parts}")
                
                # Check major.minor compatibility, allow newer patch versions
                if len(installed_parts) >= 2 and len(required_parts) >= 2:
                    # Major and minor versions must match
                    major_minor_match = (installed_parts[0] == required_parts[0] and 
                                       installed_parts[1] == required_parts[1])
                    
                    if major_minor_match:
                        # Patch version can be equal or higher
                        if len(installed_parts) >= 3 and len(required_parts) >= 3:
                            patch_ok = installed_parts[2] >= required_parts[2]
                            return patch_ok
                        return True
                return False
            except ValueError as e:
                # If version parsing fails, fall back to exact match
                exact_match = installed_base_version == required_base_version
                return exact_match
        else:
            # For other packages, require exact version match
            exact_match = installed_base_version == required_base_version
            return exact_match
            
    except (importlib.metadata.PackageNotFoundError, Exception) as e:
        return False

def install_remote_packages(package_names):
    for package_name in package_names:
        original_package_name = package_name
        required_version = None
        
        if package_name in build_config.remote_packages:
            package_attr = build_config.remote_packages[package_name]
            if hasattr(package_attr, "version"):
                required_version = package_attr.version
                package_name += f"=={required_version}"
            
            # Check if package is already installed with the correct version
            if is_package_installed(original_package_name, required_version):
                print(f"Package {original_package_name} (version {required_version or 'any'}) is already installed, skipping...")
                continue
            
            print(f"Installing {original_package_name} version {required_version or 'latest'}...")
            
            if hasattr(package_attr, "url"):
                url_option = package_attr.url_option if hasattr(package_attr, "url_option") else "--index-url"
                
                subprocess.run([
                    PYTHON_PATH, "-s", "-m", "pip", "install", 
                    package_name, url_option, package_attr.url
                ])
                continue
        else:
            # Check if package is already installed
            if is_package_installed(original_package_name):
                print(f"Package {original_package_name} is already installed, skipping...")
                continue
            
            print(f"Installing {original_package_name}...")

        subprocess.run([PYTHON_PATH, "-s", "-m", "pip", "install", package_name])

def install_platform_packages():
    if hasattr(build_config, 'platform_packages') and OS_TYPE in build_config.platform_packages:
        packages = build_config.platform_packages[OS_TYPE]
        for package in packages:
            # Extract package name (without version constraints)
            package_name = package.split('==')[0].split('>=')[0].split('<=')[0].split('<')[0].split('>')[0]
            
            if is_package_installed(package_name):
                print(f"Platform package {package_name} is already installed, skipping...")
                continue
            
            print(f"Installing platform package {package}...")
            subprocess.run([PYTHON_PATH, "-s", "-m", "pip", "install", package])

def install_isolated_packages(package_names):
    """Install packages with special flags like --no-build-isolation"""
    for package_name in package_names:
        if package_name in build_config.remote_packages:
            package_attr = build_config.remote_packages[package_name]
            
            # Check if package is already installed
            if is_package_installed(package_name):
                print(f"Package {package_name} is already installed, skipping...")
                continue
            
            print(f"Installing isolated package {package_name}...")
            
            if hasattr(package_attr, "url"):
                # Build command with install flags
                cmd = [PYTHON_PATH, "-s", "-m", "pip", "install"]
                if hasattr(package_attr, "install_flags"):
                    cmd.extend(package_attr.install_flags)
                cmd.append(package_attr.url)
                
                subprocess.run(cmd)
            else:
                print(f"No URL found for isolated package {package_name}")
        else:
            print(f"Isolated package {package_name} not found in config")

def wheels_dir_exists_and_not_empty(builds_dir):
    if not os.path.exists(builds_dir):
        return False
    
    # Check if directory has any .whl files
    wheel_files = glob.glob(os.path.join(builds_dir, "**/*.whl"), recursive=True)
    return len(wheel_files) > 0