import sys
import os
from os.path import dirname
import platform
import subprocess
import time

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

def get_cuda_version():
    # Output: e.g. "cu121" or cu118
    result = subprocess.run(["nvcc", "--version"], text=True, capture_output=True)
    if result.returncode == 0:
        for cuda_version in build_config.supported_cuda_versions:
            if "cuda_" + cuda_version in result.stdout:
                return "cu" + cuda_version.replace(".", "")
        
    raise RuntimeError(f"Please install/reinstall CUDA tookit with any of the following supported version: {build_config.supported_cuda_versions}")

OS_TYPE = get_os_type()
PYTHON_VERSION = get_python_version()
CUDA_VERSION = get_cuda_version()
build_config.cuda_version = CUDA_VERSION

def get_platform_config_name():
    platform_config_name = "_Wheels"
    
    # Add OS Type
    platform_config_name += "_" + OS_TYPE
    
    # Add Python Version
    platform_config_name += "_" + PYTHON_VERSION
    
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
    
def install_remote_packages(package_names):
    for package_name in package_names:
        package_attr = build_config.remote_packages[package_name]
        if hasattr(package_attr, "version"):
            package_name += f"=={package_attr.version}"
        url_option = package_attr.url_option if hasattr(package_attr, "url_option") else "--index-url"
            
        subprocess.run([
            PYTHON_PATH, "-s", "-m", "pip", "install", 
            package_name, url_option, package_attr.url
        ])