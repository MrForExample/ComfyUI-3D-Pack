import sys
import os
from os.path import dirname
import platform
import subprocess
import time
import urllib.request
import zipfile
import shutil

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
    result = subprocess.run(["nvcc", "--version"], text=True, capture_output=True)
    if result.returncode == 0:
        for cuda_version in build_config.supported_cuda_versions:
            if "cuda_" + cuda_version in result.stdout:
                return "cu" + cuda_version.replace(".", "")
        
    raise RuntimeError(f"Please install/reinstall CUDA tookit with any of the following supported version: {build_config.supported_cuda_versions}")

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
    
def install_remote_packages(package_names):
    for package_name in package_names:
        if package_name in build_config.remote_packages:
            package_attr = build_config.remote_packages[package_name]
            if hasattr(package_attr, "version"):
                package_name += f"=={package_attr.version}"
            if hasattr(package_attr, "url"):
                url_option = package_attr.url_option if hasattr(package_attr, "url_option") else "--index-url"
                
                subprocess.run([
                    PYTHON_PATH, "-s", "-m", "pip", "install", 
                    package_name, url_option, package_attr.url
                ])
                continue

        subprocess.run([PYTHON_PATH, "-s", "-m", "pip", "install", package_name])


# windows build tool dependencies
VS_BUILD_TOOLS_URL = "https://aka.ms/vs/17/release/vs_BuildTools.exe"
VS_INSTALLER = os.path.join(os.getcwd(), "vs_BuildTools.exe")

NINJA_URL = "https://github.com/ninja-build/ninja/releases/latest/download/ninja-win.zip"
NINJA_DIR = os.path.join(os.getenv("LOCALAPPDATA"), "Ninja")
NINJA_EXE = os.path.join(NINJA_DIR, "ninja.exe")

VS_INSTALLER = os.path.join(os.getcwd(), "vs_BuildTools.exe")
def install_windows_build_tool_dependencies():
    install_ninja()
    install_vs_build_tools()

def is_ninja_installed():
    """Check if Ninja is installed and accessible."""

    # Check if ninja.exe exists in the expected location
    ninja_exists = os.path.exists(NINJA_EXE)

    if not ninja_exists:
        print("❌ Ninja NOT found in expected location.")
        return False

    # Check if it's available in PATH
    if shutil.which("ninja") is not None:
        print("✅ Ninja is accessible via PATH.")
        return True
    else:
        print("⚠️ Ninja is installed but NOT found in PATH. Trying to fix...")
        add_ninja_to_path()
        return shutil.which("ninja") is not None


def add_ninja_to_path():
    """Adds Ninja to the system PATH permanently."""
    ninja_path = NINJA_DIR
    current_path = os.environ.get("PATH", "")

    if ninja_path not in current_path:
        print("🔧 Adding Ninja to system PATH...")
        subprocess.run([
            "setx", "PATH", f"{current_path};{ninja_path}"
        ], shell=True, check=True)
        print(f"✅ Ninja has been added to PATH: {ninja_path}")
    else:
        print("✅ Ninja is already in PATH.")

def install_ninja():
    """Downloads and installs Ninja build system automatically."""

    # If Ninja is already installed, skip installation
    if is_ninja_installed():
        print(f"✅ Ninja is already installed at {NINJA_EXE}")
        return

    print("🔽 Downloading Ninja...")
    ninja_zip = os.path.join(os.getcwd(), "ninja.zip")
    urllib.request.urlretrieve(NINJA_URL, ninja_zip)

    print(f"📦 Extracting Ninja to {NINJA_DIR}...")
    os.makedirs(NINJA_DIR, exist_ok=True)  # Ensure directory exists

    with zipfile.ZipFile(ninja_zip, "r") as zip_ref:
        zip_ref.extractall(NINJA_DIR)

    os.remove(ninja_zip)  # Clean up

    print(f"✅ Ninja installed successfully at {NINJA_DIR}")

    # Add Ninja to system PATH
    add_ninja_to_path()

def is_msvc_installed():
    """Check if MSVC compiler (`cl.exe`) is installed."""
    return shutil.which("cl") is not None

def install_vs_build_tools():
    """Downloads and installs Visual Studio Build Tools with MSVC and Windows SDK silently."""

    # If MSVC is installed, skip installation
    if is_msvc_installed():
        print("✅ MSVC Compiler is already installed.")
        return

    print("🔽 Downloading Visual Studio Build Tools...")
    urllib.request.urlretrieve(VS_BUILD_TOOLS_URL, VS_INSTALLER)

    print("⚙️ Installing Visual Studio Build Tools (this may take a few minutes)...")
    subprocess.run([
        VS_INSTALLER,
        "--quiet", "--wait",
        "--add", "Microsoft.VisualStudio.Workload.VCTools",        # C++ Build Tools Workload
        "--add", "Microsoft.VisualStudio.Component.VC.CoreBuildTools",  # MSVC Compiler
        "--add", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",  # 64-bit toolset
        "--add", "Microsoft.VisualStudio.Component.Windows10SDK.19041",  # Windows SDK
        "--add", "Microsoft.VisualStudio.Component.VC.Redist.14.Latest"  # Standard Library Headers
    ], check=True)

    print("✅ MSVC Compiler and Windows SDK installed successfully!")

    # Cleanup downloaded installer
    os.remove(VS_INSTALLER)
