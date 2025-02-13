import urllib.request
import zipfile
import shutil
import os
import subprocess

VS_BUILD_TOOLS_URL = "https://aka.ms/vs/17/release/vs_BuildTools.exe"
VS_INSTALLER = os.path.join(os.getcwd(), "vs_BuildTools.exe")

NINJA_URL = "https://github.com/ninja-build/ninja/releases/latest/download/ninja-win.zip"
NINJA_DIR = os.path.join(os.getenv("LOCALAPPDATA"), "Ninja")
NINJA_EXE = os.path.join(NINJA_DIR, "ninja.exe")

def install_windows_build_tool_dependencies():
    install_ninja()
    install_vs_build_tools()

def is_ninja_installed():
    """Check if Ninja is installed and accessible."""

    # Check if ninja.exe exists in the expected location
    ninja_exists = os.path.exists(NINJA_EXE)

    if not ninja_exists:
        print("Ninja NOT found in expected location.")
        return False

    # Check if it's available in PATH
    if shutil.which("ninja") is not None:
        print("Ninja is accessible via PATH.")
        return True
    else:
        print("Ninja is installed but NOT found in PATH. Trying to fix...")
        add_ninja_to_path()
        return shutil.which("ninja") is not None


def add_ninja_to_path():
    """Adds Ninja to the system PATH permanently."""
    ninja_path = NINJA_DIR
    current_path = os.environ.get("PATH", "")

    if ninja_path not in current_path:
        print("Adding Ninja to system PATH...")
        subprocess.run([
            "setx", "PATH", f"{current_path};{ninja_path}"
        ], shell=True, check=True)
        print(f"Ninja has been added to PATH: {ninja_path}")
    else:
        print("Ninja is already in PATH.")

def install_ninja():
    """Downloads and installs Ninja build system automatically."""

    # If Ninja is already installed, skip installation
    if is_ninja_installed():
        print(f"Ninja is already installed at {NINJA_EXE}")
        return

    print("Downloading Ninja...")
    ninja_zip = os.path.join(os.getcwd(), "ninja.zip")
    urllib.request.urlretrieve(NINJA_URL, ninja_zip)

    print(f"Extracting Ninja to {NINJA_DIR}...")
    os.makedirs(NINJA_DIR, exist_ok=True)  # Ensure directory exists

    with zipfile.ZipFile(ninja_zip, "r") as zip_ref:
        zip_ref.extractall(NINJA_DIR)

    os.remove(ninja_zip)  # Clean up

    print(f"Ninja installed successfully at {NINJA_DIR}")

    # Add Ninja to system PATH
    add_ninja_to_path()

# URL and filename for the VS Build Tools installer.
VS_BUILD_TOOLS_URL = "https://aka.ms/vs/17/release/vs_BuildTools.exe"
VS_INSTALLER = "vs_BuildTools.exe"

def find_msvc_path():
    """
    Find the MSVC toolset base path using vswhere.
    This looks for the latest Visual Studio installation that has the 64-bit C++ toolset.
    """
    vswhere_path = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if not os.path.exists(vswhere_path):
        print("vswhere.exe not found. Please ensure the Visual Studio Installer is installed.")
        return None

    try:
        result = subprocess.run([
            vswhere_path,
            "-latest",
            "-products", "*",
            "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property", "installationPath"
        ], capture_output=True, text=True, check=True)

        vs_install_path = result.stdout.strip()
        if not vs_install_path:
            print("Visual Studio installation not found by vswhere.")
            return None

        # The MSVC toolset is typically located under the "VC\Tools\MSVC" folder.
        msvc_base_path = os.path.join(vs_install_path, "VC", "Tools", "MSVC")
        if not os.path.exists(msvc_base_path):
            print("MSVC Tools folder not found in the Visual Studio installation.")
            return None
        return msvc_base_path

    except subprocess.CalledProcessError as e:
        print(f"Error running vswhere: {e}")
        return None

def find_latest_msvc_bin():
    """
    Locate the latest MSVC `cl.exe` binary path by listing the versions available
    in the MSVC toolset folder.
    """
    msvc_base_path = find_msvc_path()
    if not msvc_base_path:
        return None

    try:
        # Filter out non-directory entries and assume folder names are version numbers.
        versions = [v for v in os.listdir(msvc_base_path) if os.path.isdir(os.path.join(msvc_base_path, v))]
        if not versions:
            print("No MSVC version directories found.")
            return None

        # Choose the latest version by comparing version tuples.
        latest_version = max(versions, key=lambda v: tuple(map(int, v.split('.'))))
        msvc_bin_path = os.path.join(msvc_base_path, latest_version, "bin", "Hostx64", "x64")
        if os.path.exists(msvc_bin_path):
            return msvc_bin_path
        else:
            print(f"MSVC bin path not found for version {latest_version}.")
            return None
    except Exception as e:
        print(f"Error locating MSVC bin path: {e}")
        return None

def is_msvc_installed():
    """
    Check if the MSVC compiler (`cl.exe`) is installed and accessible via the current PATH.
    """
    if shutil.which("cl"):
        print("MSVC Compiler (`cl.exe`) is installed and accessible!")
        return True
    else:
        print("MSVC Compiler (`cl.exe`) NOT found in PATH!")
        return False

def add_msvc_to_user_path():
    """
    Adds the MSVC binary path to the current user's PATH permanently.
    This does not require administrator privileges.
    """
    msvc_bin_path = find_latest_msvc_bin()
    if not msvc_bin_path:
        print("Unable to locate `cl.exe`, MSVC installation might be broken.")
        return

    print(f"Adding `{msvc_bin_path}` to user PATH...")

    # Update the PATH for the current session.
    os.environ["PATH"] += ";" + msvc_bin_path

    try:
        import winreg
        # Open the user environment variables key.
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment", 0, winreg.KEY_ALL_ACCESS) as key:
            try:
                current_path, reg_type = winreg.QueryValueEx(key, "Path")
            except FileNotFoundError:
                current_path = ""
                reg_type = winreg.REG_EXPAND_SZ

            if msvc_bin_path not in current_path:
                new_path = current_path + (";" if current_path else "") + msvc_bin_path
                winreg.SetValueEx(key, "Path", 0, reg_type, new_path)
                print("Successfully added `cl.exe` to the user PATH! (A log off/on is required for new processes.)")
            else:
                print("MSVC bin path is already present in the user PATH.")
    except Exception as e:
        print(f"Failed to update user PATH: {e}")


def install_vs_build_tools():
    """
    Download and install Visual Studio Build Tools with MSVC and the Windows SDK silently.
    If the MSVC compiler is already installed, ensure its directory is added to the PATH.
    """
    if is_msvc_installed():
        print("MSVC Compiler is already installed.")
        #add_msvc_to_user_path()
        return

    if find_latest_msvc_bin():
        print("MSVC Compiler installed but not on PATH")
        add_msvc_to_user_path()
        return

    print("Downloading Visual Studio Build Tools...")
    urllib.request.urlretrieve(VS_BUILD_TOOLS_URL, VS_INSTALLER)

    print("Installing Visual Studio Build Tools (this may take a few minutes)...")
    try:
        subprocess.run([
            VS_INSTALLER,
            "--quiet", "--wait",
            "--add", "Microsoft.VisualStudio.Workload.VCTools",               # C++ Build Tools Workload
            "--add", "Microsoft.VisualStudio.Component.VC.CoreBuildTools",    # MSVC Compiler Core
            "--add", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",     # 64-bit toolset
            "--add", "Microsoft.VisualStudio.Component.Windows10SDK.19041",   # Windows SDK 10
            "--add", "Microsoft.VisualStudio.Component.VC.Redist.14.Latest",  # Standard Library Headers
            "--add", "Microsoft.VisualStudio.Component.VC.Redist.x64",        # Force x64 Redistributable installation
            "--add", "Microsoft.VisualStudio.Component.CMake",                # CMake (for some builds)
            "--add", "Microsoft.VisualStudio.Component.VC.ATL",               # ATL/MFC (sometimes required)
        ], check=True)
        # After installation, update the PATH.
        add_msvc_to_user_path()

    finally:
        # Clean up the installer file.
        if os.path.exists(VS_INSTALLER):
            os.remove(VS_INSTALLER)
            print("Cleaned up the VS Build Tools installer.")
