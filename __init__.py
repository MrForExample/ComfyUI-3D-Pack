import os
import sys
import folder_paths as comfy_paths

ROOT_PATH = os.path.join(comfy_paths.get_folder_paths("custom_nodes")[0], "ComfyUI-3D-Pack")
sys.path.append(ROOT_PATH)

import shutil
import __main__
import importlib
import inspect
from .webserver.server import server

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

nodes_filename = "nodes"
module = importlib.import_module(f".{nodes_filename}", package=__name__)
for name, cls in inspect.getmembers(module, inspect.isclass):
    if cls.__module__ == module.__name__:
        name = name.replace("_", " ")

        node = f"[Comfy3D] {name}"
        disp = f"{name}"

        NODE_CLASS_MAPPINGS[node] = cls
        NODE_DISPLAY_NAME_MAPPINGS[node] = disp
        
WEB_DIRECTORY = "./web"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

# # Cleanup old extension folder
folder_web = os.path.join(os.path.dirname(os.path.realpath(__main__.__file__)), "web")
extensions_folder = os.path.join(folder_web, 'extensions', 'ComfyUI-3D-Pack')

def cleanup():
    if os.path.exists(extensions_folder):
        shutil.rmtree(extensions_folder)
        print('\033[34mComfy3D: \033[92mRemoved old extension folder\033[0m')

cleanup()