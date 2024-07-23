import os
from os import listdir
from os.path import isfile, join, exists, dirname
import sys
from datetime import datetime
from shared_utils.log_utils import cstr

def get_parent_dirpath_n_level_up(abs_path, n=1):
    for i in range(n):
        abs_path = dirname(abs_path)
    return abs_path

def get_persistent_directory(folder_name):
    if sys.platform == "win32":
        folder = join(os.path.expanduser("~"), "AppData", "Local", folder_name)
    else:
        folder = join(os.path.expanduser("~"), "." + folder_name)
    
    os.makedirs(folder, exist_ok=True)
    return folder

def parse_save_filename(save_path, output_directory, supported_extensions, class_name):
    
    folder_path, filename = os.path.split(save_path)
    filename, file_extension = os.path.splitext(filename)
    if file_extension.lower() in supported_extensions:
        if not os.path.isabs(save_path):
            folder_path = join(output_directory, folder_path)
        
        os.makedirs(folder_path, exist_ok=True)
        
        # replace time date format to current time
        now = datetime.now() # current date and time
        all_date_format = ["%Y", "%m", "%d", "%H", "%M", "%S", "%f"]
        for date_format in all_date_format:
            if date_format in filename:
                filename = filename.replace(date_format, now.strftime(date_format))
                
        save_path = join(folder_path, filename) + file_extension
        cstr(f"[{class_name}] Saving model to {save_path}").msg.print()
        return save_path
    else:
        cstr(f"[{class_name}] File name {filename} does not end with supported file extensions: {supported_extensions}").error.print()
    
    return None

def get_list_filenames(directory, extension_filter=None, recursive=False):
    """
    Recursively finds files with specified extensions in a directory and returns relative paths.

    Args:
        directory (str): The directory path to search.
        extension_filter (list): List of file extensions (e.g., ['.txt', '.csv']).

    Returns:
        list: List of relative file paths matching the specified extensions.
    """
    if exists(directory):
        if recursive:
            result = []
            for root, _, files in os.walk(directory):
                for item in files:
                    if extension_filter is None or os.path.splitext(item)[1].lower() in extension_filter:
                        relative_path = os.path.relpath(os.path.join(root, item), directory)
                        result.append(relative_path)
            return result
        else:
            return [f for f in listdir(directory) if isfile(join(directory, f)) and (extension_filter is None or f.lower().endswith(extension_filter))]
    else:
        return []
    
# Download pre-trained model if it not exist locally
def resume_or_download_model_from_hf(checkpoints_dir_abs, repo_id, model_name, class_name="", repo_type="model"):
    
    ckpt_path = os.path.join(checkpoints_dir_abs, model_name)
    if not os.path.isfile(ckpt_path):
        cstr(f"[{class_name}] can't find checkpoint {ckpt_path}, will download it from repo {repo_id} instead").warning.print()
        
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=repo_id, local_dir=checkpoints_dir_abs, filename=model_name, repo_type=repo_type)

    return ckpt_path
