import os
from os import listdir
from os.path import isfile, join
import sys
from datetime import datetime

class cstr(str):
    # Modified from: WAS Node Suite
    class color:
        END = '\33[0m'
        BOLD = '\33[1m'
        ITALIC = '\33[3m'
        UNDERLINE = '\33[4m'
        BLINK = '\33[5m'
        BLINK2 = '\33[6m'
        SELECTED = '\33[7m'

        BLACK = '\33[30m'
        RED = '\33[31m'
        GREEN = '\33[32m'
        YELLOW = '\33[33m'
        BLUE = '\33[34m'
        VIOLET = '\33[35m'
        BEIGE = '\33[36m'
        WHITE = '\33[37m'

        BLACKBG = '\33[40m'
        REDBG = '\33[41m'
        GREENBG = '\33[42m'
        YELLOWBG = '\33[43m'
        BLUEBG = '\33[44m'
        VIOLETBG = '\33[45m'
        BEIGEBG = '\33[46m'
        WHITEBG = '\33[47m'

        GREY = '\33[90m'
        LIGHTRED = '\33[91m'
        LIGHTGREEN = '\33[92m'
        LIGHTYELLOW = '\33[93m'
        LIGHTBLUE = '\33[94m'
        LIGHTVIOLET = '\33[95m'
        LIGHTBEIGE = '\33[96m'
        LIGHTWHITE = '\33[97m'

        GREYBG = '\33[100m'
        LIGHTREDBG = '\33[101m'
        LIGHTGREENBG = '\33[102m'
        LIGHTYELLOWBG = '\33[103m'
        LIGHTBLUEBG = '\33[104m'
        LIGHTVIOLETBG = '\33[105m'
        LIGHTBEIGEBG = '\33[106m'
        LIGHTWHITEBG = '\33[107m'

        @staticmethod
        def add_code(name, code):
            if not hasattr(cstr.color, name.upper()):
                setattr(cstr.color, name.upper(), code)
            else:
                raise ValueError(f"'cstr' object already contains a code with the name '{name}'.")

    def __new__(cls, text):
        return super().__new__(cls, text)

    def __getattr__(self, attr):
        if attr.lower().startswith("_cstr"):
            code = getattr(self.color, attr.upper().lstrip("_cstr"))
            modified_text = self.replace(f"__{attr[1:]}__", f"{code}")
            return cstr(modified_text)
        elif attr.upper() in dir(self.color):
            code = getattr(self.color, attr.upper())
            modified_text = f"{code}{self}{self.color.END}"
            return cstr(modified_text)
        elif attr.lower() in dir(cstr):
            return getattr(cstr, attr.lower())
        else:
            raise AttributeError(f"'cstr' object has no attribute '{attr}'")

    def print(self, **kwargs):
        print(self, **kwargs)

#! MESSAGE TEMPLATES
cstr.color.add_code("msg", f"{cstr.color.BLUE}[Comfy3D] {cstr.color.END}")
cstr.color.add_code("warning", f"{cstr.color.LIGHTYELLOW}[Comfy3D] [WARNING] {cstr.color.END}")
cstr.color.add_code("error", f"{cstr.color.RED}[Comfy3D] [ERROR] {cstr.color.END}")

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
        all_date_format = ["%Y", "%m", "%d", "%M", "%S", "%f"]
        for date_format in all_date_format:
            if date_format in filename:
                filename = filename.replace(date_format, now.strftime(date_format))
                
        save_path = join(folder_path, filename) + file_extension
        cstr(f"[{class_name}] Saving model to {save_path}").msg.print()
        return save_path
    else:
        cstr(f"[{class_name}] File name {filename} does not end with supported file extensions: {supported_extensions}").error.print()
    
    return None

# Get all files in a folder, if extension_filter is provided then only reture the files with extensions in extension_filter
def get_list_filenames(directory, extension_filter=None):
    return [f for f in listdir(directory) if isfile(join(directory, f)) and (extension_filter is None or f.lower().endswith(extension_filter))]