import logging
import sys

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

class WarningFilter(logging.Filter):
    def filter(self, record):
        if record.levelno == logging.WARNING:
            record.msg = f"Warn!: {record.msg}"
        return True

def create_handler(stream, levels, formatter):
    handler = logging.StreamHandler(stream)
    handler.setLevel(min(levels))
    handler.addFilter(lambda record: record.levelno in levels)
    handler.addFilter(WarningFilter())  # Apply the custom filter
    handler.setFormatter(formatter)
    
    return handler
def setup_logger(logger_name, level, stdout_levels, stderr_levels, formatter):
    logger = logging.getLogger(logger_name)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.setLevel(level)
    stdout_handler = create_handler(sys.stdout, stdout_levels, formatter)
    stderr_handler = create_handler(sys.stderr, stderr_levels, formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)