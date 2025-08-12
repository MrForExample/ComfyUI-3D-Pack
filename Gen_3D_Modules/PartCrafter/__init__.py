import os
import sys

# Add current directory to sys.path so partcrafter_src modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir) 