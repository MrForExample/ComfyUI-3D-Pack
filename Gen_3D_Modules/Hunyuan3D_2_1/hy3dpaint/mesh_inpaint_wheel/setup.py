from setuptools import setup, find_packages
import platform

if platform.system() == "Windows":
    plat_name = "win_amd64"
elif platform.system() == "Darwin":
    plat_name = "macosx_10_9_x86_64"
else:
    plat_name = "linux_x86_64"

setup(
    name="mesh_inpaint_processor",
    version="1.0.0",
    description="Fast C++ mesh inpainting processor with Python fallback",
    long_description="Compiled module for fast mesh inpainting processing",
    author="Hunyuan3D Team",
    packages=find_packages(),
    package_data={
        'mesh_inpaint_processor': [
            '*.pyd',  
            '*.so',   # For Linux version
            '*.py'    # Python fallback files
        ]
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
    zip_safe=False,
    has_ext_modules=lambda: True,  
) 