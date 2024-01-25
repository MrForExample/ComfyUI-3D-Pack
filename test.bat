set "python_miniconda_exec=..\..\..\..\..\..\..\python_miniconda_env\ComfyUI\python.exe"

cd tgs/models/snowflake/pointnet2_ops_lib && "%python_miniconda_exec%" -s setup.py install && cd ../../../../