# Try to import C++ version first, fallback to Python version
try:
    from .mesh_inpaint_processor import meshVerticeInpaint
    print("[INFO] Using compiled C++ mesh_inpaint_processor (fast)")
except ImportError:
    try:
        from .mesh_inpaint_processor_fallback import meshVerticeInpaint
        print("[WARNING] Using Python fallback mesh_inpaint_processor (slower)")
        print("[INFO] To use faster C++ version, run: compile_mesh_painter.bat")
    except ImportError:
        print("[ERROR] Neither C++ nor Python mesh_inpaint_processor found!")
