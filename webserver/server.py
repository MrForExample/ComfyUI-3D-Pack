import server
import folder_paths as comfy_paths
import os
import time
import subprocess

from ..shared_utils.common_utils import cstr

web = server.web

SUPPORTED_VIEW_EXTENSIONS = (
    '.png',
    '.jpg',
    '.jpeg ',
    '.mtl',
    '.obj',
    '.glb',
    '.ply',
    '.splat'
)

@server.PromptServer.instance.routes.get("/extensions/ComfyUI-3D-Pack/html/viewfile")
async def view_file(request):
    query = request.rel_url.query
    # Security check to see if query client is local
    if request.remote == "127.0.0.1" and "filepath" in query:
        filepath = query["filepath"]
        
        cstr(f"[Server Query view_file] Get file {filepath}").msg.print()
        
        if filepath.lower().endswith(SUPPORTED_VIEW_EXTENSIONS) and os.path.exists(filepath):
            return web.FileResponse(filepath)
    
    return web.Response(status=404)
