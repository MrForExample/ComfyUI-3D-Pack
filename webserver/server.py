import server
import os

from ..shared_utils.log_utils import cstr

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


web_conf = None

def set_web_conf(new_web_conf):
    global web_conf
    web_conf = new_web_conf

@server.PromptServer.instance.routes.get("/viewfile")
async def view_file(request):
    query = request.rel_url.query
    # Security check to see if query client is local
    if request.remote in web_conf['clients_ip'] and "filepath" in query:
        filepath = query["filepath"]
        
        cstr(f"[Server Query view_file] Get file {filepath}").msg.print()
        
        if filepath.lower().endswith(SUPPORTED_VIEW_EXTENSIONS) and os.path.exists(filepath):
            return web.FileResponse(filepath)
    
    return web.Response(status=404)
