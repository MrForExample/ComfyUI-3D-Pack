class Preview_3DMesh:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_file_path": ("STRING", {"default": '', "multiline": False}),
            },
        }
    
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "preview_mesh"
    CATEGORY = "Comfy3D/Visualize"
    
    def preview_mesh(self, mesh_file_path):
        
        mesh_folder_path, filename = os.path.split(mesh_file_path)
        
        if not os.path.isabs(mesh_file_path):
            mesh_file_path = os.path.join(comfy_paths.output_directory, mesh_folder_path, filename)
        
        if not filename.lower().endswith(SUPPORTED_3D_EXTENSIONS):
            cstr(f"[{self.__class__.__name__}] File name {filename} does not end with supported 3D file extensions: {SUPPORTED_3D_EXTENSIONS}").error.print()
            mesh_file_path = ""
        
        print(f"[Preview_3DMesh] Final mesh path: {mesh_file_path}")
        print(f"[Preview_3DMesh] File exists: {os.path.exists(mesh_file_path) if mesh_file_path else False}")
        
        previews = [
            {
                "filepath": mesh_file_path,
            }
        ]
        return {"ui": {"previews": previews}, "result": ()}

class Load_3D_Mesh:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_file_path": ("STRING", {"default": '', "multiline": False}),
                "use_fastmesh": ("BOOLEAN", {"default": True, "tooltip": "Использовать FastMesh для GLB/GLTF файлов"}),
                "resize":  ("BOOLEAN", {"default": False},),
                "renormal":  ("BOOLEAN", {"default": True},),
                "retex":  ("BOOLEAN", {"default": False},),
                "optimizable": ("BOOLEAN", {"default": False},),
                "clean": ("BOOLEAN", {"default": False},),
                "resize_bound": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1000.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = (
        "MESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "load_mesh"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_mesh(self, mesh_file_path, use_fastmesh, resize, renormal, retex, optimizable, clean, resize_bound):
        import time
        mesh = None
        
        if not os.path.isabs(mesh_file_path):
            mesh_file_path = os.path.join(comfy_paths.input_directory, mesh_file_path)
        
        if os.path.exists(mesh_file_path):
            folder, filename = os.path.split(mesh_file_path)
            if filename.lower().endswith(SUPPORTED_3D_EXTENSIONS):
                with torch.inference_mode(not optimizable):
                    start_time = time.time()
                    if use_fastmesh and mesh_file_path.endswith((".glb", ".gltf")):
                        mesh = FastMesh.load(mesh_file_path, resize, renormal, retex, clean, resize_bound)
                        print(f"[FastMesh] Loading completed in {time.time() - start_time:.3f}s")
                    else:
                        mesh = Mesh.load(mesh_file_path, resize, renormal, retex, clean, resize_bound)
                        print(f"[Mesh] Loading completed in {time.time() - start_time:.3f}s")
            else:
                cstr(f"[{self.__class__.__name__}] File name {filename} does not end with supported 3D file extensions: {SUPPORTED_3D_EXTENSIONS}").error.print()
        else:        
            cstr(f"[{self.__class__.__name__}] File {mesh_file_path} does not exist").error.print()
        return (mesh, )

class Save_3D_Mesh:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "save_path": ("STRING", {"default": 'Mesh_%Y-%m-%d-%M-%S-%f.glb', "multiline": False}),
                "use_fastmesh": ("BOOLEAN", {"default": True, "tooltip": "Использовать FastMesh для быстрого сохранения GLB/PLY/OBJ"}),
            },
        }

    OUTPUT_NODE = True
    RETURN_TYPES = (
        "STRING",
    )
    RETURN_NAMES = (
        "save_path",
    )
    FUNCTION = "save_mesh"
    CATEGORY = "Comfy3D/Import|Export"
    
    def save_mesh(self, mesh, save_path, use_fastmesh):
        import time
        save_path = parse_save_filename(save_path, comfy_paths.output_directory, SUPPORTED_3D_EXTENSIONS, self.__class__.__name__)
        
        if save_path is not None:
            start_time = time.time()
            
            # Используем FastMesh методы для быстрого сохранения
            if use_fastmesh and save_path.endswith((".glb", ".ply", ".obj")):
                # Если это уже FastMesh - используем напрямую
                print(f"[Save_3D_Mesh] FastMesh detected")
                if hasattr(mesh, '_write_glb_fast'):
                    mesh.write(save_path)
                    save_time = time.time() - start_time
                    print(f"[FastMesh] Saving completed in {save_time:.3f}s")
                else:
                    # Если обычный Mesh - создаем временный FastMesh для быстрых методов
                    print(f"[Save_3D_Mesh] REGULAR Mesh detected")
                    fast_mesh = FastMesh(
                        v=mesh.v, f=mesh.f, vn=mesh.vn, fn=mesh.fn,
                        vt=mesh.vt, ft=mesh.ft, vc=mesh.vc,
                        albedo=mesh.albedo, metallicRoughness=getattr(mesh, 'metallicRoughness', None),
                        device=mesh.device
                    )
                    fast_mesh.ori_center = getattr(mesh, 'ori_center', 0)
                    fast_mesh.ori_scale = getattr(mesh, 'ori_scale', 1)
                    fast_mesh.write(save_path)
                    save_time = time.time() - start_time
                    print(f"[FastMesh] Saving completed in {save_time:.3f}s")
            else:
                # Обычное сохранение через стандартный метод
                mesh.write(save_path)
                save_time = time.time() - start_time
                print(f"[Mesh] Saving completed in {save_time:.3f}s")

        return (save_path, )


class Preview_3D_FastMesh:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_file_path": ("STRING", {"default": '', "multiline": False}),
            },
        }
    
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "preview_mesh"
    CATEGORY = "Comfy3D/Visualize"
    
    def preview_mesh(self, mesh_file_path):
        
        mesh_folder_path, filename = os.path.split(mesh_file_path)
        
        if not os.path.isabs(mesh_file_path):
            mesh_file_path = os.path.join(comfy_paths.output_directory, mesh_folder_path, filename)
        
        if not filename.lower().endswith(SUPPORTED_3D_EXTENSIONS):
            cstr(f"[{self.__class__.__name__}] File name {filename} does not end with supported 3D file extensions: {SUPPORTED_3D_EXTENSIONS}").error.print()
            mesh_file_path = ""
        
        print(f"[Preview_3D_FastMesh] Final mesh path: {mesh_file_path}")
        print(f"[Preview_3D_FastMesh] File exists: {os.path.exists(mesh_file_path) if mesh_file_path else False}")
        
        previews = [
            {
                "filepath": mesh_file_path,
            }
        ]
        return {"ui": {"previews": previews}, "result": ()}


class Load_3D_FastMesh:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_file_path": ("STRING", {"default": '', "multiline": False}),
                "use_fastmesh": ("BOOLEAN", {"default": True, "tooltip": "Использовать FastMesh загрузчик"}),
                "resize":  ("BOOLEAN", {"default": False},),
                "renormal":  ("BOOLEAN", {"default": True},),
                "retex":  ("BOOLEAN", {"default": False},),
                "optimizable": ("BOOLEAN", {"default": False},),
                "clean": ("BOOLEAN", {"default": False},),
                "resize_bound": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1000.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = (
        "MESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "load_mesh"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_mesh(self, mesh_file_path, use_fastmesh, resize, renormal, retex, optimizable, clean, resize_bound):
        import time
        mesh = None
        
        if not os.path.isabs(mesh_file_path):
            mesh_file_path = os.path.join(comfy_paths.input_directory, mesh_file_path)
        
        if os.path.exists(mesh_file_path):
            folder, filename = os.path.split(mesh_file_path)
            if filename.lower().endswith(SUPPORTED_3D_EXTENSIONS):
                with torch.inference_mode(not optimizable):
                    start_time = time.time()
                    if use_fastmesh and mesh_file_path.endswith((".glb", ".gltf")):
                        from mesh_processor.mesh import FastMesh
                        mesh = FastMesh.load(mesh_file_path, resize, renormal, retex, clean, resize_bound)
                        print(f"[FastMesh] Loading completed in {time.time() - start_time:.3f}s")
                    else:
                        mesh = Mesh.load(mesh_file_path, resize, renormal, retex, clean, resize_bound)
                        mesh_type = "FastMesh" if use_fastmesh else "Mesh"
                        print(f"[{mesh_type}] Loading completed in {time.time() - start_time:.3f}s")
            else:
                cstr(f"[{self.__class__.__name__}] File name {filename} does not end with supported 3D file extensions: {SUPPORTED_3D_EXTENSIONS}").error.print()
        else:        
            cstr(f"[{self.__class__.__name__}] File {mesh_file_path} does not exist").error.print()
        return (mesh, )


class Save_3D_FastMesh:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "save_path": ("STRING", {"default": 'FastMesh_%Y-%m-%d-%M-%S-%f.glb', "multiline": False}),
            },
        }

    OUTPUT_NODE = True
    RETURN_TYPES = (
        "STRING",
    )
    RETURN_NAMES = (
        "save_path",
    )
    FUNCTION = "save_mesh"
    CATEGORY = "Comfy3D/Import|Export"
    
    def save_mesh(self, mesh, save_path):
        import time
        save_path = parse_save_filename(save_path, comfy_paths.output_directory, SUPPORTED_3D_EXTENSIONS, self.__class__.__name__)
        
        if save_path is not None:
            start_time = time.time()
            mesh.write(save_path)
            save_time = time.time() - start_time
            mesh_type = "FastMesh" if hasattr(mesh, '_create_empty_albedo') else "Mesh"
            print(f"[{mesh_type}] Saving completed in {save_time:.3f}s")

        return (save_path, )