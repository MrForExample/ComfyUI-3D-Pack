import os
import cv2
from typing import NamedTuple, Optional
import torch
from torch import Tensor
import trimesh
import numpy as np
from PIL import Image

from kiui.op import safe_normalize, dot
from kiui.typing import *

from .io_gltf import load_gltf_or_glb, get_binary_data, save_glb, GltfDocument
from .mesh_ops import get_all_meshes_triangles
from .accessors import access_data, append_accessor_and_bufferview
from .mesh import Mesh

class FastMesh:
    """
    Fast implementation of mesh based on FastGLB without dependency on trimesh.
    Uses direct loading of GLB/GLTF through mesh_processor.
    """
    
    def __init__(
        self,
        v: Optional[Tensor] = None,
        f: Optional[Tensor] = None,
        vn: Optional[Tensor] = None,
        fn: Optional[Tensor] = None,
        vt: Optional[Tensor] = None,
        ft: Optional[Tensor] = None,
        vc: Optional[Tensor] = None,
        albedo: Optional[Tensor] = None,
        metallicRoughness: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize FastMesh with the same parameters as Mesh"""
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.v = v
        self.vn = vn
        self.vt = vt
        self.f = f
        self.fn = fn
        self.ft = ft
        self.vc = vc
        self.albedo = albedo
        self.metallicRoughness = metallicRoughness
        self.ori_center = 0
        self.ori_scale = 1
        self._lazy_texture_doc = None  
    
    @classmethod
    def load(cls, path, resize=True, renormal=True, retex=False, clean=False, bound=0.5, front_dir='+z', lazy_texture=True, load_pbr=True, **kwargs):
        """Loading mesh through FastGLB approach"""
        
        device = kwargs.get('device')
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if path.endswith(".obj"):
            mesh = cls._load_obj_fast(path, load_pbr=load_pbr, **{**kwargs, 'device': device})
        elif path.endswith((".glb", ".gltf")):
            mesh = cls._load_gltf_fast(path, lazy_texture=lazy_texture, load_pbr=load_pbr, **{**kwargs, 'device': device})
        else:
            standard_mesh = Mesh.load(path, resize=False, renormal=False, retex=False, clean=False, **kwargs)
            mesh = cls._from_standard_mesh(standard_mesh)
        
        if mesh is None:
            return None
        
        if clean:
            mesh._clean_mesh()
        
        
        if resize:
            mesh.auto_size(bound=bound)
        
        if renormal or mesh.vn is None:
            mesh.auto_normal()
        
        if retex or (mesh.albedo is not None and mesh.vt is None):
            mesh.auto_uv(cache_path=path)
        
        if front_dir != "+z":
            mesh._apply_front_dir_rotation(front_dir)
        
        print(f"[FastMesh] Loading completed successfully!")
        
        # Show final texture status
        has_albedo = mesh.albedo is not None
        has_pbr = mesh.metallicRoughness is not None
        
        return mesh
    
    @classmethod
    def _load_gltf_fast(cls, path, device=None, lazy_texture=True, load_pbr=True, **kwargs):
        """Fast loading of GLB/GLTF through mesh_processor"""
        
        try:
            
            # Load GLTF document
            doc = load_gltf_or_glb(path)
            
            # Extract geometry
            vertices, faces, mesh_groups = get_all_meshes_triangles(doc, transform_to_global=True)
            
            if len(vertices) == 0 or len(faces) == 0:
                return None
            
            mesh = cls(device=device)
            
            # Convert to torch tensors in parallel
            mesh.v = torch.from_numpy(vertices.astype(np.float32)).to(device)
            mesh.f = torch.from_numpy(faces.astype(np.int32)).to(device)
            
            # Extract attributes and textures in parallel
            uvs, normals, colors = cls._extract_all_mesh_attributes(doc, mesh_groups, device)
            if load_pbr:
                cls._extract_gltf_textures_fast(mesh, doc, device, lazy_empty=lazy_texture)
            else:
                mesh._create_empty_albedo()
            
            # Set attributes (avoid redundant assignments)
            if uvs is not None:
                mesh.vt, mesh.ft = uvs, mesh.f
            if normals is not None:
                mesh.vn, mesh.fn = normals, mesh.f
            if colors is not None:
                mesh.vc = colors
            
            return mesh
            
        except Exception as e:
            print(f"[FastMesh] Error loading GLB/GLTF {path}: {e}")
            return None
    
    @classmethod
    def _extract_all_mesh_attributes(cls, doc, mesh_groups, device):
        """Extract UV, normals, colors from all meshes in GLTF"""
        
        all_uvs = []
        all_normals = []
        all_colors = []
        
        for mesh_idx, gltf_mesh in enumerate(doc.meshes()):
            for primitive_idx, primitive in enumerate(gltf_mesh.get("primitives", [])):
                attributes = primitive.get("attributes", {})
                
                # Find corresponding mesh group
                mesh_group = None
                for group in mesh_groups:
                    # Try both possible key names
                    group_mesh_idx = group.get('mesh_idx', group.get('mesh_index'))
                    group_primitive_idx = group.get('primitive_idx', group.get('primitive_index'))
                    
                    if group_mesh_idx == mesh_idx and group_primitive_idx == primitive_idx:
                        mesh_group = group
                        break
                
                if mesh_group is None:
                    continue
                    
                # Extract vertex info from mesh_group (handle different key names)
                vertices_range = mesh_group.get('vertices_range', (0, 0))
                vertex_offset = vertices_range[0] if isinstance(vertices_range, tuple) else mesh_group.get('vertex_offset', 0)
                vertex_count = vertices_range[1] - vertices_range[0] if isinstance(vertices_range, tuple) else mesh_group.get('vertex_count', 0)
                
                # Extract UV coordinates
                if "TEXCOORD_0" in attributes:
                    try:
                        uv_accessor_idx = int(attributes["TEXCOORD_0"])
                        uv_data = access_data(doc, uv_accessor_idx).astype(np.float32)
                        all_uvs.append(uv_data)
                    except Exception:
                        all_uvs.append(None)
                else:
                    all_uvs.append(None)
                
                # Extract normals
                if "NORMAL" in attributes:
                    try:
                        normal_accessor_idx = int(attributes["NORMAL"])
                        normal_data = access_data(doc, normal_accessor_idx).astype(np.float32)
                        all_normals.append(normal_data)
                    except Exception:
                        all_normals.append(None)
                else:
                    all_normals.append(None)
                
                # Extract vertex colors
                if "COLOR_0" in attributes:
                    try:
                        color_accessor_idx = int(attributes["COLOR_0"])
                        color_data = access_data(doc, color_accessor_idx).astype(np.float32)
                        if color_data.shape[1] > 3:
                            color_data = color_data[:, :3]
                        all_colors.append(color_data)
                    except Exception:
                        all_colors.append(None)
                else:
                    all_colors.append(None)
        
        # Concatenate all data
        final_uvs = None
        final_normals = None  
        final_colors = None
        
        # Optimized concatenation - only process non-None arrays
        if any(uv is not None for uv in all_uvs):
            valid_uvs = [uv for uv in all_uvs if uv is not None]
            if valid_uvs:
                if len(valid_uvs) == 1:
                    final_uvs = torch.from_numpy(valid_uvs[0]).to(device)
                else:
                    final_uvs = torch.from_numpy(np.concatenate(valid_uvs, axis=0)).to(device)
        
        if any(normal is not None for normal in all_normals):
            valid_normals = [n for n in all_normals if n is not None]
            if valid_normals:
                if len(valid_normals) == 1:
                    final_normals = torch.from_numpy(valid_normals[0]).to(device)
                else:
                    final_normals = torch.from_numpy(np.concatenate(valid_normals, axis=0)).to(device)
        
        if any(color is not None for color in all_colors):
            valid_colors = [c for c in all_colors if c is not None]
            if valid_colors:
                if len(valid_colors) == 1:
                    final_colors = torch.from_numpy(valid_colors[0]).to(device)
                else:
                    final_colors = torch.from_numpy(np.concatenate(valid_colors, axis=0)).to(device)
        
        return final_uvs, final_normals, final_colors

    @classmethod
    def _extract_gltf_attributes(cls, mesh, doc, device):
        """Extract attributes from GLTF (UV, normals, colors)"""
        
        if not doc.meshes():
            return
        
        first_mesh = doc.meshes()[0]
        for primitive in first_mesh.get("primitives", []):
            attributes = primitive.get("attributes", {})
            
            # UV coordinates
            if "TEXCOORD_0" in attributes and mesh.vt is None:
                try:
                    uv_accessor_idx = int(attributes["TEXCOORD_0"])
                    uv_data = access_data(doc, uv_accessor_idx).astype(np.float32)
                    uv_data[:, 1] = 1.0 - uv_data[:, 1]  # Flip V
                    mesh.vt = torch.tensor(uv_data, dtype=torch.float32, device=device)
                    mesh.ft = mesh.f
                except Exception as e:
                    print(f"[FastMesh] Error UV: {e}")
            
            # Normals
            if "NORMAL" in attributes and mesh.vn is None:
                try:
                    normal_accessor_idx = int(attributes["NORMAL"])
                    normal_data = access_data(doc, normal_accessor_idx).astype(np.float32)
                    mesh.vn = torch.tensor(normal_data, dtype=torch.float32, device=device)
                    mesh.fn = mesh.f
                    print(f"[FastMesh] Normals: {mesh.vn.shape}")
                except Exception as e:
                    print(f"[FastMesh] Error normals: {e}")
            
            # Vertex colors
            if "COLOR_0" in attributes and mesh.vc is None:
                try:
                    color_accessor_idx = int(attributes["COLOR_0"])
                    color_data = access_data(doc, color_accessor_idx).astype(np.float32)
                    if color_data.shape[1] > 3:
                        color_data = color_data[:, :3]
                    mesh.vc = torch.tensor(color_data, dtype=torch.float32, device=device)
                except Exception as e:
                    print(f"[FastMesh] Error colors: {e}")
    
    @classmethod
    def _extract_gltf_textures_fast(cls, mesh, doc, device, lazy_empty=True):
        """Fast texture extraction"""
        
        if not (doc.materials() and doc.textures() and doc.images()):
            if not lazy_empty:
                mesh._create_empty_albedo()
            return
        
        try:
            material = doc.materials()[0]
            pbr = material.get("pbrMetallicRoughness", {})
            
            albedo_texture = None
            mr_texture = None
            
            # Albedo texture
            base_color_texture = pbr.get("baseColorTexture")
            if base_color_texture:
                albedo_texture = cls._extract_texture_by_index(doc, base_color_texture.get("index", 0))
            
            # Metallic-Roughness texture
            mr_texture_info = pbr.get("metallicRoughnessTexture")
            if mr_texture_info:
                mr_texture = cls._extract_texture_by_index(doc, mr_texture_info.get("index", 0))
            
            # Convert to tensors
            if albedo_texture is not None:
                mesh.albedo = torch.from_numpy(albedo_texture.astype(np.float32) / 255.0).to(device).contiguous()
            
            if mr_texture is not None:
                mesh.metallicRoughness = torch.from_numpy(mr_texture.astype(np.float32) / 255.0).to(device).contiguous()
            
        except Exception as e:
            print(f"[FastMesh] Error extracting textures: {e}")
            if not lazy_empty:
                mesh._create_empty_albedo()

    @classmethod
    def _extract_gltf_textures(cls, mesh, doc, device):
        """Extract textures from GLTF"""
        
        should_create_empty = True
        
        try:
            if not (doc.materials() and doc.textures() and doc.images()):
                mesh._create_empty_albedo()
                return
            
            material = doc.materials()[0]
            pbr = material.get("pbrMetallicRoughness", {})
            
            # Albedo texture
            base_color_texture = pbr.get("baseColorTexture")
            if base_color_texture:
                albedo_texture = cls._extract_texture_by_index(doc, base_color_texture.get("index", 0))
                if albedo_texture is not None:
                    albedo_float = albedo_texture.astype(np.float32) / 255.0
                    mesh.albedo = torch.tensor(albedo_float, dtype=torch.float32, device=device).contiguous()
                    should_create_empty = False
            
            # Metallic-Roughness texture
            mr_texture_info = pbr.get("metallicRoughnessTexture")
            if mr_texture_info and not should_create_empty:
                mr_texture = cls._extract_texture_by_index(doc, mr_texture_info.get("index", 0))
                if mr_texture is not None:
                    mr_float = mr_texture.astype(np.float32) / 255.0
                    mesh.metallicRoughness = torch.tensor(mr_float, dtype=torch.float32, device=device).contiguous()
            
        except Exception as e:
            print(f"[FastMesh] Error extracting textures: {e}")
        
        if should_create_empty:
            mesh._create_empty_albedo()
    
    @classmethod
    def _extract_texture_by_index(cls, doc, texture_idx):
        """Extract texture by index from GLTF"""
        
        try:
            if texture_idx >= len(doc.textures()):
                return None
            
            texture = doc.textures()[texture_idx]
            image_idx = texture.get("source", 0)
            
            if image_idx >= len(doc.images()):
                return None
            
            image_info = doc.images()[image_idx]
            buffer_view_idx = image_info.get("bufferView")
            
            if buffer_view_idx is None:
                return None
            
            buffer_view = doc.bufferViews()[buffer_view_idx]
            buffer_idx = buffer_view.get("buffer", 0)
            byte_offset = buffer_view.get("byteOffset", 0)
            byte_length = buffer_view.get("byteLength", 0)
            
            binary_data = get_binary_data(doc, buffer_idx)
            image_bytes = binary_data[byte_offset:byte_offset + byte_length]
            
            # Decode image
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            texture_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if texture_image is not None:
                texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)
                return texture_image
            
        except Exception as e:
            print(f"[FastMesh] Error extracting texture {texture_idx}: {e}")
        
        return None
    
    @classmethod
    def _load_obj_fast(cls, path, device=None, load_pbr=True, **kwargs):
        """Fast loading of OBJ using direct parsing"""
        try:
            return cls._load_obj_direct(path, device=device, load_pbr=load_pbr, **kwargs)
        except Exception as e:
            standard_mesh = Mesh.load_obj(path, device=device, **kwargs)
            return cls._from_standard_mesh(standard_mesh)
    
    @classmethod
    def _load_obj_direct(cls, path, device=None, load_pbr=True, **kwargs):
        """Direct OBJ parsing without trimesh dependency"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Parse OBJ file
        vertices, normals, uvs, faces = [], [], [], []
        mtl_path = None
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                    
                prefix = parts[0]
                
                if prefix == 'v':
                    vertices.append([float(parts[i]) for i in (1, 2, 3)])
                elif prefix == 'vn':
                    normals.append([float(parts[i]) for i in (1, 2, 3)])
                elif prefix == 'vt':
                    u, v = float(parts[1]), float(parts[2])
                    uvs.append([u, 1.0 - v])  # Flip V for consistency
                elif prefix == 'f':
                    # Parse face indices
                    face_verts = []
                    for vert_str in parts[1:]:
                        indices = vert_str.split('/')
                        vi = int(indices[0]) - 1 if indices[0] else 0
                        ti = int(indices[1]) - 1 if len(indices) > 1 and indices[1] else 0
                        ni = int(indices[2]) - 1 if len(indices) > 2 and indices[2] else 0
                        face_verts.append((vi, ti, ni))
                    
                    # Triangulate face (fan triangulation)
                    for i in range(1, len(face_verts) - 1):
                        faces.extend([face_verts[0], face_verts[i], face_verts[i + 1]])
                elif prefix == 'mtllib':
                    mtl_path = os.path.join(os.path.dirname(path), parts[1])
        
        if not vertices or not faces:
            raise ValueError("No valid geometry found in OBJ file")
        
        # Deduplicate vertices and build final arrays
        pos, normal, uv, indices = cls._deduplicate_obj_vertices(
            vertices, normals, uvs, faces
        )
        
        # Create FastMesh
        mesh = cls(device=device)
        mesh.v = torch.tensor(pos, dtype=torch.float32, device=device)
        mesh.f = torch.tensor(indices, dtype=torch.int32, device=device).reshape(-1, 3)
        
        if normal:
            mesh.vn = torch.tensor(normal, dtype=torch.float32, device=device)
            mesh.fn = mesh.f
        
        if uv:
            mesh.vt = torch.tensor(uv, dtype=torch.float32, device=device)
            mesh.ft = mesh.f
        
        # Load textures from MTL
        if mtl_path and os.path.exists(mtl_path):
            cls._load_mtl_textures(mesh, mtl_path, device, load_pbr=load_pbr)
        else:
            # Create empty texture
            mesh._create_empty_albedo()
        
        return mesh
    
    @classmethod
    def _deduplicate_obj_vertices(cls, vertices, normals, uvs, faces):
        """Deduplicate vertices based on position/normal/uv combination"""
        pos, normal, uv = [], [], []
        vertex_map = {}
        indices = []
        
        for vi, ti, ni in faces:
            # Create unique key for this vertex combination
            key = (vi, ti if ti < len(uvs) else -1, ni if ni < len(normals) else -1)
            
            if key not in vertex_map:
                vertex_map[key] = len(pos)
                
                # Add vertex position
                pos.append(vertices[vi])
                
                # Add UV coordinates
                if ti < len(uvs):
                    uv.append(uvs[ti])
                else:
                    uv.append([0.0, 0.0])
                
                # Add normal
                if ni < len(normals):
                    normal.append(normals[ni])
                else:
                    normal.append([0.0, 0.0, 1.0])
            
            indices.append(vertex_map[key])
        
        return pos, normal, uv, indices
    
    @classmethod
    def _load_mtl_textures(cls, mesh, mtl_path, device, load_pbr=True):
        """Load textures from MTL file"""
        textures = {}
        base_color = [1.0, 1.0, 1.0]
        
        try:
            with open(mtl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if not parts:
                        continue
                    
                    if parts[0] == 'Kd':
                        base_color = [float(parts[i]) for i in (1, 2, 3)]
                    elif parts[0] == 'map_Kd':
                        textures['albedo'] = os.path.join(os.path.dirname(mtl_path), parts[1])
                    elif parts[0] == 'map_Pm':
                        textures['metallic'] = os.path.join(os.path.dirname(mtl_path), parts[1])
                    elif parts[0] == 'map_Pr':
                        textures['roughness'] = os.path.join(os.path.dirname(mtl_path), parts[1])
            
            # Load albedo texture
            if 'albedo' in textures and os.path.exists(textures['albedo']):
                albedo_img = cv2.imread(textures['albedo'], cv2.IMREAD_COLOR)
                if albedo_img is not None:
                    albedo_img = cv2.cvtColor(albedo_img, cv2.COLOR_BGR2RGB)
                    albedo_tensor = torch.from_numpy(albedo_img.astype(np.float32) / 255.0).to(device)
                    mesh.albedo = albedo_tensor.contiguous()
            else:
                # Create texture with base color
                texture = np.ones((1024, 1024, 3), dtype=np.float32) * np.array(base_color)
                mesh.albedo = torch.tensor(texture, dtype=torch.float32, device=device)
            
            # Load and combine metallic/roughness textures (only if PBR requested)
            if load_pbr and 'metallic' in textures and 'roughness' in textures:
                if os.path.exists(textures['metallic']) and os.path.exists(textures['roughness']):
                    mr_texture = cls._combine_metallic_roughness(
                        textures['metallic'], textures['roughness']
                    )
                    mesh.metallicRoughness = torch.from_numpy(mr_texture.astype(np.float32) / 255.0).to(device).contiguous()
            elif not load_pbr and ('metallic' in textures or 'roughness' in textures):
                print(f"[FastMesh] PBR disabled - skipping metallic/roughness textures")
        
        except Exception as e:
            print(f"[FastMesh] Error loading MTL textures: {e}")
            mesh._create_empty_albedo()
    
    @classmethod
    def _from_standard_mesh(cls, standard_mesh):
        """Convert standard Mesh to FastMesh"""
        if standard_mesh is None:
            return None
        
        return cls(
            v=standard_mesh.v,
            f=standard_mesh.f,
            vn=standard_mesh.vn,
            fn=standard_mesh.fn,
            vt=standard_mesh.vt,
            ft=standard_mesh.ft,
            vc=standard_mesh.vc,
            albedo=standard_mesh.albedo,
            metallicRoughness=standard_mesh.metallicRoughness,
            device=standard_mesh.device
        )
    
    def _create_empty_albedo(self):
        """Create empty albedo texture without PBR materials"""
        texture = np.ones((1024, 1024, 3), dtype=np.float32) * np.array([0.5, 0.5, 0.5])
        self.albedo = torch.tensor(texture, dtype=torch.float32, device=self.device)
        self.metallicRoughness = None  # Clear PBR materials
    
    def _clean_mesh(self):
        """Clean mesh (simplified version)"""
        try:
            from kiui.mesh_utils import clean_mesh
            vertices = self.v.detach().cpu().numpy()
            triangles = self.f.detach().cpu().numpy()
            vertices, triangles = clean_mesh(vertices, triangles, remesh=False)
            self.v = torch.from_numpy(vertices).contiguous().float().to(self.device)
            self.f = torch.from_numpy(triangles).contiguous().int().to(self.device)
        except Exception as e:
            print(f"[FastMesh] Error cleaning: {e}")
    
    def _apply_front_dir_rotation(self, front_dir):
        """Apply rotation for aligning front_dir with +z"""
        # Use the same logic as in standard Mesh
        if front_dir == "+z":
            return
        
        # Define transformation matrix
        if "-z" in front_dir:
            T = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]], device=self.device, dtype=torch.float32)
        elif "+x" in front_dir:
            T = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], device=self.device, dtype=torch.float32)
        elif "-x" in front_dir:
            T = torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]], device=self.device, dtype=torch.float32)
        elif "+y" in front_dir:
            T = torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]], device=self.device, dtype=torch.float32)
        elif "-y" in front_dir:
            T = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], device=self.device, dtype=torch.float32)
        else:
            T = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], device=self.device, dtype=torch.float32)
        
        # Additional rotations
        if '1' in front_dir:
            T @= torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], device=self.device, dtype=torch.float32) 
        elif '2' in front_dir:
            T @= torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]], device=self.device, dtype=torch.float32) 
        elif '3' in front_dir:
            T @= torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], device=self.device, dtype=torch.float32) 
        
        # Apply transformation
        if self.v is not None:
            self.v @= T
        if self.vn is not None:
            self.vn @= T
    
    # Use methods from standard Mesh
    def aabb(self):
        """AABB mesh"""
        return torch.min(self.v, dim=0).values, torch.max(self.v, dim=0).values
    
    @torch.no_grad()
    def auto_size(self, bound=0.9):
        """Automatic size change"""
        vmin, vmax = self.aabb()
        self.ori_center = (vmax + vmin) / 2
        self.ori_scale = 2 * bound / torch.max(vmax - vmin).item()
        self.v = (self.v - self.ori_center) * self.ori_scale
    
    def auto_normal(self):
        """Automatic calculation of normals"""
        i0, i1, i2 = self.f[:, 0].long(), self.f[:, 1].long(), self.f[:, 2].long()
        v0, v1, v2 = self.v[i0, :], self.v[i1, :], self.v[i2, :]
        
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        
        vn = torch.zeros_like(self.v)
        vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)
        
        vn = torch.where(
            dot(vn, vn) > 1e-20,
            vn,
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device),
        )
        vn = safe_normalize(vn)
        
        self.vn = vn
        self.fn = self.f
    
    def auto_uv(self, cache_path=None, vmap=True):
        """Automatic calculation of UV (use method from standard Mesh)"""
        # Temporarily delegate to standard Mesh
        temp_mesh = Mesh(
            v=self.v, f=self.f, vn=self.vn, fn=self.fn,
            vt=self.vt, ft=self.ft, vc=self.vc,
            albedo=self.albedo, metallicRoughness=self.metallicRoughness,
            device=self.device
        )
        temp_mesh.auto_uv(cache_path=cache_path, vmap=vmap)
        
        # Copy result back
        self.vt = temp_mesh.vt
        self.ft = temp_mesh.ft
        if vmap:
            self.v = temp_mesh.v
            self.f = temp_mesh.f
            self.vn = temp_mesh.vn
            self.fn = temp_mesh.fn
            if temp_mesh.vc is not None:
                self.vc = temp_mesh.vc
    
    def to(self, device):
        """Move to another device"""
        self.device = device
        for name in ["v", "f", "vn", "fn", "vt", "ft", "albedo", "vc", "metallicRoughness"]:
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.to(device))
        return self
    
    def write(self, path):
        """Fast saving using mesh_processor for GLB/GLTF files"""
        if path.endswith(".glb"):
            self._write_glb_fast(path)
        elif path.endswith(".gltf"):
            # For GLTF, fallback to standard method (less common)
            self._write_fallback(path)
        elif path.endswith(".ply"):
            self._write_ply_fast(path)
        elif path.endswith(".obj"):
            self._write_obj_fast(path)
        else:
            # Fallback for unknown formats
            self._write_fallback(path)
    
    def _write_glb_fast(self, path):
        """Fast GLB writing using mesh_processor"""
        try:
            # Check if we need to align vertices to UV coordinates
            if self.vt is not None and self.v.shape[0] != self.vt.shape[0]:
                self._align_vertices_to_uv()
            
            # Prepare data
            vertices = self.v.detach().cpu().numpy().astype(np.float32)
            faces = self.f.detach().cpu().numpy().astype(np.uint32)
            
            # Create minimal GLTF structure
            gltf_json = {
                "asset": {"version": "2.0", "generator": "FastMesh"},
                "scenes": [{"nodes": [0]}],
                "nodes": [{"mesh": 0}],
                "meshes": [{
                    "primitives": [{
                        "mode": 4  # TRIANGLES
                    }]
                }]
            }
            
            # Create document with empty buffer
            doc = GltfDocument(json_data=gltf_json, binary_buffers={0: np.array([], dtype=np.uint8)})
            
            # Add vertex data using helper function
            pos_accessor, _ = append_accessor_and_bufferview(
                doc, vertices,
                component_type=5126,  # FLOAT
                element_type="VEC3",
                target=34962  # ARRAY_BUFFER
            )
            
            # Add face data using helper function
            faces_flat = faces.flatten()
            face_component_type = 5123 if vertices.shape[0] <= 65535 else 5125  # UNSIGNED_SHORT or UNSIGNED_INT
            face_dtype = np.uint16 if face_component_type == 5123 else np.uint32
            
            idx_accessor, _ = append_accessor_and_bufferview(
                doc, faces_flat.astype(face_dtype),
                component_type=face_component_type,
                element_type="SCALAR",
                target=34963  # ELEMENT_ARRAY_BUFFER
            )
            
            # Update primitive with accessor indices
            primitive = doc.json()["meshes"][0]["primitives"][0]
            primitive["attributes"] = {"POSITION": pos_accessor}
            primitive["indices"] = idx_accessor
            
            # Add UV coordinates if available
            if self.vt is not None:
                uv_data = self.vt.detach().cpu().numpy().astype(np.float32)
                uv_accessor, _ = append_accessor_and_bufferview(
                    doc, uv_data,
                    component_type=5126,  # FLOAT
                    element_type="VEC2",
                    target=34962  # ARRAY_BUFFER
                )
                primitive["attributes"]["TEXCOORD_0"] = uv_accessor
            
            # Add texture if available
            if self.albedo is not None and self.vt is not None:
                self._add_texture_to_gltf(doc, primitive)
            
            # Save to file
            with open(path, "wb") as f:
                save_glb(doc, f)
                
            print(f"[FastMesh] Fast GLB saved: {path}")
            
        except Exception as e:
            print(f"[FastMesh] Fast GLB save failed: {e}, using fallback")
            self._write_fallback(path)
    
    def _add_texture_to_gltf(self, doc, primitive):
        """Add texture data to GLTF document using fast approach"""
        try:
            # Convert albedo texture to PNG bytes
            albedo_np = self.albedo.detach().cpu().numpy()
            # Ensure values are in [0, 1] range
            albedo_np = np.clip(albedo_np, 0.0, 1.0)
            albedo_uint8 = (albedo_np * 255).astype(np.uint8)
            
            # Encode as PNG
            import cv2
            albedo_bgr = cv2.cvtColor(albedo_uint8, cv2.COLOR_RGB2BGR)
            success, png_bytes = cv2.imencode('.png', albedo_bgr)
            if not success:
                print(f"[FastMesh] Failed to encode texture as PNG")
                return
            
            png_data = png_bytes.tobytes()
            
            # Add image data to buffer using helper function
            image_accessor, _ = append_accessor_and_bufferview(
                doc, np.frombuffer(png_data, dtype=np.uint8),
                component_type=5121,  # UNSIGNED_BYTE
                element_type="SCALAR",
                target=None  # No specific target for image data
            )
            
            # Create GLTF structures
            gltf_json = doc.json()
            
            # Initialize arrays if they don't exist
            if "materials" not in gltf_json:
                gltf_json["materials"] = []
            if "textures" not in gltf_json:
                gltf_json["textures"] = []
            if "images" not in gltf_json:
                gltf_json["images"] = []
            if "samplers" not in gltf_json:
                gltf_json["samplers"] = []
            
            # Add image
            image_idx = len(gltf_json["images"])
            gltf_json["images"].append({
                "bufferView": image_accessor,
                "mimeType": "image/png"
            })
            
            # Add sampler
            sampler_idx = len(gltf_json["samplers"])
            gltf_json["samplers"].append({
                "magFilter": 9729,  # LINEAR
                "minFilter": 9987,  # LINEAR_MIPMAP_LINEAR
                "wrapS": 10497,     # REPEAT
                "wrapT": 10497      # REPEAT
            })
            
            # Add texture
            texture_idx = len(gltf_json["textures"])
            gltf_json["textures"].append({
                "sampler": sampler_idx,
                "source": image_idx
            })
            
            # Add material
            material_idx = len(gltf_json["materials"])
            material = {
                "pbrMetallicRoughness": {
                    "baseColorTexture": {
                        "index": texture_idx,
                        "texCoord": 0
                    },
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0
                },
                "alphaMode": "OPAQUE",
                "doubleSided": True
            }
            
            # Add metallic-roughness texture if available
            if self.metallicRoughness is not None:
                mr_np = self.metallicRoughness.detach().cpu().numpy()
                mr_uint8 = (mr_np * 255).astype(np.uint8)
                mr_bgr = cv2.cvtColor(mr_uint8, cv2.COLOR_RGB2BGR)
                mr_success, mr_png_bytes = cv2.imencode('.png', mr_bgr)
                
                if mr_success:
                    mr_png_data = mr_png_bytes.tobytes()
                    mr_image_accessor, _ = append_accessor_and_bufferview(
                        doc, np.frombuffer(mr_png_data, dtype=np.uint8),
                        component_type=5121,  # UNSIGNED_BYTE
                        element_type="SCALAR",
                        target=None
                    )
                    
                    # Add MR image and texture
                    mr_image_idx = len(gltf_json["images"])
                    gltf_json["images"].append({
                        "bufferView": mr_image_accessor,
                        "mimeType": "image/png"
                    })
                    
                    mr_texture_idx = len(gltf_json["textures"])
                    gltf_json["textures"].append({
                        "sampler": sampler_idx,  # Reuse same sampler
                        "source": mr_image_idx
                    })
                    
                    # Update material
                    material["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {
                        "index": mr_texture_idx,
                        "texCoord": 0
                    }
                    material["pbrMetallicRoughness"]["metallicFactor"] = 1.0
                    material["pbrMetallicRoughness"]["roughnessFactor"] = 1.0
                    
            
            gltf_json["materials"].append(material)
            
            # Link material to primitive
            primitive["material"] = material_idx
            
        except Exception as e:
            print(f"[FastMesh] Error adding texture to GLTF: {e}")
    
    def _align_vertices_to_uv(self):
        """Align vertices to UV coordinates (like align_v_to_vt in standard Mesh)"""
        try:
            # Use the standard Mesh method for alignment
            temp_mesh = Mesh(
                v=self.v, f=self.f, vn=self.vn, fn=self.fn,
                vt=self.vt, ft=self.ft, vc=self.vc,
                albedo=self.albedo, metallicRoughness=self.metallicRoughness,
                device=self.device
            )
            
            # Apply alignment
            temp_mesh.align_v_to_vt()
            
            # Copy back aligned data
            self.v = temp_mesh.v
            self.f = temp_mesh.f
            self.vn = temp_mesh.vn
            self.fn = temp_mesh.fn
            if temp_mesh.vc is not None:
                self.vc = temp_mesh.vc
                
            
        except Exception as e:
            print(f"[FastMesh] Error during vertex-UV alignment: {e}")
    
    def _write_ply_fast(self, path):
        """Fast PLY writing"""
        vertices = self.v.detach().cpu().numpy()
        faces = self.f.detach().cpu().numpy()
        
        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            for v in vertices:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")
            
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    
    def _write_obj_fast(self, path):
        """Fast OBJ writing with MTL support"""
        vertices = self.v.detach().cpu().numpy()
        faces = self.f.detach().cpu().numpy()
        
        # Prepare file paths
        mtl_path = path.replace(".obj", ".mtl")
        albedo_path = path.replace(".obj", "_albedo.png")
        metallic_path = path.replace(".obj", "_metallic.png")
        roughness_path = path.replace(".obj", "_roughness.png")
        
        # Get UV and normal data if available
        uvs = self.vt.detach().cpu().numpy() if self.vt is not None else None
        normals = self.vn.detach().cpu().numpy() if self.vn is not None else None
        uv_faces = self.ft.detach().cpu().numpy() if self.ft is not None else None
        normal_faces = self.fn.detach().cpu().numpy() if self.fn is not None else None
        
        # Write OBJ file
        with open(path, 'w') as f:
            if self.albedo is not None:
                f.write(f"mtllib {os.path.basename(mtl_path)}\n")
            
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            
            # Write UV coordinates
            if uvs is not None:
                for uv in uvs:
                    f.write(f"vt {uv[0]} {1.0 - uv[1]}\n")  # Flip V back for OBJ
            
            # Write normals
            if normals is not None:
                for n in normals:
                    f.write(f"vn {n[0]} {n[1]} {n[2]}\n")
            
            # Write material usage
            if self.albedo is not None:
                f.write("usemtl defaultMat\n")
            
            # Write faces
            for i, face in enumerate(faces):
                face_str = "f"
                for j in range(3):
                    v_idx = face[j] + 1
                    uv_idx = uv_faces[i][j] + 1 if uv_faces is not None else ""
                    n_idx = normal_faces[i][j] + 1 if normal_faces is not None else ""
                    
                    if uvs is not None and normals is not None:
                        face_str += f" {v_idx}/{uv_idx}/{n_idx}"
                    elif uvs is not None:
                        face_str += f" {v_idx}/{uv_idx}"
                    elif normals is not None:
                        face_str += f" {v_idx}//{n_idx}"
                    else:
                        face_str += f" {v_idx}"
                f.write(face_str + "\n")
        
        # Write MTL file if we have textures
        if self.albedo is not None:
            self._write_mtl_file(mtl_path, albedo_path, metallic_path, roughness_path)
            
            # Save texture files
            self._save_texture_files(albedo_path, metallic_path, roughness_path)
    
    def _write_mtl_file(self, mtl_path, albedo_path, metallic_path, roughness_path):
        """Write MTL file for OBJ"""
        with open(mtl_path, 'w') as f:
            f.write("newmtl defaultMat\n")
            f.write("Ka 1 1 1\n")
            f.write("Kd 1 1 1\n") 
            f.write("Ks 1 1 1\n")
            f.write("illum 1\n")
            f.write("Ns 10\n")
            
            if self.albedo is not None:
                f.write(f"map_Kd {os.path.basename(albedo_path)}\n")
            
            if self.metallicRoughness is not None:
                f.write(f"map_Pm {os.path.basename(metallic_path)}\n")
                f.write(f"map_Pr {os.path.basename(roughness_path)}\n")
    
    def _save_texture_files(self, albedo_path, metallic_path, roughness_path):
        """Save texture files for OBJ"""
        try:
            # Save albedo texture
            if self.albedo is not None:
                albedo_np = self.albedo.detach().cpu().numpy()
                albedo_uint8 = (np.clip(albedo_np, 0, 1) * 255).astype(np.uint8)
                albedo_bgr = cv2.cvtColor(albedo_uint8, cv2.COLOR_RGB2BGR)
                cv2.imwrite(albedo_path, albedo_bgr)
            
            # Save metallic/roughness textures
            if self.metallicRoughness is not None:
                mr_np = self.metallicRoughness.detach().cpu().numpy()
                mr_uint8 = (np.clip(mr_np, 0, 1) * 255).astype(np.uint8)
                
                # Extract metallic (B channel) and roughness (G channel)
                metallic_uint8 = mr_uint8[:, :, 2]  # B channel
                roughness_uint8 = mr_uint8[:, :, 1]  # G channel
                
                cv2.imwrite(metallic_path, metallic_uint8)
                cv2.imwrite(roughness_path, roughness_uint8)
                
        except Exception as e:
            print(f"[FastMesh] Error saving texture files: {e}")
    
    def _write_fallback(self, path):
        """Fallback to standard Mesh write method"""
        temp_mesh = Mesh(
            v=self.v, f=self.f, vn=self.vn, fn=self.fn,
            vt=self.vt, ft=self.ft, vc=self.vc,
            albedo=self.albedo, metallicRoughness=self.metallicRoughness,
            device=self.device
        )
        temp_mesh.ori_center = self.ori_center
        temp_mesh.ori_scale = self.ori_scale
        return temp_mesh.write(path)
    
    @classmethod
    def create_glb_with_pbr_materials(cls, obj_path, textures_dict, output_path, device=None):
        """Create GLB with PBR materials from separate texture files"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            mesh = cls.load(obj_path, **{'device': device})
            if mesh is None:
                raise ValueError(f"Failed to load mesh from {obj_path}")
            
            # Combine metallic and roughness if both provided
            if "metallic" in textures_dict and "roughness" in textures_dict:
                mr_texture = cls._combine_metallic_roughness(
                    textures_dict["metallic"], 
                    textures_dict["roughness"]
                )
                mesh.metallicRoughness = torch.from_numpy(mr_texture.astype(np.float32) / 255.0).to(device).contiguous()
            
            # Load albedo texture
            if "albedo" in textures_dict and os.path.exists(textures_dict["albedo"]):
                albedo_img = np.array(Image.open(textures_dict["albedo"]).convert("RGB"))
                mesh.albedo = torch.from_numpy(albedo_img.astype(np.float32) / 255.0).to(device).contiguous()
            
            # Generate UV if not present
            if mesh.vt is None:
                mesh.auto_uv()
            
            # Write GLB
            mesh.write(output_path)
            print(f"PBR GLB file saved: {output_path}")
            
        except Exception as e:
            print(f"Error creating GLB with PBR materials: {e}")
            raise e
    
    @staticmethod
    def _combine_metallic_roughness(metallic_path, roughness_path):
        """Combine metallic and roughness maps into single texture"""
        metallic_img = Image.open(metallic_path).convert("L")
        roughness_img = Image.open(roughness_path).convert("L")
        
        if metallic_img.size != roughness_img.size:
            roughness_img = roughness_img.resize(metallic_img.size)
        
        width, height = metallic_img.size
        metallic_array = np.array(metallic_img)
        roughness_array = np.array(roughness_img)
        
        # Create combined array (R, G, B) = (AO, Roughness, Metallic)
        combined_array = np.zeros((height, width, 3), dtype=np.uint8)
        combined_array[:, :, 0] = 255  # R: AO (white if no AO map)
        combined_array[:, :, 1] = roughness_array  # G: Roughness
        combined_array[:, :, 2] = metallic_array  # B: Metallic
        
        return combined_array
    
    @classmethod
    def obj_to_glb_converter(cls, obj_path, output_path, device=None):
        """Convert OBJ to GLB using FastMesh direct parsing"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            
            # Load OBJ using direct parsing
            mesh = cls._load_obj_direct(obj_path, device=device)
            if mesh is None:
                raise ValueError(f"Failed to load OBJ file: {obj_path}")
            
            # Generate UV if not present
            if mesh.vt is None and mesh.albedo is not None:
                mesh.auto_uv()
            
            # Write as GLB
            mesh.write(output_path)
            return True
            
        except Exception as e:
            print(f"[FastMesh] OBJ to GLB conversion failed: {e}")
            return False
    
    @classmethod  
    def read_obj_structure(cls, path):
        """Read OBJ file structure like your original code"""
        vertices, normals, uvs, faces = [], [], [], []
        mtl_path = None
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                    
                prefix = parts[0]
                
                if prefix == 'v':
                    vertices.append([float(parts[i]) for i in (1, 2, 3)])
                elif prefix == 'vn':
                    normals.append([float(parts[i]) for i in (1, 2, 3)])
                elif prefix == 'vt':
                    u, v = float(parts[1]), float(parts[2])
                    uvs.append([u, 1.0 - v])  # Flip V for glTF
                elif prefix == 'f':
                    face_indices = []
                    for vert_str in parts[1:]:
                        indices = vert_str.split('/')
                        vi = int(indices[0]) - 1 if indices[0] else 0
                        ti = int(indices[1]) - 1 if len(indices) > 1 and indices[1] else 0
                        ni = int(indices[2]) - 1 if len(indices) > 2 and indices[2] else 0
                        face_indices.append((vi, ti, ni))
                    
                    # Triangulate (fan triangulation)
                    for i in range(1, len(face_indices) - 1):
                        faces.extend([face_indices[0], face_indices[i], face_indices[i + 1]])
                elif prefix == 'mtllib':
                    mtl_path = os.path.join(os.path.dirname(path), parts[1])
        
        # Parse MTL file
        textures = {}
        base_color = [1.0, 1.0, 1.0]
        
        if mtl_path and os.path.exists(mtl_path):
            try:
                with open(mtl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        
                        parts = line.split()
                        if not parts:
                            continue
                        
                        if parts[0] == 'Kd':
                            base_color = [float(parts[i]) for i in (1, 2, 3)]
                        elif parts[0] == 'map_Kd':
                            textures['baseColor'] = os.path.join(os.path.dirname(mtl_path), parts[1])
                        elif parts[0] == 'map_Pm':
                            textures['metallic'] = os.path.join(os.path.dirname(mtl_path), parts[1])
                        elif parts[0] == 'map_Pr':
                            textures['roughness'] = os.path.join(os.path.dirname(mtl_path), parts[1])
                        elif parts[0] == 'map_Bump':
                            if '-bm' in parts:
                                idx = parts.index('-bm')
                                textures['normal'] = os.path.join(os.path.dirname(mtl_path), parts[idx+2])
                            else:
                                textures['normal'] = os.path.join(os.path.dirname(mtl_path), parts[1])
            except Exception as e:
                print(f"[FastMesh] Error parsing MTL: {e}")
        
        # Deduplicate vertices
        pos, normal, uv, indices, original_vi = cls._deduplicate_obj_vertices_with_mapping(
            vertices, normals, uvs, faces
        )
        
        return pos, normal, uv, indices, original_vi, textures, base_color
    
    @classmethod
    def _deduplicate_obj_vertices_with_mapping(cls, vertices, normals, uvs, faces):
        """Deduplicate vertices and keep original vertex mapping"""
        pos, normal, uv = [], [], []
        vertex_map = {}
        indices = []
        original_vi = []  # Maps new vertex index to original vertex index
        
        for vi, ti, ni in faces:
            # Create unique key for this vertex combination
            key = (vi, ti if ti < len(uvs) else -1, ni if ni < len(normals) else -1)
            
            if key not in vertex_map:
                vertex_map[key] = len(pos)
                
                # Add vertex position
                pos.append(vertices[vi])
                original_vi.append(vi)  # Track original vertex index
                
                # Add UV coordinates
                if ti < len(uvs):
                    uv.append(uvs[ti])
                else:
                    uv.append([0.0, 0.0])
                
                # Add normal
                if ni < len(normals):
                    normal.append(normals[ni])
                else:
                    normal.append([0.0, 0.0, 1.0])
            
            indices.append(vertex_map[key])
        
        return pos, normal, uv, indices, original_vi