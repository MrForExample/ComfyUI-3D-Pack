import os
import cv2
from typing import NamedTuple, Optional
import torch
from torch import Tensor
import trimesh
import numpy as np

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
    def load(cls, path, resize=True, renormal=True, retex=False, clean=False, bound=0.5, front_dir='+z', lazy_texture=True, **kwargs):
        """Loading mesh through FastGLB approach"""
        
        device = kwargs.get('device')
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if path.endswith(".obj"):
            mesh = cls._load_obj_fast(path, device=device, **kwargs)
        elif path.endswith((".glb", ".gltf")):
            mesh = cls._load_gltf_fast(path, device=device, lazy_texture=lazy_texture, **kwargs)
        else:
            standard_mesh = Mesh.load(path, resize=False, renormal=False, retex=False, clean=False, **kwargs)
            mesh = cls._from_standard_mesh(standard_mesh)
        
        if mesh is None:
            return None
        
        if clean:
            mesh._clean_mesh()
        
        print(f"[FastMesh loading] v: {mesh.v.shape}, f: {mesh.f.shape}")
        
        if resize:
            mesh.auto_size(bound=bound)
        
        if renormal or mesh.vn is None:
            mesh.auto_normal()
            print(f"[FastMesh loading] vn: {mesh.vn.shape}, fn: {mesh.fn.shape}")
        
        print(f"[FastMesh] UV check: retex={retex}, albedo={mesh.albedo is not None}, vt={mesh.vt is not None}")
        if retex or (mesh.albedo is not None and mesh.vt is None):
            print(f"[FastMesh] Starting UV generation for {mesh.v.shape[0]} vertices...")
            mesh.auto_uv(cache_path=path)
            print(f"[FastMesh loading] vt: {mesh.vt.shape}, ft: {mesh.ft.shape}")
        
        print(f"[FastMesh] Checking front_dir: {front_dir}")
        if front_dir != "+z":
            print(f"[FastMesh] Applying front_dir rotation...")
            mesh._apply_front_dir_rotation(front_dir)
        
        print(f"[FastMesh] Loading completed successfully!")
        
        return mesh
    
    @classmethod
    def _load_gltf_fast(cls, path, device=None, lazy_texture=True, **kwargs):
        """Fast loading of GLB/GLTF through mesh_processor"""
        
        try:
            print(f"[FastMesh] Loading GLB/GLTF: {path}")
            
            # Load GLTF document
            
            doc = load_gltf_or_glb(path)
            print(f"[FastMesh] GLTF document: {len(doc.meshes())} meshes")
            
            # Extract geometry
            vertices, faces, mesh_groups = get_all_meshes_triangles(doc, transform_to_global=True)
            
            if len(vertices) == 0 or len(faces) == 0:
                print(f"[FastMesh] No geometry in {path}")
                return None
            
            mesh = cls(device=device)
            mesh.v = torch.tensor(vertices, dtype=torch.float32, device=device)
            mesh.f = torch.tensor(faces, dtype=torch.int32, device=device)
            
            # Extract UV, normals, colors from all meshes
            uvs, normals, colors = cls._extract_all_mesh_attributes(doc, mesh_groups, device)
            if uvs is not None:
                mesh.vt = uvs
                mesh.ft = mesh.f  # Same topology
                print(f"[FastMesh] UV: {mesh.vt.shape}")
            if normals is not None:
                mesh.vn = normals
                mesh.fn = mesh.f
                print(f"[FastMesh] Normals: {mesh.vn.shape}")
            if colors is not None:
                mesh.vc = colors
                print(f"[FastMesh] Colors: {mesh.vc.shape}")
            if not lazy_texture:
                cls._extract_gltf_textures(mesh, doc, device)
            else:
                mesh._lazy_texture_doc = doc
            
            print(f"[FastMesh] Loaded: {len(vertices)} vertices, {len(faces)} faces")
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
                    if group.get('mesh_idx') == mesh_idx and group.get('primitive_idx') == primitive_idx:
                        mesh_group = group
                        break
                
                if mesh_group is None:
                    continue
                    
                vertex_offset = mesh_group['vertex_offset']
                vertex_count = mesh_group['vertex_count']
                
                # Extract UV coordinates
                if "TEXCOORD_0" in attributes:
                    try:
                        uv_accessor_idx = int(attributes["TEXCOORD_0"])
                        uv_data = access_data(doc, uv_accessor_idx).astype(np.float32)
                        uv_data[:, 1] = 1.0 - uv_data[:, 1]  # Flip V
                        all_uvs.append(uv_data)
                        print(f"[FastMesh] Mesh {mesh_idx}.{primitive_idx} UV: {uv_data.shape}")
                    except Exception as e:
                        print(f"[FastMesh] Error extracting UV from mesh {mesh_idx}.{primitive_idx}: {e}")
                        all_uvs.append(None)
                else:
                    all_uvs.append(None)
                
                # Extract normals
                if "NORMAL" in attributes:
                    try:
                        normal_accessor_idx = int(attributes["NORMAL"])
                        normal_data = access_data(doc, normal_accessor_idx).astype(np.float32)
                        all_normals.append(normal_data)
                        print(f"[FastMesh] Mesh {mesh_idx}.{primitive_idx} Normals: {normal_data.shape}")
                    except Exception as e:
                        print(f"[FastMesh] Error extracting normals from mesh {mesh_idx}.{primitive_idx}: {e}")
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
                        print(f"[FastMesh] Mesh {mesh_idx}.{primitive_idx} Colors: {color_data.shape}")
                    except Exception as e:
                        print(f"[FastMesh] Error extracting colors from mesh {mesh_idx}.{primitive_idx}: {e}")
                        all_colors.append(None)
                else:
                    all_colors.append(None)
        
        # Concatenate all data
        final_uvs = None
        final_normals = None  
        final_colors = None
        
        if any(uv is not None for uv in all_uvs):
            # Pad missing UV data with zeros
            padded_uvs = []
            for i, uv in enumerate(all_uvs):
                if uv is not None:
                    padded_uvs.append(uv)
                else:
                    # Create dummy UV coordinates
                    vertex_count = mesh_groups[i]['vertex_count'] if i < len(mesh_groups) else 1000
                    dummy_uv = np.zeros((vertex_count, 2), dtype=np.float32)
                    padded_uvs.append(dummy_uv)
            
            if padded_uvs:
                final_uvs = torch.tensor(np.concatenate(padded_uvs, axis=0), dtype=torch.float32, device=device)
        
        if any(normal is not None for normal in all_normals):
            valid_normals = [n for n in all_normals if n is not None]
            if valid_normals:
                final_normals = torch.tensor(np.concatenate(valid_normals, axis=0), dtype=torch.float32, device=device)
        
        if any(color is not None for color in all_colors):
            valid_colors = [c for c in all_colors if c is not None]
            if valid_colors:
                final_colors = torch.tensor(np.concatenate(valid_colors, axis=0), dtype=torch.float32, device=device)
        
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
                    print(f"[FastMesh] UV: {mesh.vt.shape}")
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
                    print(f"[FastMesh] Colors: {mesh.vc.shape}")
                except Exception as e:
                    print(f"[FastMesh] Error colors: {e}")
    
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
                    print(f"[FastMesh] Albedo: {mesh.albedo.shape}")
            
            # Metallic-Roughness texture
            mr_texture_info = pbr.get("metallicRoughnessTexture")
            if mr_texture_info and not should_create_empty:
                mr_texture = cls._extract_texture_by_index(doc, mr_texture_info.get("index", 0))
                if mr_texture is not None:
                    mr_float = mr_texture.astype(np.float32) / 255.0
                    mesh.metallicRoughness = torch.tensor(mr_float, dtype=torch.float32, device=device).contiguous()
                    print(f"[FastMesh] MetallicRoughness: {mesh.metallicRoughness.shape}")
            
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
    def _load_obj_fast(cls, path, device=None, **kwargs):
        """Fast loading of OBJ (fallback to standard method)"""
        standard_mesh = Mesh.load_obj(path, device=device, **kwargs)
        return cls._from_standard_mesh(standard_mesh)
    
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
        """Create empty albedo texture"""
        texture = np.ones((1024, 1024, 3), dtype=np.float32) * np.array([0.5, 0.5, 0.5])
        self.albedo = torch.tensor(texture, dtype=torch.float32, device=self.device)
        print(f"[FastMesh] Empty texture: {self.albedo.shape}")
    
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
            
            # Save to file
            with open(path, "wb") as f:
                save_glb(doc, f)
                
            print(f"[FastMesh] Fast GLB saved: {path}")
            
        except Exception as e:
            print(f"[FastMesh] Fast GLB save failed: {e}, using fallback")
            self._write_fallback(path)
    
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
        """Fast OBJ writing"""
        vertices = self.v.detach().cpu().numpy()
        faces = self.f.detach().cpu().numpy()
        
        with open(path, 'w') as f:
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
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