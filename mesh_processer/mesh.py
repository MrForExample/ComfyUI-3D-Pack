import os
import cv2
from typing import NamedTuple
import torch
import trimesh
import numpy as np

from kiui.op import safe_normalize, dot
from kiui.typing import *

from shared_utils.sh_utils import SH2RGB
from shared_utils.image_utils import prepare_torch_img


class Mesh:
    """
    A torch-native trimesh class, with support for ``ply/obj/glb`` formats.

    Note:
        This class only supports one mesh with a single texture image (an albedo texture and a metallic-roughness texture).
    """
    def __init__(
        self,
        v: Optional[Tensor] = None,
        f: Optional[Tensor] = None,
        vn: Optional[Tensor] = None,
        fn: Optional[Tensor] = None,
        vt: Optional[Tensor] = None,
        ft: Optional[Tensor] = None,
        vc: Optional[Tensor] = None, # vertex color
        albedo: Optional[Tensor] = None,
        metallicRoughness: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        """Init a mesh directly using all attributes.

        Args:
            v (Optional[Tensor]): vertices, float [N, 3]. Defaults to None.
            f (Optional[Tensor]): faces, int [M, 3]. Defaults to None.
            vn (Optional[Tensor]): vertex normals, float [N, 3]. Defaults to None.
            fn (Optional[Tensor]): faces for normals, int [M, 3]. Defaults to None.
            vt (Optional[Tensor]): vertex uv coordinates, float [N, 2]. Defaults to None.
            ft (Optional[Tensor]): faces for uvs, int [M, 3]. Defaults to None.
            vc (Optional[Tensor]): vertex colors, float [N, 3]. Defaults to None.
            albedo (Optional[Tensor]): albedo texture, float [H, W, 3], RGB format. Defaults to None.
            metallicRoughness (Optional[Tensor]): metallic-roughness texture, float [H, W, 3], metallic(Blue) = metallicRoughness[..., 2], roughness(Green) = metallicRoughness[..., 1]. Defaults to None.
            device (Optional[torch.device]): torch device. Defaults to None.
        """
        self.device = device
        self.v = v
        self.vn = vn
        self.vt = vt
        self.f = f
        self.fn = fn
        self.ft = ft
        # will first see if there is vertex color to use
        self.vc = vc
        # only support a single albedo image, (H, W, 3)
        self.albedo = albedo
        # pbr extension, metallic(Blue) = metallicRoughness[..., 2], roughness(Green) = metallicRoughness[..., 1]
        # ref: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
        self.metallicRoughness = metallicRoughness

        self.ori_center = 0
        self.ori_scale = 1

    @classmethod
    def load(cls, path, resize=True, renormal=True, retex=False, clean=False, bound=0.5, front_dir='+z', **kwargs):
        """load mesh from path.

        Args:
            path (str): path to mesh file, supports ply, obj, glb.
            resize (bool, optional): auto resize the mesh using ``bound`` into [-bound, bound]^3. Defaults to True.
            renormal (bool, optional): re-calc the vertex normals. Defaults to True.
            retex (bool, optional): re-calc the uv coordinates, will overwrite the existing uv coordinates. Defaults to False.
            clean (bool, optional): perform mesh cleaning at load (e.g., merge close vertices). Defaults to False.
            bound (float, optional): bound to resize. Defaults to 0.9.
            front_dir (str, optional): front-view direction of the mesh, should be [+-][xyz][ 123]. Defaults to '+z'.
            device (torch.device, optional): torch device. Defaults to None.
        
        Note:
            a ``device`` keyword argument can be provided to specify the torch device. 
            If it's not provided, we will try to use ``'cuda'`` as the device if it's available.

        Returns:
            Mesh: the loaded Mesh object.
        """
        # obj supports face uv
        if path.endswith(".obj"):
            mesh = cls.load_obj(path, **kwargs)
        # trimesh only supports vertex uv, but can load more formats
        else:
            mesh = cls.load_trimesh(path, **kwargs)
        
        # clean
        if clean:
            from kiui.mesh_utils import clean_mesh
            vertices = mesh.v.detach().cpu().numpy()
            triangles = mesh.f.detach().cpu().numpy()
            vertices, triangles = clean_mesh(vertices, triangles, remesh=False)
            mesh.v = torch.from_numpy(vertices).contiguous().float().to(mesh.device)
            mesh.f = torch.from_numpy(triangles).contiguous().int().to(mesh.device)

        print(f"[Mesh loading] v: {mesh.v.shape}, f: {mesh.f.shape}")
        # auto-normalize
        if resize:
            mesh.auto_size(bound=bound)
        # auto-fix normal
        if renormal or mesh.vn is None:
            mesh.auto_normal()
            print(f"[Mesh loading] vn: {mesh.vn.shape}, fn: {mesh.fn.shape}")
        # auto-fix texcoords
        if retex or (mesh.albedo is not None and mesh.vt is None):
            mesh.auto_uv(cache_path=path)
            print(f"[Mesh loading] vt: {mesh.vt.shape}, ft: {mesh.ft.shape}")

        # rotate front dir to +z
        if front_dir != "+z":
            # axis switch
            if "-z" in front_dir:
                T = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]], device=mesh.device, dtype=torch.float32)
            elif "+x" in front_dir:
                T = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], device=mesh.device, dtype=torch.float32)
            elif "-x" in front_dir:
                T = torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]], device=mesh.device, dtype=torch.float32)
            elif "+y" in front_dir:
                T = torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]], device=mesh.device, dtype=torch.float32)
            elif "-y" in front_dir:
                T = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], device=mesh.device, dtype=torch.float32)
            else:
                T = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], device=mesh.device, dtype=torch.float32)
            # rotation (how many 90 degrees)
            if '1' in front_dir:
                T @= torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], device=mesh.device, dtype=torch.float32) 
            elif '2' in front_dir:
                T @= torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]], device=mesh.device, dtype=torch.float32) 
            elif '3' in front_dir:
                T @= torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], device=mesh.device, dtype=torch.float32) 
            mesh.v @= T
            mesh.vn @= T

        return mesh

    # load from obj file
    @classmethod
    def load_obj(cls, path, albedo_path=None, device=None):
        """load an ``obj`` mesh.

        Args:
            path (str): path to mesh.
            albedo_path (str, optional): path to the albedo texture image, will overwrite the existing texture path if specified in mtl. Defaults to None.
            device (torch.device, optional): torch device. Defaults to None.
        
        Note: 
            We will try to read `mtl` path from `obj`, else we assume the file name is the same as `obj` but with `mtl` extension.
            The `usemtl` statement is ignored, and we only use the last material path in `mtl` file.

        Returns:
            Mesh: the loaded Mesh object.
        """
        assert os.path.splitext(path)[-1] == ".obj"

        mesh = cls()

        # device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mesh.device = device

        # load obj
        with open(path, "r") as f:
            lines = f.readlines()

        def parse_f_v(fv):
            # pass in a vertex term of a face, return {v, vt, vn} (-1 if not provided)
            # supported forms:
            # f v1 v2 v3
            # f v1/vt1 v2/vt2 v3/vt3
            # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
            # f v1//vn1 v2//vn2 v3//vn3
            xs = [int(x) - 1 if x != "" else -1 for x in fv.split("/")]
            xs.extend([-1] * (3 - len(xs)))
            return xs[0], xs[1], xs[2]

        vertices, texcoords, normals = [], [], []
        faces, tfaces, nfaces = [], [], []
        mtl_path = None

        for line in lines:
            split_line = line.split()
            # empty line
            if len(split_line) == 0:
                continue
            prefix = split_line[0].lower()
            # mtllib
            if prefix == "mtllib":
                mtl_path = split_line[1]
            # usemtl
            elif prefix == "usemtl":
                pass # ignored
            # v/vn/vt
            elif prefix == "v":
                vertices.append([float(v) for v in split_line[1:]])
            elif prefix == "vn":
                normals.append([float(v) for v in split_line[1:]])
            elif prefix == "vt":
                val = [float(v) for v in split_line[1:]]
                texcoords.append([val[0], 1.0 - val[1]])
            elif prefix == "f":
                vs = split_line[1:]
                nv = len(vs)
                v0, t0, n0 = parse_f_v(vs[0])
                for i in range(nv - 2):  # triangulate (assume vertices are ordered)
                    v1, t1, n1 = parse_f_v(vs[i + 1])
                    v2, t2, n2 = parse_f_v(vs[i + 2])
                    faces.append([v0, v1, v2])
                    tfaces.append([t0, t1, t2])
                    nfaces.append([n0, n1, n2])

        mesh.v = torch.tensor(vertices, dtype=torch.float32, device=device)
        mesh.vt = (
            torch.tensor(texcoords, dtype=torch.float32, device=device)
            if len(texcoords) > 0
            else None
        )
        mesh.vn = (
            torch.tensor(normals, dtype=torch.float32, device=device)
            if len(normals) > 0
            else None
        )

        mesh.f = torch.tensor(faces, dtype=torch.int32, device=device)
        mesh.ft = (
            torch.tensor(tfaces, dtype=torch.int32, device=device)
            if len(texcoords) > 0
            else None
        )
        mesh.fn = (
            torch.tensor(nfaces, dtype=torch.int32, device=device)
            if len(normals) > 0
            else None
        )

        # see if there is vertex color
        use_vertex_color = False
        if mesh.v.shape[1] == 6:
            use_vertex_color = True
            mesh.vc = mesh.v[:, 3:]
            mesh.v = mesh.v[:, :3]
            print(f"[load_obj] use vertex color: {mesh.vc.shape}")

        # try to load texture image
        if not use_vertex_color:
            # try to retrieve mtl file
            mtl_path_candidates = []
            if mtl_path is not None:
                mtl_path_candidates.append(mtl_path)
                mtl_path_candidates.append(os.path.join(os.path.dirname(path), mtl_path))
            mtl_path_candidates.append(path.replace(".obj", ".mtl"))

            mtl_path = None
            for candidate in mtl_path_candidates:
                if os.path.exists(candidate):
                    mtl_path = candidate
                    break
            
            # if albedo_path is not provided, try retrieve it from mtl
            metallic_path = None
            roughness_path = None
            if mtl_path is not None and albedo_path is None:
                with open(mtl_path, "r") as f:
                    lines = f.readlines()

                for line in lines:
                    split_line = line.split()
                    # empty line
                    if len(split_line) == 0:
                        continue
                    prefix = split_line[0]
                    
                    if "map_Kd" in prefix:
                        # assume relative path!
                        albedo_path = os.path.join(os.path.dirname(path), split_line[1])
                        print(f"[load_obj] use texture from: {albedo_path}")
                    elif "map_Pm" in prefix:
                        metallic_path = os.path.join(os.path.dirname(path), split_line[1])
                    elif "map_Pr" in prefix:
                        roughness_path = os.path.join(os.path.dirname(path), split_line[1])
                    
            # still not found albedo_path, or the path doesn't exist
            if albedo_path is None or not os.path.exists(albedo_path):
                # init an empty texture
                print(f"[load_obj] init empty albedo!")
                # albedo = np.random.rand(1024, 1024, 3).astype(np.float32)
                albedo = np.ones((1024, 1024, 3), dtype=np.float32) * np.array([0.5, 0.5, 0.5])  # default color
            else:
                albedo = cv2.imread(albedo_path, cv2.IMREAD_UNCHANGED)
                albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)
                albedo = albedo.astype(np.float32) / 255
                print(f"[load_obj] load texture: {albedo.shape}")
            
            mesh.albedo = torch.tensor(albedo, dtype=torch.float32, device=device)
            
            # try to load metallic and roughness
            if metallic_path is not None and roughness_path is not None:
                print(f"[load_obj] load metallicRoughness from: {metallic_path}, {roughness_path}")
                metallic = cv2.imread(metallic_path, cv2.IMREAD_UNCHANGED)
                metallic = metallic.astype(np.float32) / 255
                roughness = cv2.imread(roughness_path, cv2.IMREAD_UNCHANGED)
                roughness = roughness.astype(np.float32) / 255
                metallicRoughness = np.stack([np.zeros_like(metallic), roughness, metallic], axis=-1)

                mesh.metallicRoughness = torch.tensor(metallicRoughness, dtype=torch.float32, device=device).contiguous()

        return mesh

    @classmethod
    def load_trimesh(cls, path=None, given_mesh=None, device=None):
        """load a mesh using ``trimesh.load()``.

        Can load various formats like ``glb`` and serves as a fallback.

        Note:
            We will try to merge all meshes if the glb contains more than one, 
            but **this may cause the texture to lose**, since we only support one texture image!

        Args:
            path (str): path to the mesh file.
            given_mesh (trimesh): trimesh instance
            device (torch.device, optional): torch device. Defaults to None.

        Returns:
            Mesh: the loaded Mesh object.
        """
        mesh = cls()

        # device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mesh.device = device

        # use trimesh to load ply/glb, assume only has one single RootMesh...
        if given_mesh is not None:
            _mesh = given_mesh
        elif path is not None:
            _data = trimesh.load(path)
            if isinstance(_data, trimesh.Scene):
                if len(_data.geometry) == 1:
                    _mesh = list(_data.geometry.values())[0]
                else:
                    print(f"[load_trimesh] concatenating {len(_data.geometry)} meshes.")
                    _concat = []
                    # loop the scene graph and apply transform to each mesh
                    scene_graph = _data.graph.to_flattened() # dict {name: {transform: 4x4 mat, geometry: str}}
                    for k, v in scene_graph.items():
                        name = v['geometry']
                        if name in _data.geometry and isinstance(_data.geometry[name], trimesh.Trimesh):
                            transform = v['transform']
                            _concat.append(_data.geometry[name].apply_transform(transform))
                    _mesh = trimesh.util.concatenate(_concat)
            else:
                _mesh = _data
        else:
            print(f"[load_trimesh] failed to load mesh, either path or given_mesh must be given")
            return None
        
        should_create_empty_albedo = False
        if _mesh.visual.kind == 'vertex':
            vertex_colors = _mesh.visual.vertex_colors
            vertex_colors = np.array(vertex_colors[..., :3]).astype(np.float32) / 255
            mesh.vc = torch.tensor(vertex_colors, dtype=torch.float32, device=device)
            print(f"[load_trimesh] use vertex color: {mesh.vc.shape}")
        elif _mesh.visual.kind == 'texture':
            _material = _mesh.visual.material
            if isinstance(_material, trimesh.visual.material.PBRMaterial):
                texture = np.array(_material.baseColorTexture).astype(np.float32) / 255
                # load metallicRoughness if present
                if _material.metallicRoughnessTexture is not None:
                    metallicRoughness = np.array(_material.metallicRoughnessTexture).astype(np.float32) / 255
                    mesh.metallicRoughness = torch.tensor(metallicRoughness, dtype=torch.float32, device=device).contiguous()
            elif isinstance(_material, trimesh.visual.material.SimpleMaterial):
                texture = np.array(_material.to_pbr().baseColorTexture).astype(np.float32) / 255
            else:
                raise NotImplementedError(f"material type {type(_material)} not supported!")
            
            if texture.ndim ==3:
                mesh.albedo = torch.tensor(texture[..., :3], dtype=torch.float32, device=device).contiguous()
                print(f"[load_trimesh] loaded albedo texture: {texture.shape}")
            else:
                should_create_empty_albedo = True
        else:
            should_create_empty_albedo = True
            
        if should_create_empty_albedo:
            mesh.set_new_albedo(1024, 1024)
            print(f"[load_trimesh] failed to load albedo texture, create empty abbedo texture with shape: {mesh.albedo.shape}")

        vertices = _mesh.vertices

        try:
            texcoords = _mesh.visual.uv
            texcoords[:, 1] = 1 - texcoords[:, 1]
        except Exception as e:
            texcoords = None

        try:
            normals = _mesh.vertex_normals
        except Exception as e:
            normals = None

        # trimesh only support vertex uv...
        faces = tfaces = nfaces = _mesh.faces

        mesh.v = torch.tensor(vertices, dtype=torch.float32, device=device)
        mesh.vt = (
            torch.tensor(texcoords, dtype=torch.float32, device=device)
            if texcoords is not None
            else None
        )
        mesh.vn = (
            torch.tensor(normals, dtype=torch.float32, device=device)
            if normals is not None
            else None
        )

        mesh.f = torch.tensor(faces, dtype=torch.int32, device=device)
        mesh.ft = (
            torch.tensor(tfaces, dtype=torch.int32, device=device)
            if texcoords is not None
            else None
        )
        mesh.fn = (
            torch.tensor(nfaces, dtype=torch.int32, device=device)
            if normals is not None
            else None
        )

        return mesh
    
    def set_new_albedo(self, res_H, res_W):
        if self.albedo is None:
            texture = np.ones((res_H, res_W, 3), dtype=np.float32) * np.array([0.5, 0.5, 0.5])
            self.albedo = torch.tensor(texture, dtype=torch.float32, device=self.device)
        else:
            self.albedo = prepare_torch_img(self.albedo.unsqueeze(0), res_H, res_W, self.device).squeeze(0).permute(1, 2, 0).contiguous() # (1, 3, H, W) -> (H, W, 3)

    # aabb
    def aabb(self):
        """get the axis-aligned bounding box of the mesh.

        Returns:
            Tuple[torch.Tensor]: the min xyz and max xyz of the mesh.
        """
        return torch.min(self.v, dim=0).values, torch.max(self.v, dim=0).values

    # unit size
    @torch.no_grad()
    def auto_size(self, bound=0.9):
        """auto resize the mesh.

        Args:
            bound (float, optional): resizing into ``[-bound, bound]^3``. Defaults to 0.9.
        """
        vmin, vmax = self.aabb()
        self.ori_center = (vmax + vmin) / 2
        self.ori_scale = 2 * bound / torch.max(vmax - vmin).item()
        self.v = (self.v - self.ori_center) * self.ori_scale

    def auto_normal(self):
        """auto calculate the vertex normals.
        """
        i0, i1, i2 = self.f[:, 0].long(), self.f[:, 1].long(), self.f[:, 2].long()
        v0, v1, v2 = self.v[i0, :], self.v[i1, :], self.v[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)

        # Splat face normals to vertices
        vn = torch.zeros_like(self.v)
        vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        vn = torch.where(
            dot(vn, vn) > 1e-20,
            vn,
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device),
        )
        vn = safe_normalize(vn)

        self.vn = vn
        self.fn = self.f

    def auto_uv(self, cache_path=None, vmap=True):
        """auto calculate the uv coordinates.

        Args:
            cache_path (str, optional): path to save/load the uv cache as a npz file, this can avoid calculating uv every time when loading the same mesh, which is time-consuming. Defaults to None.
            vmap (bool, optional): remap vertices based on uv coordinates, so each v correspond to a unique vt (necessary for formats like gltf). 
                Usually this will duplicate the vertices on the edge of uv atlas. Defaults to True.
        """
        # try to load cache
        if cache_path is not None:
            cache_path = os.path.splitext(cache_path)[0] + "_uv.npz"
        if cache_path is not None and os.path.exists(cache_path):
            data = np.load(cache_path)
            vt_np, ft_np, vmapping = data["vt"], data["ft"], data["vmapping"]
        else:
            import xatlas

            v_np = self.v.detach().cpu().numpy()
            f_np = self.f.detach().int().cpu().numpy()
            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            chart_options = xatlas.ChartOptions()
            # chart_options.max_iterations = 4
            atlas.generate(chart_options=chart_options)
            vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

            # save to cache
            if cache_path is not None:
                np.savez(cache_path, vt=vt_np, ft=ft_np, vmapping=vmapping)
        
        vt = torch.from_numpy(vt_np.astype(np.float32)).to(self.device)
        ft = torch.from_numpy(ft_np.astype(np.int32)).to(self.device)
        self.vt = vt
        self.ft = ft

        if vmap:
            # remap v/f to vt/ft, so each v correspond to a unique vt. (necessary for gltf)
            vmapping = torch.from_numpy(vmapping.astype(np.int64)).long().to(self.device)
            self.align_v_to_vt(vmapping)
    
    def align_v_to_vt(self, vmapping=None):
        """ remap v/f and vn/fn to vt/ft.

        Args:
            vmapping (np.ndarray, optional): the mapping relationship from f to ft. Defaults to None.
        """
        if vmapping is None:
            vt2v_mapping = self.get_default_vt_to_v_mapping()
        else:
            vt2v_mapping = vmapping

        self.v = self.v[vt2v_mapping]
        if self.vc is not None:
            self.vc = self.vc[vt2v_mapping]
        self.f = self.ft
        # assume fn == f
        if self.vn is not None:
            if vmapping is None:
                vt2vn_mapping = self.get_default_vt_to_vn_mapping()
            else:
                vt2vn_mapping = vmapping
            self.vn = self.vn[vt2vn_mapping]
            self.fn = self.ft

    def get_default_vt_to_v_mapping(self):
        """map from ft to f

        Returns:
            vmapping: tensor array
        """
        ft = self.ft.view(-1).long()
        f = self.f.view(-1).long()
        vmapping = torch.zeros(self.vt.shape[0], dtype=torch.long, device=self.device)
        vmapping[ft] = f # scatter, randomly choose one if index is not unique
        return vmapping
    
    def get_default_vt_to_vn_mapping(self):
        """map from ft to f

        Returns:
            vmapping: tensor array
        """
        ft = self.ft.view(-1).long()
        fn = self.fn.view(-1).long()
        vmapping = torch.zeros(self.vt.shape[0], dtype=torch.long, device=self.device)
        vmapping[ft] = fn # scatter, randomly choose one if index is not unique
        return vmapping

    def to(self, device):
        """move all tensor attributes to device.

        Args:
            device (torch.device): target device.

        Returns:
            Mesh: self.
        """
        self.device = device
        for name in ["v", "f", "vn", "fn", "vt", "ft", "albedo", "vc", "metallicRoughness"]:
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.to(device))
        return self
    
    def write(self, path):
        """write the mesh to a path.

        Args:
            path (str): path to write, supports ply, obj and glb.
        """
        if path.endswith(".ply"):
            self.write_ply(path)
        elif path.endswith(".obj"):
            self.write_obj(path)
        elif path.endswith(".glb") or path.endswith(".gltf"):
            self.write_glb(path)
        else:
            raise NotImplementedError(f"format {path} not supported!")
    
    def write_ply(self, path):
        """write the mesh in ply format. Only for geometry!

        Args:
            path (str): path to write.
        """

        if self.albedo is not None:
            print(f'[WARN] ply format does not support exporting texture, will ignore!')

        v_np = self.v.detach().cpu().numpy()
        f_np = self.f.detach().cpu().numpy()

        _mesh = trimesh.Trimesh(vertices=v_np, faces=f_np)
        _mesh.export(path)


    def write_glb(self, path):
        """write the mesh in glb/gltf format.
          This will create a scene with a single mesh.

        Args:
            path (str): path to write.
        """

        # assert self.v.shape[0] == self.vn.shape[0] and self.v.shape[0] == self.vt.shape[0]
        if self.vt is not None and self.v.shape[0] != self.vt.shape[0]:
            self.align_v_to_vt()

        import pygltflib

        f_np = self.f.detach().cpu().numpy().astype(np.uint32)
        f_np_blob = f_np.flatten().tobytes()

        v_np = self.v.detach().cpu().numpy().astype(np.float32)
        v_np_blob = v_np.tobytes()

        blob = f_np_blob + v_np_blob
        byteOffset = len(blob)

        # base mesh
        gltf = pygltflib.GLTF2(
            scene=0,
            scenes=[pygltflib.Scene(nodes=[0])],
            nodes=[pygltflib.Node(mesh=0)],
            meshes=[pygltflib.Mesh(primitives=[pygltflib.Primitive(
                # indices to accessors (0 is triangles)
                attributes=pygltflib.Attributes(
                    POSITION=1,
                ),
                indices=0,
            )])],
            buffers=[
                pygltflib.Buffer(byteLength=len(f_np_blob) + len(v_np_blob))
            ],
            # buffer view (based on dtype)
            bufferViews=[
                # triangles; as flatten (element) array
                pygltflib.BufferView(
                    buffer=0,
                    byteLength=len(f_np_blob),
                    target=pygltflib.ELEMENT_ARRAY_BUFFER, # GL_ELEMENT_ARRAY_BUFFER (34963)
                ),
                # positions; as vec3 array
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(f_np_blob),
                    byteLength=len(v_np_blob),
                    byteStride=12, # vec3
                    target=pygltflib.ARRAY_BUFFER, # GL_ARRAY_BUFFER (34962)
                ),
            ],
            accessors=[
                # 0 = triangles
                pygltflib.Accessor(
                    bufferView=0,
                    componentType=pygltflib.UNSIGNED_INT, # GL_UNSIGNED_INT (5125)
                    count=f_np.size,
                    type=pygltflib.SCALAR,
                    max=[int(f_np.max())],
                    min=[int(f_np.min())],
                ),
                # 1 = positions
                pygltflib.Accessor(
                    bufferView=1,
                    componentType=pygltflib.FLOAT, # GL_FLOAT (5126)
                    count=len(v_np),
                    type=pygltflib.VEC3,
                    max=v_np.max(axis=0).tolist(),
                    min=v_np.min(axis=0).tolist(),
                ),
            ],
        )

        # append texture info
        if self.vt is not None and self.albedo is not None:

            vt_np = self.vt.detach().cpu().numpy().astype(np.float32)
            vt_np_blob = vt_np.tobytes()

            albedo = self.albedo.detach().cpu().numpy()
            albedo = (albedo * 255).astype(np.uint8)
            albedo = cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR)
            albedo_blob = cv2.imencode('.png', albedo)[1].tobytes()

            # update primitive
            gltf.meshes[0].primitives[0].attributes.TEXCOORD_0 = 2
            gltf.meshes[0].primitives[0].material = 0

            # update materials
            gltf.materials.append(pygltflib.Material(
                pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                    baseColorTexture=pygltflib.TextureInfo(index=0, texCoord=0),
                    metallicFactor=0.0,
                    roughnessFactor=1.0,
                ),
                alphaMode=pygltflib.OPAQUE,
                alphaCutoff=None,
                doubleSided=True,
            ))

            gltf.textures.append(pygltflib.Texture(sampler=0, source=0))
            gltf.samplers.append(pygltflib.Sampler(magFilter=pygltflib.LINEAR, minFilter=pygltflib.LINEAR_MIPMAP_LINEAR, wrapS=pygltflib.REPEAT, wrapT=pygltflib.REPEAT))
            gltf.images.append(pygltflib.Image(bufferView=3, mimeType="image/png"))

            # update buffers
            gltf.bufferViews.append(
                # index = 2, texcoords; as vec2 array
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=byteOffset,
                    byteLength=len(vt_np_blob),
                    byteStride=8, # vec2
                    target=pygltflib.ARRAY_BUFFER,
                )
            )

            gltf.accessors.append(
                # 2 = texcoords
                pygltflib.Accessor(
                    bufferView=2,
                    componentType=pygltflib.FLOAT,
                    count=len(vt_np),
                    type=pygltflib.VEC2,
                    max=vt_np.max(axis=0).tolist(),
                    min=vt_np.min(axis=0).tolist(),
                )
            )

            blob += vt_np_blob 
            byteOffset += len(vt_np_blob)

            gltf.bufferViews.append(
                # index = 3, albedo texture; as none target
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=byteOffset,
                    byteLength=len(albedo_blob),
                )
            )

            blob += albedo_blob
            byteOffset += len(albedo_blob)

            gltf.buffers[0].byteLength = byteOffset

            # append metllic roughness
            if self.metallicRoughness is not None:
                metallicRoughness = self.metallicRoughness.detach().cpu().numpy()
                metallicRoughness = (metallicRoughness * 255).astype(np.uint8)
                metallicRoughness = cv2.cvtColor(metallicRoughness, cv2.COLOR_RGB2BGR)
                metallicRoughness_blob = cv2.imencode('.png', metallicRoughness)[1].tobytes()

                # update texture definition
                gltf.materials[0].pbrMetallicRoughness.metallicFactor = 1.0
                gltf.materials[0].pbrMetallicRoughness.roughnessFactor = 1.0
                gltf.materials[0].pbrMetallicRoughness.metallicRoughnessTexture = pygltflib.TextureInfo(index=1, texCoord=0)

                gltf.textures.append(pygltflib.Texture(sampler=1, source=1))
                gltf.samplers.append(pygltflib.Sampler(magFilter=pygltflib.LINEAR, minFilter=pygltflib.LINEAR_MIPMAP_LINEAR, wrapS=pygltflib.REPEAT, wrapT=pygltflib.REPEAT))
                gltf.images.append(pygltflib.Image(bufferView=4, mimeType="image/png"))

                # update buffers
                gltf.bufferViews.append(
                    # index = 4, metallicRoughness texture; as none target
                    pygltflib.BufferView(
                        buffer=0,
                        byteOffset=byteOffset,
                        byteLength=len(metallicRoughness_blob),
                    )
                )

                blob += metallicRoughness_blob
                byteOffset += len(metallicRoughness_blob)

                gltf.buffers[0].byteLength = byteOffset

            
        # set actual data
        gltf.set_binary_blob(blob)

        # glb = b"".join(gltf.save_to_bytes())
        gltf.save(path)


    def write_obj(self, path):
        """write the mesh in obj format. Will also write the texture and mtl files.

        Args:
            path (str): path to write.
        """

        mtl_path = path.replace(".obj", ".mtl")
        albedo_path = path.replace(".obj", "_albedo.png")
        metallic_path = path.replace(".obj", "_metallic.png")
        roughness_path = path.replace(".obj", "_roughness.png")

        v_np = self.v.detach().cpu().numpy()
        vt_np = self.vt.detach().cpu().numpy() if self.vt is not None else None
        vn_np = self.vn.detach().cpu().numpy() if self.vn is not None else None
        vc_np = self.vc.detach().cpu().numpy() if self.vc is not None else None
        f_np = self.f.detach().cpu().numpy()
        ft_np = self.ft.detach().cpu().numpy() if self.ft is not None else None
        fn_np = self.fn.detach().cpu().numpy() if self.fn is not None else None

        with open(path, "w") as fp:
            fp.write(f"mtllib {os.path.basename(mtl_path)} \n")

            if vc_np is None:
                for v in v_np:
                    fp.write(f"v {v[0]} {v[1]} {v[2]} \n")
            else:
                for v_i in range(v_np.shape[0]):
                    v = v_np[v_i]
                    vc = vc_np[v_i]
                    fp.write(f"v {v[0]} {v[1]} {v[2]} {vc[0]} {vc[1]} {vc[2]}\n")

            if vt_np is not None:
                for v in vt_np:
                    fp.write(f"vt {v[0]} {1 - v[1]} \n")

            if vn_np is not None:
                for v in vn_np:
                    fp.write(f"vn {v[0]} {v[1]} {v[2]} \n")

            fp.write(f"usemtl defaultMat \n")
            for i in range(len(f_np)):
                fp.write(
                    f'f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1 if ft_np is not None else ""}/{fn_np[i, 0] + 1 if fn_np is not None else ""} \
                             {f_np[i, 1] + 1}/{ft_np[i, 1] + 1 if ft_np is not None else ""}/{fn_np[i, 1] + 1 if fn_np is not None else ""} \
                             {f_np[i, 2] + 1}/{ft_np[i, 2] + 1 if ft_np is not None else ""}/{fn_np[i, 2] + 1 if fn_np is not None else ""} \n'
                )

        with open(mtl_path, "w") as fp:
            fp.write(f"newmtl defaultMat \n")
            fp.write(f"Ka 1 1 1 \n")
            fp.write(f"Kd 1 1 1 \n")
            fp.write(f"Ks 1 1 1 \n")
            #fp.write(f"Tr 1 \n") # will cause the mesh materials in three.js completely transparent
            fp.write(f"illum 1 \n")
            fp.write(f"Ns 10 \n")
            if self.albedo is not None:
                fp.write(f"map_Kd {os.path.basename(albedo_path)} \n")
            if self.metallicRoughness is not None:
                # ref: https://en.wikipedia.org/wiki/Wavefront_.obj_file#Physically-based_Rendering
                fp.write(f"map_Pm {os.path.basename(metallic_path)} \n")
                fp.write(f"map_Pr {os.path.basename(roughness_path)} \n")

        if self.albedo is not None:
            albedo = self.albedo.detach().cpu().numpy()
            albedo = (albedo * 255).astype(np.uint8)
            cv2.imwrite(albedo_path, cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR))
        
        if self.metallicRoughness is not None:
            metallicRoughness = self.metallicRoughness.detach().cpu().numpy()
            metallicRoughness = (metallicRoughness * 255).astype(np.uint8)
            cv2.imwrite(metallic_path, metallicRoughness[..., 2])
            cv2.imwrite(roughness_path, metallicRoughness[..., 1])
        
    def convert_to_pointcloud(self):
        xyz = self.v.detach().cpu().numpy()
        num_pts = self.v.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        normals = np.zeros((num_pts, 3))
        pcd = PointCloud(points=xyz, colors=SH2RGB(shs), normals=normals)
        return pcd
        
        
class PointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array