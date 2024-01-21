import os
import tqdm

import numpy as np
import torch
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing
from pytorch3d.structures.meshes import Meshes

import comfy.utils

from ..mesh_processer.mesh import Mesh
from ..mesh_processer.mesh_utils import sample_points
from .dmtet_network import Decoder

# Deep Marching Tetrahedrons implementation, adapted from
# https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/tetmesh.py
class DMTet:
    def __init__(self, device='cuda'):
        self.device = device
        self.triangle_table = torch.tensor([
                [-1, -1, -1, -1, -1, -1],
                [ 1,  0,  2, -1, -1, -1],
                [ 4,  0,  3, -1, -1, -1],
                [ 1,  4,  2,  1,  3,  4],
                [ 3,  1,  5, -1, -1, -1],
                [ 2,  3,  0,  2,  5,  3],
                [ 1,  4,  0,  1,  5,  4],
                [ 4,  2,  5, -1, -1, -1],
                [ 4,  5,  2, -1, -1, -1],
                [ 4,  1,  0,  4,  5,  1],
                [ 3,  2,  0,  3,  5,  2],
                [ 1,  3,  5, -1, -1, -1],
                [ 4,  1,  2,  4,  3,  1],
                [ 3,  0,  4, -1, -1, -1],
                [ 2,  0,  1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1]
                ], dtype=torch.long, device=self.device)

        self.num_triangles_table = torch.tensor([0,1,1,2,1,2,2,1,1,2,2,1,2,1,1,0], dtype=torch.long, device=self.device)
        self.base_tet_edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype=torch.long, device=self.device)

    # Utility functions
    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:,0] > edges_ex2[:,1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)      
            b = torch.gather(input=edges_ex2, index=1-order, dim=1)  

        return torch.stack([a, b],-1)

    def map_uv(self, faces, face_gidx, max_idx):
        N = int(np.ceil(np.sqrt((max_idx+1)//2)))
        tex_y, tex_x = torch.meshgrid(
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            indexing='ij'
        )

        pad = 0.9 / N

        uvs = torch.stack([
            tex_x      , tex_y,
            tex_x + pad, tex_y,
            tex_x + pad, tex_y + pad,
            tex_x      , tex_y + pad
        ], dim=-1).view(-1, 2)

        def _idx(tet_idx, N):
            x = tet_idx % N
            y = torch.div(tet_idx, N, rounding_mode='trunc')
            return y * N + x

        tet_idx = _idx(torch.div(face_gidx, 2, rounding_mode='trunc'), N)
        tri_idx = face_gidx % 2

        uv_idx = torch.stack((
            tet_idx * 4, tet_idx * 4 + tri_idx + 1, tet_idx * 4 + tri_idx + 2
        ), dim = -1). view(-1, 3)

        return uvs, uv_idx

    # Marching tets implementation
    def __call__(self, vertices, tet, sdf, with_uv=False):
        
        r"""Convert discrete signed distance fields encoded on tetrahedral grids to triangle 
        meshes using marching tetrahedra algorithm as described in `An efficient method of 
        triangulating equi-valued surfaces by using tetrahedral cells`_. The output surface is differentiable with respect to
        input vertex positions and the SDF values. For more details and example usage in learning, see 
        `Deep Marching Tetrahedra\: a Hybrid Representation for High-Resolution 3D Shape Synthesis`_ NeurIPS 2021.


        Args:
            vertices (torch.tensor): batched vertices of tetrahedral meshes, of shape
                                    : (num_vertices, 3).
            tets (torch.tensor): unbatched tetrahedral mesh topology, of shape
                                : (num_tetrahedrons, 4).
            sdf (torch.tensor): batched SDFs which specify the SDF value of each vertex, of shape
                                : (num_vertices).
            return_tet_idx (optional, bool): if True, return index of tetrahedron
                                            where each face is extracted. Default: False.

        Returns:
            (list[torch.Tensor], list[torch.LongTensor], (optional) list[torch.LongTensor]): 

                - the list of vertices for mesh converted from each tetrahedral grid.
                - the list of faces for mesh converted from each tetrahedral grid.
                - the list of indices that correspond to tetrahedra where faces are extracted.

        Example:
            >>> vertices = torch.tensor([[[0, 0, 0],
            ...               [1, 0, 0],
            ...               [0, 1, 0],
            ...               [0, 0, 1]]], dtype=torch.float)
            >>> tets = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
            >>> sdf = torch.tensor([[-1., -1., 0.5, 0.5]], dtype=torch.float)
            >>> verts_list, faces_list, tet_idx_list = marching_tetrahedra(vertices, tets, sdf, True)
            >>> verts_list[0]
            tensor([[0.0000, 0.6667, 0.0000],
                    [0.0000, 0.0000, 0.6667],
                    [0.3333, 0.6667, 0.0000],
                    [0.3333, 0.0000, 0.6667]])
            >>> faces_list[0]
            tensor([[3, 0, 1],
                    [3, 2, 0]])
            >>> tet_idx_list[0]
            tensor([0, 0])

        .. _An efficient method of triangulating equi-valued surfaces by using tetrahedral cells:
            https://search.ieice.org/bin/summary.php?id=e74-d_1_214

        .. _Deep Marching Tetrahedra\: a Hybrid Representation for High-Resolution 3D Shape Synthesis:
                https://arxiv.org/abs/2111.04276
        """
        
        with torch.no_grad():
            occ_n = sdf > 0
            occ_fx4 = occ_n[tet.reshape(-1)].reshape(-1,4)
            occ_sum = torch.sum(occ_fx4, -1)
            valid_tets = (occ_sum>0) & (occ_sum<4)
            occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = tet[valid_tets][:,self.base_tet_edges].reshape(-1,2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges,dim=0, return_inverse=True)  
            
            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1,2).sum(-1) == 1
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device="cuda") * -1
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long,device="cuda")
            idx_map = mapping[idx_map] # map edges to verts

            interp_v = unique_edges[mask_edges]
        edges_to_interp = vertices[interp_v.reshape(-1)].reshape(-1,2,3)
        edges_to_interp_sdf = sdf[interp_v.reshape(-1)].reshape(-1,2,1)
        edges_to_interp_sdf[:,-1] *= -1

        denominator = edges_to_interp_sdf.sum(1,keepdim = True)

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1])/denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1,6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device="cuda"))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat((
            torch.gather(input=idx_map[num_triangles == 1], dim=1, index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1,3),
            torch.gather(input=idx_map[num_triangles == 2], dim=1, index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1,3),
        ), dim=0)

        if with_uv:
            # Get global face index (static, does not depend on topology)
            num_tets = tet.shape[0]
            tet_gidx = torch.arange(num_tets, dtype=torch.long, device="cuda")[valid_tets]
            face_gidx = torch.cat((
                tet_gidx[num_triangles == 1]*2,
                torch.stack((tet_gidx[num_triangles == 2]*2, tet_gidx[num_triangles == 2]*2 + 1), dim=-1).view(-1)
            ), dim=0)

            uvs, uv_idx = self.map_uv(faces, face_gidx, num_tets*2)

            return verts, faces, uvs, uv_idx
        else:
            return verts, faces

#  Geometry interface
class DMTetMesh:
    def __init__(self, iterations=5000, chamfer_distance_weight=1, laplacian_weight=0.1, chamfer_faces_sample_scale=1, mesh_scale=1.0, grid_res=128, geom_lr=0.0001, multires=2, mlp_internal_dims=128, mlp_hidden_layer_num=5, device='cuda'):
        
        self.device = torch.device(device)
        self.chamfer_distance_weight = chamfer_distance_weight
        self.laplacian_weight = laplacian_weight
        self.chamfer_faces_sample_scale = chamfer_faces_sample_scale
        self.iterations = iterations
        self.grid_res = grid_res

        self.marching_tets = DMTet(device=self.device)
        
        self.model = Decoder(multires=multires, internal_dims=mlp_internal_dims, hidden=mlp_hidden_layer_num).to(self.device)
        self.model.pre_train_sphere()
        
        self.geom_optimizer = torch.optim.Adam(self.model.parameters(), lr=geom_lr)
        self.geom_scheduler = torch.optim.lr_scheduler.LambdaLR(self.geom_optimizer, lr_lambda=lambda x: self.lr_schedule(x)) 

        # Load tetrahedral tile, for grid_res==128: verts: torch.Size([277410, 3]), indices: torch.Size([1524684, 4])
        # Get pre-generated tetrahedral grid from https://github.com/NVlabs/nvdiffrec/tree/main/data/tets
        tets_path = os.path.join(os.path.dirname(__file__), f'../data/tets/{self.grid_res}_tets.npz')
        tets = np.load(tets_path)
        self.verts = torch.tensor(tets['vertices'], dtype=torch.float32, device=self.device) * mesh_scale
        self.indices = torch.tensor(tets['indices'], dtype=torch.long, device=self.device)
        
        self.generate_edges()
        
    def lr_schedule(self, iter):
        return max(0.0, 10**(-iter*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs.

    def generate_edges(self):
        with torch.no_grad():
            edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype = torch.long, device = "cuda")
            all_edges = self.indices[:,edges].reshape(-1,2)
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)
            
    def get_mesh(self):
        verts, faces = self.pred_verts_faces()
        mesh = Mesh(v=verts, f=faces)
        mesh.auto_normal()
        mesh.auto_uv()
        return mesh
        
    def pred_verts_faces(self):
        pred = self.model(self.verts)   # predict SDF and per-vertex deformation
        sdf, deform = self.model.get_sdf_and_deform(pred)
        v_deformed = self.verts + torch.tanh(deform) / self.grid_res   # constraint deformation to avoid flipping tets
        return self.marching_tets(v_deformed, self.indices, sdf)
    
    def training(self, pcd, ref_imgs, ref_masks):
        
        allow_ref_points_loss = pcd is not None
        if allow_ref_points_loss:
            ref_points_batched = torch.from_numpy(np.asarray(pcd.points)).float().to(self.device).unsqueeze(0)
        
        allow_imgs_loss = ref_imgs is not None
        allow_masks_loss = ref_masks is not None
        
        comfy_pbar = comfy.utils.ProgressBar(self.iterations)
        
        for step in tqdm.trange(self.iterations):
            
            loss = 0
            
            verts, faces = self.pred_verts_faces()

            # chamfer distance loss
            if allow_ref_points_loss:
                num_faces_samples = int(self.chamfer_faces_sample_scale * faces.shape[0])
                sampled_points_per_faces = sample_points(verts.unsqueeze(0), faces, num_faces_samples)[0]
                loss += self.chamfer_distance_weight * chamfer_distance(sampled_points_per_faces, ref_points_batched)[0].mean()
                
            # laplacian smoothing regularization loss
            mesh = Meshes([verts], [faces])
            loss += self.laplacian_weight * mesh_laplacian_smoothing(mesh)
            
            # TODO: add images based diffrast loss, i.e (RGB, Alpha, Normal; with L1, SSIM loss)

            loss.backward()
            self.geom_optimizer.step()
            self.geom_scheduler.step()
            self.geom_optimizer.zero_grad()
            
            comfy_pbar.update_absolute(step + 1)