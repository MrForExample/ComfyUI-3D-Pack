import torch
import numpy as np
from kornia.geometry.conversions import (
    quaternion_to_axis_angle,
    axis_angle_to_quaternion,
)
import pymeshlab as pml
from plyfile import PlyData, PlyElement

from typing import Optional

pytorch3d_capable = True
try:
    import pytorch3d
    from pytorch3d.ops import knn_points
except ImportError:
    pytorch3d_capable = False

from .mesh import PointCloud
from shared_utils.sh_utils import SH2RGB, RGB2SH

def _base_face_areas(face_vertices_0, face_vertices_1, face_vertices_2):
    """Base function to compute the face areas."""
    x1, x2, x3 = torch.split(face_vertices_0 - face_vertices_1, 1, dim=-1)
    y1, y2, y3 = torch.split(face_vertices_1 - face_vertices_2, 1, dim=-1)

    a = (x2 * y3 - x3 * y2) ** 2
    b = (x3 * y1 - x1 * y3) ** 2
    c = (x1 * y2 - x2 * y1) ** 2
    areas = torch.sqrt(a + b + c) * 0.5

    return areas


def _base_sample_points_selected_faces(face_vertices, face_features=None):
    """Base function to sample points over selected faces, sample one point per face.
       The coordinates of the face vertices are interpolated to generate new samples.

    Args:
        face_vertices (tuple of torch.Tensor):
            Coordinates of vertices, corresponding to selected faces to sample from.
            A tuple of 3 entries corresponding to each of the face vertices.
            Each entry is a torch.Tensor of shape :math:`(\\text{batch_size}, \\text{num_samples}, 3)`.
        face_features (tuple of torch.Tensor, Optional):
            Features of face vertices, corresponding to selected faces to sample from.
            A tuple of 3 entries corresponding to each of the face vertices.
            Each entry is a torch.Tensor of shape
            :math:`(\\text{batch_size}, \\text{num_samples}, \\text{feature_dim})`.

    Returns:
        (torch.Tensor, torch.Tensor):
            Sampled point coordinates of shape :math:`(\\text{batch_size}, \\text{num_samples}, 3)`.
            Sampled points interpolated features of shape
            :math:`(\\text{batch_size}, \\text{num_samples}, \\text{feature_dim})`.
            If `face_vertices_features` arg is not specified, the returned interpolated features are None.
    """

    face_vertices0, face_vertices1, face_vertices2 = face_vertices

    sampling_shape = tuple(int(d) for d in face_vertices0.shape[:-1]) + (1,)
    # u is proximity to middle point between v1 and v2 against v0.
    # v is proximity to v2 against v1.
    #
    # The probability density for u should be f_U(u) = 2u.
    # However, torch.rand use a uniform (f_X(x) = x) distribution,
    # so using torch.sqrt we make a change of variable to have the desired density
    # f_Y(y) = f_X(y ^ 2) * |d(y ^ 2) / dy| = 2y
    u = torch.sqrt(torch.rand(sampling_shape,
                              device=face_vertices0.device,
                              dtype=face_vertices0.dtype))

    v = torch.rand(sampling_shape,
                   device=face_vertices0.device,
                   dtype=face_vertices0.dtype)
    w0 = 1 - u
    w1 = u * (1 - v)
    w2 = u * v

    points = w0 * face_vertices0 + w1 * face_vertices1 + w2 * face_vertices2

    features = None
    if face_features is not None:
        face_features0, face_features1, face_features2 = face_features
        features = w0 * face_features0 + w1 * face_features1 + \
            w2 * face_features2

    return points, features

# Modified from https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/mesh/trianglemesh.py#L159
def sample_points(vertices, faces, num_samples, areas=None, face_features=None):
    r"""Uniformly sample points over the surface of triangle meshes.

    First, face on which the point is sampled is randomly selected,
    with the probability of selection being proportional to the area of the face.
    then the coordinate on the face is uniformly sampled.

    If ``face_features`` is defined for the mesh faces,
    the sampled points will be returned with interpolated features as well,
    otherwise, no feature interpolation will occur.

    Args:
        vertices (torch.Tensor):
            The vertices of the meshes, of shape
            :math:`(\text{batch_size}, \text{num_vertices}, 3)`.
        faces (torch.LongTensor):
            The faces of the mesh, of shape :math:`(\text{num_faces}, 3)`.
        num_samples (int):
            The number of point sampled per mesh.
            Also the number of faces sampled per mesh, and then sample a single point per face.
        areas (torch.Tensor, optional):
            The areas of each face, of shape :math:`(\text{batch_size}, \text{num_faces})`,
            can be preprocessed, for fast on-the-fly sampling,
            will be computed if None (default).
        face_features (torch.Tensor, optional):
            Per-vertex-per-face features, matching ``faces`` order,
            of shape :math:`(\text{batch_size}, \text{num_faces}, 3, \text{feature_dim})`.
            For example:

                1. Texture uv coordinates would be of shape
                   :math:`(\text{batch_size}, \text{num_faces}, 3, 2)`.
                2. RGB color values would be of shape
                   :math:`(\text{batch_size}, \text{num_faces}, 3, 3)`.

            When specified, it is used to interpolate the features for new sampled points.

    See also:
        :func:`~kaolin.ops.mesh.index_vertices_by_faces` for conversion of features defined per vertex
        and need to be converted to per-vertex-per-face shape of :math:`(\text{num_faces}, 3)`.

    Returns:
        (torch.Tensor, torch.LongTensor, (optional) torch.Tensor):
            the pointclouds of shape :math:`(\text{batch_size}, \text{num_samples}, 3)`,
            and the indexes of the faces selected,
            of shape :math:`(\text{batch_size}, \text{num_samples})`.

            If ``face_features`` arg is specified, then the interpolated features of sampled points of shape
            :math:`(\text{batch_size}, \text{num_samples}, \text{feature_dim})` are also returned.
    """
    if faces.shape[-1] != 3:
        raise NotImplementedError("sample_points is only implemented for triangle meshes")
    faces_0, faces_1, faces_2 = torch.split(faces, 1, dim=1)        # (num_faces, 3) -> tuple of (num_faces,)
    face_v_0 = torch.index_select(vertices, 1, faces_0.reshape(-1))  # (batch_size, num_faces, 3)
    face_v_1 = torch.index_select(vertices, 1, faces_1.reshape(-1))  # (batch_size, num_faces, 3)
    face_v_2 = torch.index_select(vertices, 1, faces_2.reshape(-1))  # (batch_size, num_faces, 3)

    if areas is None:
        areas = _base_face_areas(face_v_0, face_v_1, face_v_2).squeeze(-1)
    face_dist = torch.distributions.Categorical(areas)
    face_choices = face_dist.sample([num_samples]).transpose(0, 1)
    _face_choices = face_choices.unsqueeze(-1).repeat(1, 1, 3)
    v0 = torch.gather(face_v_0, 1, _face_choices)  # (batch_size, num_samples, 3)
    v1 = torch.gather(face_v_1, 1, _face_choices)  # (batch_size, num_samples, 3)
    v2 = torch.gather(face_v_2, 1, _face_choices)  # (batch_size, num_samples, 3)
    face_vertices_choices = (v0, v1, v2)

    # UV coordinates are available, make sure to calculate them for sampled points as well
    face_features_choices = None
    if face_features is not None:
        feat_dim = face_features.shape[-1]
        # (num_faces, 3) -> tuple of (num_faces,)
        _face_choices = face_choices[..., None, None].repeat(1, 1, 3, feat_dim)
        face_features_choices = torch.gather(face_features, 1, _face_choices)
        face_features_choices = tuple(
            tmp_feat.squeeze(2) for tmp_feat in torch.split(face_features_choices, 1, dim=2))

    points, point_features = _base_sample_points_selected_faces(
        face_vertices_choices, face_features_choices)

    if point_features is not None:
        return points, face_choices, point_features
    else:
        return points, face_choices

def poisson_mesh_reconstruction(points, normals=None):
    # points/normals: [N, 3] np.ndarray

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # outlier removal
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=10)

    # normals
    if normals is None:
        pcd.estimate_normals()
    else:
        pcd.normals = o3d.utility.Vector3dVector(normals[ind])

    # visualize
    o3d.visualization.draw_geometries([pcd], point_show_normal=False)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9
    )
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # visualize
    o3d.visualization.draw_geometries([mesh])

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    print(
        f"[INFO] poisson mesh reconstruction: {points.shape} --> {vertices.shape} / {triangles.shape}"
    )

    return vertices, triangles


def decimate_mesh(
    verts, faces, target=5e4, backend="pymeshlab", remesh=False, optimalplacement=True, verbose=True
):
    """ perform mesh decimation.

    Args:
        verts (np.ndarray): mesh vertices, float [N, 3]
        faces (np.ndarray): mesh faces, int [M, 3]
        target (int): targeted number of faces
        backend (str, optional): algorithm backend, can be "pymeshlab" or "pyfqmr". Defaults to "pymeshlab".
        remesh (bool, optional): whether to remesh after decimation. Defaults to False.
        optimalplacement (bool, optional): For flat mesh, use False to prevent spikes. Defaults to True.
        verbose (bool, optional): whether to print the decimation process. Defaults to True.

    Returns:
        Tuple[np.ndarray]: vertices and faces after decimation.
    """

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    if backend == "pyfqmr":
        import pyfqmr

        solver = pyfqmr.Simplify()
        solver.setMesh(verts, faces)
        solver.simplify_mesh(target_count=target, preserve_border=False, verbose=False)
        verts, faces, normals = solver.getMesh()
    else:
        m = pml.Mesh(verts, faces)
        ms = pml.MeshSet()
        ms.add_mesh(m, "mesh")  # will copy!
        
        ms.meshing_merge_close_vertices()

        # filters
        # ms.meshing_decimation_clustering(threshold=pml.PercentageValue(1))
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target), optimalplacement=optimalplacement
        )

        if remesh:
            # ms.apply_coord_taubin_smoothing()
            ms.meshing_isotropic_explicit_remeshing(
                iterations=3, targetlen=pml.PercentageValue(1)
            )

        # extract mesh
        m = ms.current_mesh()
        m.compact()
        verts = m.vertex_matrix()
        faces = m.face_matrix()

    if verbose:
        print(f"[INFO] mesh decimation: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}")

    return verts, faces


def clean_mesh(
    verts,
    faces,
    v_pct=1,
    min_f=64,
    min_d=20,
    repair=True,
    remesh=True,
    remesh_size=0.01,
):
    # verts: [N, 3]
    # faces: [N, 3]

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, "mesh")  # will copy!

    # filters
    ms.meshing_remove_unreferenced_vertices()  # verts not refed by any faces

    if v_pct > 0:
        ms.meshing_merge_close_vertices(
            threshold=pml.Percentage(v_pct)
        )  # 1/10000 of bounding box diagonal

    ms.meshing_remove_duplicate_faces()  # faces defined by the same verts
    ms.meshing_remove_null_faces()  # faces with area == 0

    if min_d > 0:
        ms.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=pml.Percentage(min_d)
        )

    if min_f > 0:
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_f)

    if repair:
        # ms.meshing_remove_t_vertices(method=0, threshold=40, repeat=True)
        ms.meshing_repair_non_manifold_edges(method=0)
        ms.meshing_repair_non_manifold_vertices(vertdispratio=0)

    if remesh:
        # ms.apply_coord_taubin_smoothing()
        ms.meshing_isotropic_explicit_remeshing(
            iterations=3, targetlen=pml.AbsoluteValue(remesh_size)
        )

    # extract mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(
        f"[INFO] mesh cleaning: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}"
    )

    return verts, faces

def construct_list_of_gs_attributes(features_dc, features_rest, scaling, rotation):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(features_dc.shape[1]*features_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(features_rest.shape[1]*features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l

def calculate_max_sh_degree_from_gs_ply(plydata):
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    #assert len(extra_f_names)!=3*(max_sh_degree + 1) ** 2 - 3:
    max_sh_degree = int(((len(extra_f_names) + 3) / 3) ** 0.5 - 1)
    return max_sh_degree, extra_f_names

def write_gs_ply(xyz, normals, f_dc, f_rest, opacities, scale, rotation, list_of_attributes):
    dtype_full = [(attribute, 'f4') for attribute in list_of_attributes]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    return PlyData([el])

def read_gs_ply(plydata):
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    max_sh_degree, extra_f_names = calculate_max_sh_degree_from_gs_ply(plydata)

    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
    return xyz, features_dc, features_extra, opacities, scales, rots

def ply_to_points_cloud(plydata):
    xyz, features_dc, features_extra, opacities, scales, rots = read_gs_ply(plydata)
    
    features_dc = np.transpose(features_dc, (0, 2, 1))  # equivalent of torch.transpose(features_dc, 1, 2)
    features_extra = np.transpose(features_extra, (0, 2, 1))
    shs = np.concatenate((features_dc, features_extra), axis=1)
    normals = np.zeros_like(xyz)
    pcd = PointCloud(points=xyz, colors=SH2RGB(shs), normals=normals)
    return pcd


def get_target_axis_and_scale(axis_string, scale_value=1.0):
    """
    Coordinate system inverts when:
    1. Any of the axis inverts
    2. Two of the axises switch
    
    If coordinate system inverts twice in a row then it will not be inverted
    """
    axis_names = ["x", "y", "z"]
    
    target_axis, target_scale, coordinate_invert_count = [], [], 0
    axis_switch_count = 0
    for i in range(len(axis_names)):
        s = axis_string[i]
        if s[0] == "-":
            target_scale.append(-scale_value)
            coordinate_invert_count += 1
        else:
            target_scale.append(scale_value)
            
        new_axis_i = axis_names.index(s[1])
        if new_axis_i != i:
            axis_switch_count += 1
        target_axis.append(new_axis_i)
        
    if axis_switch_count == 2:
        coordinate_invert_count += 1
        
    return target_axis, target_scale, coordinate_invert_count

def switch_vector_axis(vector3s, target_axis):
    """
    Example:
        vector3s = torch.tensor([[1, 2, 3], [3, 2, 1], [2, 3, 1]])  # shape (N, 3)

        target_axis = (2, 0, 1) # or [2, 0, 1]
        vector3s[:, [0, 1, 2]] = vector3s[:, target_axis]
        
        # Result: tensor([[3, 1, 2], [1, 3, 2], [1, 2, 3]])
    """
    vector3s[:, [0, 1, 2]] = vector3s[:, target_axis]
    return vector3s

def switch_ply_axis_and_scale(plydata, target_axis, target_scale, coordinate_invert_count):
    """
    Args:
        target_axis (array): shape (3)
        target_scale (array): shape (3)
    """
    xyz, features_dc, features_extra, opacities, scales, rots = read_gs_ply(plydata)
    normals = np.zeros_like(xyz)
    features_dc_2d = features_dc.reshape(features_dc.shape[0], features_dc.shape[1]*features_dc.shape[2])
    features_extra_2d = features_extra.reshape(features_extra.shape[0], features_extra.shape[1]*features_extra.shape[2])
    
    target_scale = torch.tensor(target_scale).float().cuda()
    xyz = switch_vector_axis(torch.tensor(xyz).float().cuda() * target_scale, target_axis).detach().cpu().numpy()
    scales = switch_vector_axis(torch.tensor(scales).float().cuda(), target_axis).detach().cpu().numpy()
    
    # change rotation representation from quaternion (w, x, y, z) to axis angle vector (x, y, z) to make swich axis easier
    rots_axis_angle = quaternion_to_axis_angle(torch.tensor(rots).float().cuda())
    rots_axis_angle = switch_vector_axis(rots_axis_angle * target_scale, target_axis)
    """
    Since axis–angle vector is composed of axis (unit vector/direction) and clockwise radians angle (vector magnitude),
    so in order to invert the sign of angle when coordinate system inverts, we also need to invert the direction of axis–angle vector
    """
    if coordinate_invert_count % 2 != 0:
        rots_axis_angle = -rots_axis_angle
    rots = axis_angle_to_quaternion(rots_axis_angle).detach().cpu().numpy()
    
    return write_gs_ply(xyz, normals, features_dc_2d, features_extra_2d, opacities, scales, rots, construct_list_of_gs_attributes(features_dc, features_extra, scales, rots))
    
def switch_mesh_axis_and_scale(mesh, target_axis, target_scale, flip_normal=False):
    """
    Args:
        target_axis (array): shape (3)
        target_scale (array): shape (3)
    """
    target_scale = torch.tensor(target_scale).float().cuda()
    mesh.v = switch_vector_axis(mesh.v * target_scale, target_axis)
    mesh.vn = switch_vector_axis(mesh.vn * target_scale, target_axis)
    if flip_normal:
        mesh.vn *= -1
    return mesh


def marching_cubes_density_to_mesh(get_density_func, grid_size=256, S=128, density_thresh=10, decimate_target=5e4):
    from mcubes import marching_cubes
    from kiui.mesh_utils import clean_mesh, decimate_mesh
    
    
    sigmas = np.zeros([grid_size, grid_size, grid_size], dtype=np.float32)

    X = torch.linspace(-1, 1, grid_size).split(S)
    Y = torch.linspace(-1, 1, grid_size).split(S)
    Z = torch.linspace(-1, 1, grid_size).split(S)

    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                val = get_density_func(pts)
                sigmas[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val.reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]

    print(f'[INFO] marching cubes thresh: {density_thresh} ({sigmas.min()} ~ {sigmas.max()})')

    vertices, triangles = marching_cubes(sigmas, density_thresh)
    vertices = vertices / (grid_size - 1.0) * 2 - 1
    
    # clean
    vertices = vertices.astype(np.float32)
    triangles = triangles.astype(np.int32)
    vertices, triangles = clean_mesh(vertices, triangles, remesh=True, remesh_size=0.01)
    if triangles.shape[0] > decimate_target:
        vertices, triangles = decimate_mesh(vertices, triangles, decimate_target, optimalplacement=False)
    
    return vertices, triangles

def color_func_to_albedo(mesh, get_rgb_func, texture_resolution=1024, padding=2, batch_size=640000, device="cuda", force_cuda_rast=False, glctx=None):
    import nvdiffrast.torch as dr
    from kiui.op import uv_padding
    
    if glctx is None:
        if force_cuda_rast:
            glctx = dr.RasterizeCudaContext()
        else:
            glctx = dr.RasterizeGLContext()
    
    # render uv maps
    h = w = texture_resolution
    if mesh.vt is None:
        mesh.auto_uv()

    uv = mesh.vt * 2.0 - 1.0 # uvs to range [-1, 1]
    uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1) # [N, 4]

    rast, _ = dr.rasterize(glctx, uv.unsqueeze(0), mesh.ft, (h, w)) # [1, h, w, 4]
    xyzs, _ = dr.interpolate(mesh.v.unsqueeze(0), rast, mesh.f) # [1, h, w, 3]
    mask, _ = dr.interpolate(torch.ones_like(mesh.v[:, :1]).unsqueeze(0), rast, mesh.f) # [1, h, w, 1]

    # masked query 
    xyzs = xyzs.view(-1, 3)
    mask = (mask > 0).view(-1)
    
    albedo = torch.zeros(h * w, 3, device=device, dtype=torch.float32)

    if mask.any():
        print(f"[INFO] querying texture...")

        xyzs = xyzs[mask] # [M, 3]

        # batched inference to avoid OOM
        batch = []
        head = 0
        while head < xyzs.shape[0]:
            tail = min(head + batch_size, xyzs.shape[0])
            batch.append(get_rgb_func(xyzs[head:tail]).float())
            head += batch_size

        albedo[mask] = torch.cat(batch, dim=0)
    
    albedo = albedo.view(h, w, -1)
    mask = mask.view(h, w)
    albedo = uv_padding(albedo, mask, padding)
    
    return albedo

@torch.no_grad()
def K_nearest_neighbors_func(
    points: torch.Tensor,
    K: int,
    query: Optional[torch.Tensor] = None,
    return_dist=False,
    skip_first_index=True,
):
    if not pytorch3d_capable:
        raise ImportError("pytorch3d is not installed, which is required for KNN")
    
    # query/points: Tensor of shape (N, P1/P2, D) giving a batch of N point clouds, each containing up to P1/P2 points of dimension D
    if query is None:
        query = points
    dist, idx, nn = knn_points(query[None, ...], points[None, ...], K=K, return_nn=True)

    # idx: Tensor of shape (N, P1, K)
    # nn: Tensor of shape (N, P1, K, D)

    # take the index 1 since index 0 is the point itself
    if skip_first_index:
        if not return_dist:
            return nn[0, :, 1:, :], idx[0, :, 1:]
        else:
            return nn[0, :, 1:, :], idx[0, :, 1:], dist[0, :, 1:]
    else:
        if not return_dist:
            return nn[0, :, :, :], idx[0, :, :]
        else:
            return nn[0, :, :, :], idx[0, :, :], dist[0, :, :]

def interpolate_texture_map_attr(mesh, texture_size: int = 256, batch_size: int = 64, interpolate_color=True, interpolate_position=False):
    # Get UV coordinates and faces
    if mesh.vt is None:
        mesh.auto_uv()
        
    # Get Faces on UV
    texture_size_minus_one = texture_size - 1
    faces = mesh.ft
    verts_uvs = mesh.vt * texture_size_minus_one
    verts_uvs_idx = verts_uvs.to(torch.long)

    vmapping = mesh.get_default_vt_to_v_mapping()
    # Get vertex colors
    interpolate_color = interpolate_color and mesh.vc is not None
    texture_map = mesh.albedo
    if interpolate_color:
        verts_colors = mesh.vc[vmapping]
        # Create a blank texture map
        texture_map = torch.zeros((texture_size, texture_size, 3), device=verts_colors.device)

    # Get vertex positions
    position_map = None
    if interpolate_position:
        verts_positions = mesh.v[vmapping]
        position_map = torch.zeros((texture_size, texture_size, 3), device=verts_colors.device)
    
    # Create a grid of UV coordinates
    grid_x, grid_y = torch.meshgrid(torch.linspace(0, texture_size_minus_one, texture_size, dtype=verts_uvs.dtype, device=verts_colors.device), torch.linspace(0, texture_size_minus_one, texture_size, dtype=verts_uvs.dtype, device=verts_colors.device))
    grid = torch.stack([grid_x, grid_y], dim=-1)
    
    batch_idx_range = torch.arange(0, batch_size**2, 1, device=verts_colors.device, dtype=torch.long)
    
    all_face_verts = verts_uvs_idx[faces]
    for uv_i in range(0, texture_size_minus_one, batch_size):
        for uv_j in range(0, texture_size_minus_one, batch_size):
            # Divide faces into squre batchs
            faces_mask = (uv_i <= all_face_verts[:, :, 0]) & (all_face_verts[:, :, 0] <= uv_i + batch_size) & (uv_j <= all_face_verts[:, :, 1]) & (all_face_verts[:, :, 1] <= uv_j + batch_size)
            faces_mask, _ = torch.max(faces_mask, dim=1)
            faces_mask = faces_mask.nonzero().squeeze(-1)
            batch_faces = faces[faces_mask]
            
            uvs = verts_uvs[batch_faces]
            if interpolate_color:
                colors = verts_colors[batch_faces]
            if interpolate_position:
                positions = verts_positions[batch_faces]
                        
            v0, v1, v2 = uvs[:, 0], uvs[:, 1], uvs[:, 2]
            v0_0, v0_1 = v0[:, 0], v0[:, 1]
            v1_0, v1_1 = v1[:, 0], v1[:, 1]
            v2_0, v2_1 = v2[:, 0], v2[:, 1]
            
            sub_grid = grid[uv_i:uv_i + batch_size, uv_j:uv_j + batch_size].reshape(-1, 2).unsqueeze(1)
                        
            # Compute barycentric coordinates for the batch of faces
            denom = (v1_1 - v2_1) * (v0_0 - v2_0) + (v2_0 - v1_0) * (v0_1 - v2_1)
            a = ((v1_1 - v2_1) * (sub_grid[:, :, 0] - v2_0) + (v2_0 - v1_0) * (sub_grid[:, :, 1] - v2_1)) / denom
            b = ((v2_1 - v0_1) * (sub_grid[:, :, 0] - v2_0) + (v0_0 - v2_0) * (sub_grid[:, :, 1] - v2_1)) / denom
            c = 1 - a - b
            
            # Mask for points inside the triangle
            mask = (a >= 0) & (b >= 0) & (c >= 0)
            mask, mask_idx = torch.max(mask, dim=1)
            
            # Get texture coordinates
            sub_grid = sub_grid.squeeze(1)
            uv_coords = sub_grid[mask].to(torch.long)
            
            # Interpolate colors
            a, b, c = a[batch_idx_range, mask_idx][mask], b[batch_idx_range, mask_idx][mask], c[batch_idx_range, mask_idx][mask]
            
            if interpolate_color:
                colors = colors[mask_idx][mask]
                interpolated_colors = (a[:, None] * colors[:, 0] + (b[:, None] * colors[:, 1]) + (c[:, None] * colors[:, 2]))
                texture_map[uv_coords[:, 1], uv_coords[:, 0]] = interpolated_colors
            if interpolate_position:
                positions = positions[mask_idx][mask]
                interpolated_positions = (a[:, None] * positions[:, 0] + (b[:, None] * positions[:, 1]) + (c[:, None] * positions[:, 2]))
                position_map[uv_coords[:, 1], uv_coords[:, 0]] = interpolated_positions
            
    return texture_map, position_map